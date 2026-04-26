# noinspection PyUnresolvedReferences
import faiss  # force single OpenMP init

import argparse
import json
import pathlib
import sys
from typing import Dict, Optional, List, Tuple, Union, Any

from rich.live import Live
from rich.console import Console
from rich.markdown import Markdown

from src.config import RAGConfig
from src.generator import answer, double_answer, dedupe_generated_text
from src.index_builder import build_index

from src.instrumentation.logging import get_logger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.query_enhancement import (
    generate_hypothetical_document,
    contextualize_query,
    decompose_complex_query,
)
from src.retriever import (
    filter_retrieved_chunks,
    BM25Retriever,
    FAISSRetriever,
    IndexKeywordRetriever,
    KnowledgeGraphRetriever,
    get_page_numbers,
    load_artifacts,
)
from src.ranking.reranker import rerank

ANSWER_NOT_FOUND = "I'm sorry, but I don't have enough information to answer that question."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Welcome to TokenSmith!")
    parser.add_argument("mode", choices=["index", "chat"], help="operation mode")
    parser.add_argument("--pdf_dir", default="data/chapters/", help="directory containing PDF files")
    parser.add_argument("--index_prefix", default="textbook_index", help="prefix for generated index files")
    parser.add_argument("--model_path", help="path to generation model")
    parser.add_argument(
        "--system_prompt_mode",
        choices=["baseline", "tutor", "concise", "detailed"],
        default="baseline",
    )

    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument("--keep_tables", action="store_true")
    indexing_group.add_argument("--multiproc_indexing", action="store_true")
    indexing_group.add_argument("--embed_with_headings", action="store_true")

    parser.add_argument(
        "--double_prompt",
        action="store_true",
        help="enable double prompting for higher quality answers",
    )
    parser.add_argument(
        "--multihop",
        action="store_true",
        default=None,
        help="force-enable multihop retrieval (overrides config)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Index mode
# ---------------------------------------------------------------------------

def run_index_mode(args: argparse.Namespace, cfg: RAGConfig):

    strategy = cfg.get_chunk_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    artifacts_dir = cfg.get_artifacts_directory()

    data_dir = pathlib.Path("data")
    print(f"Looking for markdown files in {data_dir.resolve()}...")
    md_files = sorted(data_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    print(f"First 5 markdown files: {[str(f) for f in md_files[:5]]}")

    if not md_files:
        print("ERROR: No markdown files found in data/.", file=sys.stderr)
        sys.exit(1)

    build_index(
        markdown_file=str(md_files[0]),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        use_multiprocessing=args.multiproc_indexing,
        use_headings=args.embed_with_headings,
    )


# ---------------------------------------------------------------------------
# Index-keyword helper (unchanged)
# ---------------------------------------------------------------------------

def use_indexed_chunks(question: str, chunks: list) -> list:
    try:
        with open("index/sections/textbook_index_page_to_chunk_map.json", "r") as f:
            page_to_chunk_map = json.load(f)
        with open("data/extracted_index.json", "r") as f:
            extracted_index = json.load(f)
    except FileNotFoundError:
        return [], []

    keywords = get_keywords(question)
    chunk_ids = {
        chunk_id
        for word in keywords
        if word in extracted_index
        for page_no in extracted_index[word]
        for chunk_id in page_to_chunk_map.get(str(page_no), [])
    }
    return [chunks[cid] for cid in chunk_ids], list(chunk_ids)


def get_keywords(question: str) -> list:
    stopwords = {
        "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in",
        "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", "what",
    }
    words = question.lower().split()
    return [word.strip(".,!?()[]") for word in words if word not in stopwords]


# ---------------------------------------------------------------------------
# Multihop retrieval helper
# ---------------------------------------------------------------------------

def _multihop_retrieve(
    question: str,
    cfg: RAGConfig,
    chunks: list,
    retrievers: list,
    ranker: "EnsembleRanker",
) -> Tuple[List[str], List[int], Dict[str, Dict[int, float]]]:
    """
    Decompose a complex question into sub-questions, run the full retriever
    stack for each, then merge by taking the max score per (retriever, chunk)
    pair before ranking.  This surfaces "bridge" chunks that are strongly
    relevant to one sub-question but not the original query as a whole.
    """
    pool_n = max(cfg.num_candidates, cfg.top_k + 10)

    sub_questions = decompose_complex_query(question, cfg.gen_model)

    # Always include the original question; deduplicate case-insensitively
    all_queries: List[str] = [question] + [
        sq for sq in sub_questions if sq.lower().strip() != question.lower().strip()
    ]
    print(f"[Multihop] {len(all_queries)} retrieval queries: {all_queries}")

    # merged_raw[retriever_name][chunk_idx] = max score seen across sub-questions
    merged_raw: Dict[str, Dict[int, float]] = {}

    for query in all_queries:
        for retriever in retrievers:
            sub_scores = retriever.get_scores(query, pool_n, chunks)
            bucket = merged_raw.setdefault(retriever.name, {})
            for idx, score in sub_scores.items():
                if score > bucket.get(idx, 0.0):
                    bucket[idx] = score

    ordered, scores = ranker.rank(raw_scores=merged_raw)
    topk_idxs = filter_retrieved_chunks(cfg, chunks, ordered)
    ranked_chunks = [chunks[i] for i in topk_idxs]

    return ranked_chunks, topk_idxs, merged_raw


# ---------------------------------------------------------------------------
# Core answer function
# ---------------------------------------------------------------------------

def get_answer(
    question: str,
    cfg: RAGConfig,
    args: argparse.Namespace,
    logger: Any,
    console: Optional["Console"],
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None,
    is_test_mode: bool = False,
    additional_log_info: Optional[Dict[str, Any]] = None,
) -> Union[str, Tuple[str, List[Dict[str, Any]], Optional[str]]]:
    """Run a single query through the full pipeline."""

    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]

    # Ensure locals exist for all control-flow paths
    ranked_chunks: List[str] = []
    topk_idxs: List[int] = []
    scores: List = []
    raw_scores: Dict[str, Dict[int, float]] = {}
    chunks_info = None
    hyde_query = None

    # ------------------------------------------------------------------
    # Step 1: obtain chunks
    # ------------------------------------------------------------------
    if golden_chunks and cfg.use_golden_chunks:
        ranked_chunks = golden_chunks

    elif cfg.disable_chunks:
        ranked_chunks = []

    elif cfg.use_indexed_chunks:
        ranked_chunks, topk_idxs = use_indexed_chunks(question, chunks)

    else:
        # ── Query preparation ──────────────────────────────────────────
        retrieval_query = question
        if cfg.use_hyde:
            retrieval_query = generate_hypothetical_document(
                question, cfg.gen_model, max_tokens=cfg.hyde_max_tokens
            )
            hyde_query = retrieval_query

        # ── Decide whether to use multihop decomposition ───────────────
        # Priority: CLI flag > config flag > False
        cli_multihop = getattr(args, "multihop", None)
        cfg_multihop = getattr(cfg, "use_multihop", False)
        kg_retriever = next((r for r in retrievers if r.name == "kg"), None)
        kg_available = kg_retriever is not None and kg_retriever.kg is not None
        use_multihop = kg_available and (
            cli_multihop if cli_multihop is not None else cfg_multihop
        )

        if use_multihop:
            # ── Multihop path ──────────────────────────────────────────
            ranked_chunks, topk_idxs, raw_scores = _multihop_retrieve(
                question=retrieval_query,
                cfg=cfg,
                chunks=chunks,
                retrievers=retrievers,
                ranker=ranker,
            )
            # Reconstruct scores list aligned to topk_idxs for logging
            ensemble_scores = {
                idx: sum(
                    raw_scores.get(r.name, {}).get(idx, 0.0) for r in retrievers
                )
                for idx in topk_idxs
            }
            scores = [ensemble_scores.get(idx, 0.0) for idx in topk_idxs]
            # After raw_scores are collected, before rerank
            if use_multihop:
                # Get FAISS-only top chunks for comparison
                faiss_only_ordered, _ = ranker.rank(
                    raw_scores={"faiss": raw_scores.get("faiss", {})}
                )
                faiss_only_idxs = filter_retrieved_chunks(cfg, chunks, faiss_only_ordered)
                
                # Get KG-only chunks
                kg_only_idxs = set()
                kg_scores = raw_scores.get("kg", {})
                if kg_scores:
                    kg_ordered = sorted(kg_scores.items(), key=lambda x: x[1], reverse=True)
                    kg_only_idxs = {idx for idx, _ in kg_ordered[:cfg.top_k]}
                
                # Find chunks KG added that FAISS missed
                kg_added = kg_only_idxs - set(faiss_only_idxs)
                kg_removed = set(faiss_only_idxs) - set(topk_idxs)
                
                additional_log_info.update({
                    "faiss_only_idxs": faiss_only_idxs,
                    "kg_only_idxs": list(kg_only_idxs),
                    "kg_added_chunks": [
                        {
                            "idx": idx,
                            "kg_score": kg_scores.get(idx, 0),
                            "chunk": chunks[idx][:200],
                        }
                        for idx in kg_added
                    ],
                    "faiss_only_chunks": [
                        {
                            "idx": idx,
                            "chunk": chunks[idx][:200],
                        }
                        for idx in kg_removed
                    ],
                })
        else:
            # ── Standard single-query path ─────────────────────────────
            pool_n = max(cfg.num_candidates, cfg.top_k + 10)
            for retriever in retrievers:
                raw_scores[retriever.name] = retriever.get_scores(
                    retrieval_query, pool_n, chunks
                )
            ordered, scores = ranker.rank(raw_scores=raw_scores)
            topk_idxs = filter_retrieved_chunks(cfg, chunks, ordered)
            ranked_chunks = [chunks[i] for i in topk_idxs]

        # ── Test-mode chunk diagnostics ────────────────────────────────
        if is_test_mode:
            faiss_scores = raw_scores.get("faiss", {})
            bm25_scores  = raw_scores.get("bm25", {})
            index_scores = raw_scores.get("index_keywords", {})
            kg_scores    = raw_scores.get("kg", {})

            def _ranks(score_dict: dict) -> dict:
                ranked = sorted(score_dict, key=score_dict.get, reverse=True)
                return {idx: rank + 1 for rank, idx in enumerate(ranked)}

            faiss_ranks = _ranks(faiss_scores)
            bm25_ranks  = _ranks(bm25_scores)
            index_ranks = _ranks(index_scores)
            kg_ranks    = _ranks(kg_scores)

            chunks_info = [
                {
                    "rank":        rank,
                    "chunk_id":    idx,
                    "content":     chunks[idx],
                    "faiss_score": faiss_scores.get(idx, 0),
                    "faiss_rank":  faiss_ranks.get(idx, 0),
                    "bm25_score":  bm25_scores.get(idx, 0),
                    "bm25_rank":   bm25_ranks.get(idx, 0),
                    "index_score": index_scores.get(idx, 0),
                    "index_rank":  index_ranks.get(idx, 0),
                    "kg_score":    kg_scores.get(idx, 0),
                    "kg_rank":     kg_ranks.get(idx, 0),
                }
                for rank, idx in enumerate(topk_idxs, 1)
            ]

        # ── Final re-ranking ───────────────────────────────────────────
        ranked_chunks = rerank(
            question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.rerank_top_k
        )

    # ------------------------------------------------------------------
    # Guard: no chunks found
    # ------------------------------------------------------------------
    if not ranked_chunks and not cfg.disable_chunks:
        if console:
            console.print(f"\n{ANSWER_NOT_FOUND}\n")
        if is_test_mode:
            return ANSWER_NOT_FOUND, chunks_info, hyde_query
        return ANSWER_NOT_FOUND

    # ------------------------------------------------------------------
    # Step 2: generation
    # ------------------------------------------------------------------
    model_path    = cfg.gen_model
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode
    use_double    = getattr(args, "double_prompt", False) or cfg.use_double_prompt

    stream_iter = (
        double_answer(
            question, ranked_chunks, model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )
        if use_double
        else answer(
            question, ranked_chunks, model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )
    )

    if is_test_mode:
        ans = dedupe_generated_text("".join(stream_iter))
        return ans, chunks_info, hyde_query

    # ------------------------------------------------------------------
    # Step 3: stream + log
    # ------------------------------------------------------------------
    ans = render_streaming_ans(console, stream_iter)

    meta     = artifacts.get("meta", [])
    page_nums = get_page_numbers(topk_idxs, meta)
    logger.save_chat_log(
        query=question,
        config_state=cfg.get_config_state(),
        ordered_scores=scores[: len(topk_idxs)],
        chat_request_params={
            "system_prompt": system_prompt,
            "max_tokens":    cfg.max_gen_tokens,
        },
        top_idxs=topk_idxs,
        chunks=[chunks[i] for i in topk_idxs],
        sources=[sources[i] for i in topk_idxs],
        page_map=page_nums,
        full_response=ans,
        top_k=len(topk_idxs),
        additional_log_info=additional_log_info,
    )
    return ans


# ---------------------------------------------------------------------------
# Streaming renderer
# ---------------------------------------------------------------------------

def render_streaming_ans(console: "Console", stream_iter) -> str:
    ans      = ""
    is_first = True
    with Live(console=console, refresh_per_second=8) as live:
        for delta in stream_iter:
            if is_first:
                console.print("\n[bold cyan]=== START OF ANSWER ===[/bold cyan]\n")
                is_first = False
            ans += delta
            live.update(Markdown(ans))
    ans = dedupe_generated_text(ans)
    live.update(Markdown(ans))
    console.print("\n[bold cyan]=== END OF ANSWER ===[/bold cyan]\n")
    return ans


# ---------------------------------------------------------------------------
# Chat session
# ---------------------------------------------------------------------------

def run_chat_session(args: argparse.Namespace, cfg: RAGConfig):
    logger  = get_logger()
    console = Console()

    print("Initializing TokenSmith Chat...")
    try:
        artifacts_dir = cfg.get_artifacts_directory()
        faiss_idx, bm25_idx, chunks, sources, meta, kg = load_artifacts(
            artifacts_dir, args.index_prefix
        )
        print(f"Loaded {len(chunks)} chunks and {len(sources)} sources from artifacts.")

        retrievers: list = [
            FAISSRetriever(faiss_idx, cfg.embed_model),
            BM25Retriever(bm25_idx),
            KnowledgeGraphRetriever(kg, meta),   # kg=None is safe → returns {}
        ]
        if cfg.ranker_weights.get("index_keywords", 0) > 0:
            retrievers.append(
                IndexKeywordRetriever(
                    cfg.extracted_index_path, cfg.page_to_chunk_map_path
                )
            )

        ranker = EnsembleRanker(
            ensemble_method=cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k),
        )
        print("Loaded retrievers and initialized ranker.")
        if kg is not None:
            test_scores = KnowledgeGraphRetriever(kg, meta).get_scores(
                "What is a primary key?", 10, chunks
            )
            print(f"[KG sanity check] scored {len(test_scores)} chunks for test query")
            if not test_scores:
                print("WARNING: KG retriever returning empty scores — check chunk ID mapping")
        if kg is not None:
            print(
                f"Knowledge graph loaded: {len(kg.entities)} entities, "
                f"{sum(len(v) for v in kg._adj.values())} relations."
            )
            multihop_active = getattr(args, "multihop", None) or getattr(cfg, "use_multihop", False)
            print(f"Multihop retrieval: {'ENABLED' if multihop_active else 'disabled'}.")
        else:
            print("No knowledge graph found — KG retriever will be skipped.")

        artifacts = {
            "chunks":     chunks,
            "sources":    sources,
            "retrievers": retrievers,
            "ranker":     ranker,
            "meta":       meta,
        }
    except Exception as e:
        print(f"ERROR: {e}. Run 'index' mode first.")
        sys.exit(1)

    chat_history: List[Dict[str, str]] = []
    additional_log_info: Dict[str, Any] = {}

    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            effective_q = q
            if cfg.enable_history and chat_history:
                try:
                    effective_q = contextualize_query(q, chat_history, cfg.gen_model)
                    additional_log_info.update({
                        "is_contextualizing_query": True,
                        "contextualized_query":     effective_q,
                        "original_query":           q,
                        "chat_history":             chat_history,
                    })
                    print(f"Contextualized query: {effective_q}")
                except Exception as e:
                    print(f"Warning: query contextualization failed ({e}). Using original.")
                    effective_q = q

            ans = get_answer(
                effective_q, cfg, args, logger, console,
                artifacts=artifacts,
                additional_log_info=additional_log_info,
            )

            try:
                chat_history += [
                    {"role": "user",      "content": q},
                    {"role": "assistant", "content": ans},
                ]
            except Exception as e:
                print(f"Warning: failed to update chat history: {e}")

            # Trim to context window budget
            if len(chat_history) > cfg.max_history_turns * 2:
                chat_history = chat_history[-cfg.max_history_turns * 2:]

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config_path = pathlib.Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config/config.yaml not found.")
    cfg = RAGConfig.from_yaml(config_path)
    print(f"Loaded configuration from {config_path.resolve()}.")

    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)


if __name__ == "__main__":
    main()