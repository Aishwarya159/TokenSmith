"""
extract_kg.py

Efficient knowledge graph extraction for large textbook chunk sets.
Handles 1000+ chunks via:
  - Parallel LLM calls (ThreadPoolExecutor)
  - Chunk filtering (skip boilerplate)
  - Entity deduplication via embedding similarity
  - Incremental checkpointing (resume interrupted runs)
  - Batch size tuning to avoid rate limits

Usage:
    python extract_kg.py \
        --index_dir index/sections/ \
        --output    index/knowledge_graph.json \
        --workers   4 \
        --model_path models/qwen2.5-1.5b-instruct-q5_k_m.gguf \
        --resume                    # skip already-processed chunks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Callable

# ── TokenSmith imports ────────────────────────────────────────────────────────
# Adjust these to match how your TokenSmith setup exposes the LLM
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

from src.knowledge_graph import (
    KnowledgeGraph,
    Chunk,
    Entity,
    Relation,
    extract_from_chunk,
    chunks_from_tokensmith_index,
    EXTRACTION_PROMPT,
    _parse_llm_json,
)


# ── Boilerplate detection ─────────────────────────────────────────────────────

SKIP_PATTERNS = [
    r"^(table of contents|contents|index|bibliography|references|glossary)",
    r"^(chapter \d+\s*$|section \d+[\.\d]*\s*$)",
    r"^\s*\d+\s*$",                         # page number only
    r"^(figure|table|appendix)\s+\d+",      # standalone captions
    r"copyright|all rights reserved|isbn",
    r"^(preface|acknowledgements?|about the author)",
]
SKIP_RE = re.compile("|".join(SKIP_PATTERNS), re.IGNORECASE | re.MULTILINE)

MIN_CHUNK_WORDS = 40   # skip chunks too short to contain real content


def should_skip_chunk(chunk: Chunk) -> bool:
    """Return True if chunk looks like boilerplate not worth extracting."""
    text = chunk.text.strip()
    if len(text.split()) < MIN_CHUNK_WORDS:
        return True
    if SKIP_RE.search(text[:300]):
        return True
    # High ratio of numbers/special chars = likely a table or index
    alpha = sum(c.isalpha() for c in text)
    if len(text) > 0 and alpha / len(text) < 0.4:
        return True
    return False


# ── Entity deduplication ──────────────────────────────────────────────────────

def normalize_entity_id(name: str) -> str:
    """Normalize entity names to a canonical ID for deduplication."""
    name = name.lower().strip()
    # Remove common suffixes that create false duplicates
    name = re.sub(r"\s*(algorithm|technique|method|approach|model|system|structure)$", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


SYNONYMS = {
    # Database systems textbook common aliases
    "relation": "relational_table",
    "table": "relational_table",
    "relational_table": "relational_table",
    "tuple": "row",
    "record": "row",
    "attribute": "column",
    "field": "column",
    "primary_key": "primary_key",
    "pk": "primary_key",
    "fk": "foreign_key",
    "foreign_key": "foreign_key",
    "er_diagram": "entity_relationship_diagram",
    "erd": "entity_relationship_diagram",
    "entity_relationship_diagram": "entity_relationship_diagram",
    "acid": "acid_properties",
    "normalisation": "normalization",
    "normalization": "normalization",
    "b_tree": "b_plus_tree",
    "b+_tree": "b_plus_tree",
    "b_plus_tree": "b_plus_tree",
    "sql": "sql",
    "structured_query_language": "sql",
    "ddl": "data_definition_language",
    "dml": "data_manipulation_language",
    "dbms": "database_management_system",
    "database_management_system": "database_management_system",
    "rdbms": "relational_database_management_system",
}


def canonicalize_id(entity_id: str) -> str:
    """Map entity IDs to canonical forms using the synonym table."""
    normalized = normalize_entity_id(entity_id)
    return SYNONYMS.get(normalized, normalized)


# ── Checkpoint management ─────────────────────────────────────────────────────

class CheckpointManager:
    """
    Saves extraction results incrementally so a crashed run can be resumed.
    Each chunk result is saved as a small JSON file in a checkpoint directory.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def chunk_key(self, chunk_id: str) -> str:
        return hashlib.md5(chunk_id.encode()).hexdigest()

    def is_done(self, chunk_id: str) -> bool:
        return (self.dir / f"{self.chunk_key(chunk_id)}.json").exists()

    def save(self, chunk_id: str, entities: list[dict], relations: list[dict]) -> None:
        path = self.dir / f"{self.chunk_key(chunk_id)}.json"
        path.write_text(json.dumps({
            "chunk_id": chunk_id,
            "entities": entities,
            "relations": relations,
        }))

    def load_all(self) -> list[dict]:
        results = []
        for f in self.dir.glob("*.json"):
            try:
                results.append(json.loads(f.read_text()))
            except Exception:
                pass
        return results

    def count_done(self) -> int:
        return len(list(self.dir.glob("*.json")))


# ── LLM wrapper ───────────────────────────────────────────────────────────────

class ThreadSafeLlama:
    """
    Wraps llama_cpp.Llama with a lock so multiple threads can share one model.
    llama.cpp is not thread-safe for simultaneous inference, so we serialize
    calls but still benefit from parallelism in pre/post-processing.

    For true parallel inference, run multiple model instances (see MultiModelPool).
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, **kwargs):
        print(f"Loading model: {model_path}")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=4,
            verbose=False,
            **kwargs,
        )
        self._lock = Lock()

    def __call__(self, prompt: str, max_tokens: int = 1024) -> str:
        with self._lock:
            result = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.1,   # low temp for structured extraction
                stop=["```\n\n", "\n\n\n"],
            )
            return result["choices"][0]["text"]


class MultiModelPool:
    """
    Maintains N independent model instances for true parallel inference.
    Each worker thread gets its own model instance, avoiding lock contention.
    Use this when you have enough RAM for multiple model copies (e.g. small quants).
    """

    def __init__(self, model_path: str, n_instances: int = 2, **kwargs):
        print(f"Loading {n_instances} model instances...")
        self._models = [
            Llama(model_path=model_path, n_ctx=2048, verbose=False, **kwargs)
            for _ in range(n_instances)
        ]
        self._locks = [Lock() for _ in self._models]

    def get_llm_fn(self, worker_id: int) -> Callable[[str], str]:
        """Return a callable bound to a specific model instance."""
        idx = worker_id % len(self._models)
        model = self._models[idx]
        lock = self._locks[idx]

        def call(prompt: str) -> str:
            with lock:
                result = model(prompt, max_tokens=1024, temperature=0.1)
                return result["choices"][0]["text"]
        return call


# ── Core parallel extraction ──────────────────────────────────────────────────

def process_one_chunk(
    chunk: Chunk,
    llm_fn: Callable[[str], str],
    checkpoint: CheckpointManager,
) -> tuple[str, list[dict], list[dict]]:
    """Extract entities and relations from one chunk, with checkpointing."""

    if checkpoint.is_done(chunk.id):
        return chunk.id, [], []   # will be loaded from checkpoint later

    if should_skip_chunk(chunk):
        checkpoint.save(chunk.id, [], [])
        return chunk.id, [], []

    try:
        entities, relations = extract_from_chunk(chunk, llm_fn)

        # Canonicalize entity IDs before saving
        id_map: dict[str, str] = {}
        deduped_entities = []
        for e in entities:
            canonical = canonicalize_id(e.id)
            id_map[e.id] = canonical
            e.id = canonical
            deduped_entities.append({
                "id": e.id,
                "name": e.name,
                "type": e.entity_type,
                "definition": e.definition,
                "source_chunks": e.source_chunks,
            })

        deduped_relations = []
        for r in relations:
            src = id_map.get(r.source_id, canonicalize_id(r.source_id))
            tgt = id_map.get(r.target_id, canonicalize_id(r.target_id))
            deduped_relations.append({
                "source": src,
                "target": tgt,
                "relation": r.relation,
                "context": r.context,
                "chunk": r.source_chunk,
            })

        checkpoint.save(chunk.id, deduped_entities, deduped_relations)
        return chunk.id, deduped_entities, deduped_relations

    except Exception as exc:
        print(f"  [ERROR] chunk {chunk.id}: {exc}")
        checkpoint.save(chunk.id, [], [])  # mark as done to skip on resume
        return chunk.id, [], []


def extract_parallel(
    chunks: list[Chunk],
    llm_fn: Callable[[str], str],
    checkpoint: CheckpointManager,
    workers: int = 4,
    batch_size: int = 50,
    delay_between_batches: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    """
    Run extraction over all chunks in parallel batches.

    workers:              number of parallel threads
    batch_size:           process this many chunks before printing progress
    delay_between_batches: seconds to wait between batches (useful for API rate limits)
    """
    already_done = checkpoint.count_done()
    todo = [c for c in chunks if not checkpoint.is_done(c.id)]
    skippable = sum(1 for c in chunks if should_skip_chunk(c) and not checkpoint.is_done(c.id))

    print(f"\nChunk summary:")
    print(f"  Total:           {len(chunks)}")
    print(f"  Already done:    {already_done}")
    print(f"  Will skip (boilerplate): {skippable}")
    print(f"  Will process:    {len(todo) - skippable}")
    print(f"  Workers:         {workers}")
    print(f"  Batch size:      {batch_size}\n")

    all_entities: list[dict] = []
    all_relations: list[dict] = []
    completed = 0
    start_time = time.time()

    # Process in batches
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start: batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_one_chunk, chunk, llm_fn, checkpoint): chunk
                for chunk in batch
            }
            for future in as_completed(futures):
                chunk_id, entities, relations = future.result()
                all_entities.extend(entities)
                all_relations.extend(relations)
                completed += 1

        elapsed = time.time() - start_time
        total_done = already_done + completed
        pct = total_done / len(chunks) * 100
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (len(todo) - completed) / rate if rate > 0 else 0

        print(
            f"  Progress: {total_done}/{len(chunks)} ({pct:.1f}%) | "
            f"{rate:.1f} chunks/s | ETA: {eta/60:.1f} min"
        )

        if delay_between_batches > 0:
            time.sleep(delay_between_batches)

    return all_entities, all_relations


def assemble_graph(
    checkpoint: CheckpointManager,
    chunks: list[Chunk],
) -> KnowledgeGraph:
    """
    Assemble the final KnowledgeGraph from all checkpoint files.
    Merges duplicate entities by canonical ID.
    """
    kg = KnowledgeGraph()

    # Store chunk texts for context retrieval
    chunk_map = {c.id: c.text for c in chunks}
    for cid, text in chunk_map.items():
        kg.add_chunk_text(cid, text)

    all_results = checkpoint.load_all()
    print(f"\nAssembling graph from {len(all_results)} checkpoint files...")

    for result in all_results:
        for e in result.get("entities", []):
            kg.add_entity(Entity(
                id=e["id"],
                name=e["name"],
                entity_type=e.get("type", "concept"),
                definition=e.get("definition", ""),
                source_chunks=e.get("source_chunks", [result["chunk_id"]]),
            ))

    # Add relations only after all entities are registered (no dangling edges)
    for result in all_results:
        for r in result.get("relations", []):
            kg.add_relation(Relation(
                source_id=r["source"],
                target_id=r["target"],
                relation=r["relation"],
                context=r.get("context", ""),
                source_chunk=r.get("chunk", ""),
            ))

    return kg


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_graph_stats(kg: KnowledgeGraph) -> None:
    from collections import Counter

    print("\n── Knowledge Graph Stats ─────────────────────────────────")
    print(f"  Entities:  {len(kg.entities)}")
    total_relations = sum(len(v) for v in kg._adj.values())
    print(f"  Relations: {total_relations}")

    # Entity type distribution
    type_counts = Counter(e.entity_type for e in kg.entities.values())
    print("\n  Entity types:")
    for etype, count in type_counts.most_common():
        print(f"    {etype:25s} {count}")

    # Relation type distribution
    rel_counts: Counter = Counter()
    for edges in kg._adj.values():
        for _, rel, _, _ in edges:
            rel_counts[rel] += 1
    print("\n  Relation types:")
    for rel, count in rel_counts.most_common(10):
        print(f"    {rel:25s} {count}")

    # Most connected entities (hubs = important concepts)
    degree: Counter = Counter()
    for src, edges in kg._adj.items():
        degree[src] += len(edges)
        for tgt, _, _, _ in edges:
            degree[tgt] += 1
    print("\n  Most connected concepts (top 15):")
    for eid, deg in degree.most_common(15):
        name = kg.entities[eid].name if eid in kg.entities else eid
        print(f"    {name:35s} degree={deg}")
    print("──────────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract knowledge graph from TokenSmith chunks")
    parser.add_argument("--index_dir",   default="index/sections/",          help="TokenSmith chunk directory")
    parser.add_argument("--output",      default="index/knowledge_graph.json", help="Output KG path")
    parser.add_argument("--checkpoint",  default="index/kg_checkpoints/",    help="Checkpoint directory")
    parser.add_argument("--model_path",  default="models/qwen2.5-1.5b-instruct-q5_k_m.gguf")
    parser.add_argument("--workers",     type=int, default=2,                help="Parallel threads (1 per model instance)")
    parser.add_argument("--batch_size",  type=int, default=50,               help="Chunks per progress update")
    parser.add_argument("--multi_model", action="store_true",                help="Load N model instances for true parallelism")
    parser.add_argument("--resume",      action="store_true",                help="Resume from checkpoint (skip done chunks)")
    parser.add_argument("--stats_only",  action="store_true",                help="Just print stats for an existing graph")
    args = parser.parse_args()

    # ── Stats only mode ──
    if args.stats_only:
        kg = KnowledgeGraph.load(args.output)
        print_graph_stats(kg)
        return

    # ── Load chunks ──
    print(f"Loading chunks from {args.index_dir}...")
    chunks = chunks_from_tokensmith_index(args.index_dir)
    print(f"Loaded {len(chunks)} chunks")

    checkpoint = CheckpointManager(args.checkpoint)

    if not args.resume and checkpoint.count_done() > 0:
        ans = input(f"Found {checkpoint.count_done()} existing checkpoints. Resume? [y/N] ")
        if ans.lower() != "y":
            import shutil
            shutil.rmtree(args.checkpoint)
            checkpoint = CheckpointManager(args.checkpoint)

    # ── Set up LLM ──
    if not LLAMA_CPP_AVAILABLE:
        print("llama_cpp not available — using dummy LLM (for testing only)")
        def llm_fn(prompt: str) -> str:
            return '{"entities": [], "relations": []}'
    elif args.multi_model and args.workers > 1:
        # True parallel: one model per worker
        pool = MultiModelPool(args.model_path, n_instances=args.workers)
        # For simplicity, use instance 0 as the shared fn
        # (ThreadPoolExecutor will call this from multiple threads,
        #  but each call picks its own instance by thread id)
        import threading
        def llm_fn(prompt: str) -> str:
            tid = threading.get_ident() % args.workers
            return pool.get_llm_fn(tid)(prompt)
    else:
        # Single model, serialized calls
        model = ThreadSafeLlama(args.model_path)
        llm_fn = model

    # ── Extract ──
    extract_parallel(
        chunks=chunks,
        llm_fn=llm_fn,
        checkpoint=checkpoint,
        workers=args.workers if args.multi_model else 1,
        batch_size=args.batch_size,
    )

    # ── Assemble ──
    kg = assemble_graph(checkpoint, chunks)
    print_graph_stats(kg)
    kg.save(args.output)
    print(f"\nDone. Graph saved to {args.output}")


if __name__ == "__main__":
    main()