"""
retriever.py

Stores core retrieval logic using FAISS and BM25 scoring.
It also contains helpers for loading artifacts and filtering chunks.
"""

from __future__ import annotations

import pathlib
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import nltk
from nltk.stem import WordNetLemmatizer

import faiss
import numpy as np
from src.embedder import CachedEmbedder
from src.knowledge_graph import KnowledgeGraph

from src.config import RAGConfig
from src.index_builder import preprocess_for_bm25

# -------------------------- Embedder cache ------------------------------

_EMBED_CACHE: Dict[str, CachedEmbedder] = {}

def _get_embedder(model_name: str) -> CachedEmbedder:
    if model_name not in _EMBED_CACHE:
        _EMBED_CACHE[model_name] = CachedEmbedder(model_name)
    return _EMBED_CACHE[model_name]


# -------------------------- Read artifacts -------------------------------

def load_artifacts(
    artifacts_dir: os.PathLike,
    index_prefix: str,
) -> Tuple[faiss.Index, Any, List[str], List[str], Any, Any]:
    """
    Loads:
      - FAISS index:  {index_prefix}.faiss
      - BM25 index:   {index_prefix}_bm25.pkl
      - chunks:       {index_prefix}_chunks.pkl
      - sources:      {index_prefix}_sources.pkl
      - metadata:     {index_prefix}_meta.pkl
      - KG:           knowledge_graph.json (preferred) or {index_prefix}_kg.pkl (fallback)
    """
    artifacts_dir = pathlib.Path(artifacts_dir)

    faiss_index = faiss.read_index(str(artifacts_dir / f"{index_prefix}.faiss"))
    bm25_index  = pickle.load(open(artifacts_dir / f"{index_prefix}_bm25.pkl",   "rb"))
    chunks      = pickle.load(open(artifacts_dir / f"{index_prefix}_chunks.pkl",  "rb"))
    sources     = pickle.load(open(artifacts_dir / f"{index_prefix}_sources.pkl", "rb"))
    metadata    = pickle.load(open(artifacts_dir / f"{index_prefix}_meta.pkl",    "rb"))

    # KG: prefer JSON (portable) → fall back to pkl (legacy)
    kg = None
    kg_json = artifacts_dir / f"{index_prefix}_kg.json"
    kg_pkl  = artifacts_dir / f"{index_prefix}_kg.pkl"

    if kg_json.exists():
        kg = KnowledgeGraph.load(str(kg_json))
    elif kg_pkl.exists():
        with open(kg_pkl, "rb") as f:
            kg = pickle.load(f)

    return faiss_index, bm25_index, chunks, sources, metadata, kg


# -------------------------- Helper to get page nums for chunks -----------

def get_page_numbers(
    chunk_indices: list[int],
    metadata: list[dict],
) -> dict[int, List[int]]:
    if not metadata or not chunk_indices:
        return {}

    page_map: dict[int, List[int]] = {}
    for chunk_idx in chunk_indices:
        chunk_idx = int(chunk_idx)
        if 0 <= chunk_idx < len(metadata):
            chunk_pages = metadata[chunk_idx].get("page_numbers")
            if chunk_pages is None:
                continue
            page_map[chunk_idx] = chunk_pages

    return page_map

# -------------------------- Filtering logic ------------------------------

def filter_retrieved_chunks(cfg: RAGConfig, chunks, ordered):
    return ordered[:cfg.top_k]

# -------------------------- Retrieval core -------------------------------

class Retriever(ABC):
    @abstractmethod
    def get_scores(self, query: str, pool_size: int, chunks: List[str]):
        """Return a {chunk_index: score} dict for the top pool_size chunks."""
        pass


class FAISSRetriever(Retriever):
    name = "faiss"

    def __init__(self, index, embed_model: str):
        self.index   = index
        self.embedder = _get_embedder(embed_model)

    def get_scores(self, query: str, pool_size: int, chunks: List[str]) -> Dict[int, float]:
        q_vec = self.embedder.encode([query]).astype("float32")

        if q_vec.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dim mismatch: index={self.index.d} vs query={q_vec.shape[1]}"
            )

        distances, indices = self.index.search(q_vec, pool_size)
        cand_idxs = [i for i in indices[0] if 0 <= i < len(chunks)]
        dists = {
            idx: float(dist)
            for idx, dist in zip(cand_idxs, distances[0][: len(cand_idxs)])
        }
        return {idx: 1.0 / (1.0 + dist) for idx, dist in dists.items()}


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, index):
        self.index = index

    def get_scores(self, query: str, pool_size: int, chunks: List[str]) -> Dict[int, float]:
        tokenized_query = preprocess_for_bm25(query)
        all_scores      = self.index.get_scores(tokenized_query)

        num_candidates = min(pool_size, len(all_scores))
        top_k_indices  = np.argpartition(-all_scores, kth=num_candidates - 1)[:num_candidates]
        top_k_indices  = [i for i in top_k_indices if 0 <= i < len(chunks)]
        top_scores     = all_scores[top_k_indices]

        return {int(idx): float(score) for idx, score in zip(top_k_indices, top_scores)}


class IndexKeywordRetriever(Retriever):
    name = "index_keywords"

    def __init__(
        self,
        extracted_index_path: os.PathLike,
        page_to_chunk_map_path: os.PathLike,
    ):
        import json
        nltk.download("wordnet", quiet=True)
        self.page_to_chunk_map = {}

        if os.path.exists(extracted_index_path):
            lemmatizer = WordNetLemmatizer()
            with open(extracted_index_path, "r") as f:
                raw_index = json.load(f)

            self.phrase_to_pages: dict = {}
            self.token_to_phrases: dict = {}

            for key, pages in raw_index.items():
                words = key.lower().split()
                lemmatized_words = [
                    self._lemmatize_word(w.strip('.,!?()[]:"\''), lemmatizer)
                    for w in words
                    if w.strip('.,!?()[]:"\'')
                ]
                lemmatized_phrase = " ".join(lemmatized_words)
                self.phrase_to_pages[lemmatized_phrase] = pages
                for token in lemmatized_words:
                    self.token_to_phrases.setdefault(token, []).append(lemmatized_phrase)
        else:
            self.phrase_to_pages  = {}
            self.token_to_phrases = {}

        if os.path.exists(page_to_chunk_map_path):
            with open(page_to_chunk_map_path, "r") as f:
                self.page_to_chunk_map = json.load(f)

    def get_scores(self, query: str, pool_size: int, chunks: List[str]) -> Dict[int, float]:
        keywords        = self._extract_keywords(query)
        chunk_hit_counts: Dict[int, int] = {}

        for keyword in keywords:
            if keyword not in self.token_to_phrases:
                continue
            for phrase in self.token_to_phrases[keyword]:
                for page_no in self.phrase_to_pages[phrase]:
                    for chunk_id in self.page_to_chunk_map.get(str(page_no), []):
                        if 0 <= chunk_id < len(chunks):
                            chunk_hit_counts[chunk_id] = chunk_hit_counts.get(chunk_id, 0) + 1

        if not chunk_hit_counts:
            return {}

        max_hits = max(chunk_hit_counts.values())
        return {
            chunk_id: float(hit_count) / max_hits
            for chunk_id, hit_count in chunk_hit_counts.items()
        }

    @staticmethod
    def _lemmatize_word(word: str, lemmatizer) -> str:
        lemma = lemmatizer.lemmatize(word, pos="n")
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, pos="v")
        return lemma

    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        stopwords = {
            "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in",
            "to", "of", "by", "with", "that", "this", "it", "as", "are", "was",
            "what", "how", "why", "when", "where", "who", "does", "do", "be",
        }
        lemmatizer = WordNetLemmatizer()
        keywords   = []
        for word in query.lower().split():
            cleaned = word.strip('.,!?()[]:"\'')
            if not cleaned or cleaned in stopwords:
                continue
            keywords.append(IndexKeywordRetriever._lemmatize_word(cleaned, lemmatizer))
        return keywords


class KnowledgeGraphRetriever(Retriever):
    name = "kg"

    def __init__(self, kg: KnowledgeGraph, metadata: list[dict]):
        self.kg       = kg
        self.metadata = metadata

        # Build chunk_id string → metadata list index map
        # Handles both bare numeric IDs ("512") and prefixed IDs ("chunk_0512")
        self._chunk_id_to_meta_idx: dict[str, int] = {}
        for idx, meta in enumerate(metadata):
            cid     = meta.get("chunk_id") or meta.get("id") or str(idx)
            cid_str = str(cid)
            self._chunk_id_to_meta_idx[cid_str] = idx
            try:
                numeric = int(cid_str)
                self._chunk_id_to_meta_idx[f"chunk_{numeric:04d}"] = idx
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Main scoring entry point
    # ------------------------------------------------------------------

    def get_scores(self, query: str, pool_size: int, chunks: list[str]) -> dict[int, float]:
        if self.kg is None:
            return {}

        # ── Step 1: find seed entities ─────────────────────────────────
        seed_entities = self.kg.find_entities_by_name(query, top_k=5)
        if not seed_entities:
            return {}

        seed_ids = [e.id for e in seed_entities]
        print(f"[KG] Seed entities: {[e.name for e in seed_entities]}")

        # ── Step 2: expand subgraph ────────────────────────────────────
        subgraph_ids = self.kg.subgraph_around(seed_ids, hops=1)
        subgraph_set = set(subgraph_ids)

        # ── Step 3: pre-compute ALL hop distances in one pass ──────────
        # Avoids running a full BFS for every entity on every edge
        hop_distances: dict[str, int] = {}
        for eid in subgraph_ids:
            hop_distances[eid] = self._hop_distance(eid, seed_ids)

        # ── Step 4: score chunks by entity hop distance ─────────────────
        raw_chunk_scores: dict[str, float] = {}

        for eid in subgraph_ids:
            entity = self.kg.entities.get(eid)
            if entity is None:
                continue
            hop           = hop_distances[eid]
            entity_weight = 1.0 / (2.0 ** hop)  # hop 0 → 1.0, hop 1 → 0.5
            # First source chunk only — prevents over-extracted entities
            # (e.g. "atomicity" from 20 chapters) flooding the score map
            for cid in entity.source_chunks[:1]:
                raw_chunk_scores[cid] = raw_chunk_scores.get(cid, 0.0) + entity_weight

        # ── Step 5: bridge boost — deduplicated per chunk_id ───────────
        # Track which chunk_ids have already received a bridge bonus to
        # prevent the same chunk being boosted multiple times when the LLM
        # extracted several relations from it in the same chunk
        seen_bridge_chunks: set[str] = set()

        for src_id in subgraph_set:
            src_hop = hop_distances.get(src_id, 3)
            for tgt_id, relation, context, chunk_id in self.kg.neighbors(src_id):
                if tgt_id not in subgraph_set or not chunk_id:
                    continue
                if chunk_id in seen_bridge_chunks:
                    continue  # already boosted this chunk, skip

                tgt_hop      = hop_distances.get(tgt_id, 3)
                # Higher bonus when bridging nodes at different hop distances
                # — that is the actual multihop case (seed → hop-1 neighbor)
                bridge_bonus = 0.75 if src_hop != tgt_hop else 0.25
                raw_chunk_scores[chunk_id] = (
                    raw_chunk_scores.get(chunk_id, 0.0) + bridge_bonus
                )
                seen_bridge_chunks.add(chunk_id)

        if not raw_chunk_scores:
            return {}

        # ── Step 6: map chunk string IDs → integer indices ──────────────
        scores: dict[int, float] = {}
        for cid, raw_score in raw_chunk_scores.items():
            idx = self._chunk_id_to_meta_idx.get(cid)
            if idx is not None and 0 <= idx < len(chunks):
                scores[idx] = scores.get(idx, 0.0) + raw_score

        if not scores:
            return {}

        # ── Step 7: normalize to [0, 1] ─────────────────────────────────
        max_score = max(scores.values())
        scores    = {k: v / max_score for k, v in scores.items()}

        # ── Step 8: filter out low-confidence chunks ─────────────────────
        # Removes noise from weakly-matched entities that scraped a 
        # tiny score from a single distant hop
        MIN_KG_SCORE = 0.3
        scores = {k: v for k, v in scores.items() if v >= MIN_KG_SCORE}

        print(
            f"[KG] Scored {len(scores)} chunks "
            f"(seeds={len(seed_ids)}, subgraph={len(subgraph_ids)})"
        )
        return scores
    
    def _hop_distance(self, entity_id: str, seed_ids: list[str]) -> int:
        """
        Shortest hop count from any seed entity to entity_id via _adj.
        Returns 0 if entity_id is itself a seed, capped at 3 otherwise.
        """
        if entity_id in seed_ids:
            return 0

        visited  = set(seed_ids)
        frontier = set(seed_ids)

        for hop in range(1, 4):
            next_frontier: set[str] = set()
            for src in frontier:
                for tgt_id, _, _, _ in self.kg.neighbors(src):
                    if tgt_id == entity_id:
                        return hop
                    if tgt_id not in visited:
                        next_frontier.add(tgt_id)
                        visited.add(tgt_id)
            frontier = next_frontier
            if not frontier:
                break

        return 3