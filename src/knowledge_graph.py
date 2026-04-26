"""
knowledge_graph.py

Extracts a knowledge graph from TokenSmith textbook chunks and supports
multihop reasoning queries over it.

Fits into the TokenSmith pipeline like so:

    index (chunks) --> extract_graph() --> KnowledgeGraph
                                               |
                                          multihop_query()

Usage:
    from src.knowledge_graph import KnowledgeGraph, build_graph_from_chunks

    # After TokenSmith has indexed your PDFs, load the chunks and build the graph
    chunks = load_chunks("index/sections/")   # your existing chunk loader
    kg = build_graph_from_chunks(chunks, llm_fn=your_llm_fn)
    kg.save("index/knowledge_graph.json")

    # Later, at query time
    kg = KnowledgeGraph.load("index/knowledge_graph.json")
    answer = kg.multihop_query("What causes buffer overflow and how does the OS handle it?")
"""

from __future__ import annotations

import json
import re
import textwrap
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Optional
import pickle

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A concept, term, or named entity extracted from a chunk."""
    id: str                         # unique slug, e.g. "buffer_overflow"
    name: str                       # canonical display name
    entity_type: str                # e.g. "concept", "algorithm", "data_structure"
    definition: str = ""            # one-sentence definition from the source text
    source_chunks: list[str] = field(default_factory=list)  # chunk IDs where seen


@dataclass
class Relation:
    """A directed relationship between two entities."""
    source_id: str                  # Entity.id
    target_id: str                  # Entity.id
    relation: str                   # e.g. "causes", "is_part_of", "requires", "leads_to"
    context: str = ""               # sentence from the chunk that supports this relation
    source_chunk: str = ""          # chunk ID


@dataclass
class Chunk:
    """A text chunk from TokenSmith's indexer."""
    id: str
    text: str
    source_doc: str = ""
    page: int = 0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a knowledge graph extractor for a computer science textbook.

Given the following textbook chunk, extract:
1. Important entities (concepts, algorithms, data structures, OS components, etc.)
2. Relationships between those entities

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{
      "id": "snake_case_unique_id",
      "name": "Display Name",
      "type": "concept|algorithm|data_structure|component|process|property",
      "definition": "One sentence definition from the text"
    }}
  ],
  "relations": [
    {{
      "source": "entity_id",
      "target": "entity_id",
      "relation": "causes|requires|is_part_of|leads_to|implements|uses|defines|enables|prevents|contrasts_with",
      "context": "The exact sentence or phrase from the text supporting this relation"
    }}
  ]
}}

Rules:
- Only extract entities that are clearly defined or explained in this chunk
- Only extract relations that are explicitly stated, not inferred
- Use snake_case for entity IDs
- Keep definitions to one sentence
- If no clear entities or relations exist, return empty lists

Chunk (from {source_doc}, page {page}):
---
{text}
---

JSON output:"""


MULTIHOP_PROMPT = """You are a reasoning assistant with access to a knowledge graph extracted from a textbook.

Question: {question}

Relevant knowledge graph context (entities and their relationships):
{graph_context}

Supporting text from the original textbook:
{chunk_context}

Using the knowledge graph relationships as a reasoning chain, answer the question step by step.
Show your reasoning path through the concepts explicitly, e.g.:
  "A causes B, and B leads to C, therefore..."

Answer:"""


# ---------------------------------------------------------------------------
# Core KnowledgeGraph class
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    An in-memory knowledge graph with multihop traversal and LLM-assisted reasoning.

    The graph is a directed multigraph: entities are nodes, relations are edges.
    Multihop reasoning works by finding all paths between entities relevant to
    a query (BFS up to max_hops), then feeding those paths as structured context
    to the LLM.
    """

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        # adjacency: source_id -> list of (target_id, relation_label, context, chunk_id)
        self._adj: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
        # reverse index: entity name tokens -> entity IDs (for fuzzy lookup)
        self._name_index: dict[str, set[str]] = defaultdict(set)
        # store raw chunks for context retrieval
        self._chunks: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        if entity.id in self.entities:
            # Merge: accumulate source chunks, keep first definition
            existing = self.entities[entity.id]
            existing.source_chunks.extend(
                c for c in entity.source_chunks if c not in existing.source_chunks
            )
            if not existing.definition and entity.definition:
                existing.definition = entity.definition
        else:
            self.entities[entity.id] = entity
            for token in entity.name.lower().split():
                self._name_index[token].add(entity.id)

    def add_relation(self, relation: Relation) -> None:
        if relation.source_id not in self.entities or relation.target_id not in self.entities:
            return  # skip dangling edges
        self._adj[relation.source_id].append((
            relation.target_id,
            relation.relation,
            relation.context,
            relation.source_chunk,
        ))

    def add_chunk_text(self, chunk_id: str, text: str) -> None:
        self._chunks[chunk_id] = text

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def find_entities_by_name(self, query: str, top_k: int = 5) -> list[Entity]:
        """
        Fuzzy entity lookup by name token overlap.

        Changes vs original:
        - Extended stopword list filters out common DB/query words that match
          too broadly (e.g. "data", "update", "relation").
        - Minimum score threshold raised to 2 — an entity must share at least
          2 tokens with the query to be considered a seed. This prevents
          single-token matches like "atomicity" appearing in a blockchain
          chapter from flooding the subgraph with irrelevant chunks.
        """
        STOPWORDS = {
            # Generic English
            "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in",
            "to", "of", "by", "with", "that", "this", "it", "as", "are", "was",
            "what", "how", "why", "when", "where", "who", "does", "do", "be",
            "not", "but", "if", "then", "from", "its", "into", "than", "so",
            # Too-broad DB/CS terms that appear everywhere
            "data", "database", "relation", "update", "updating", "relate",
            "relating", "system", "used", "using", "each", "also", "only",
            "can", "may", "must", "will", "have", "has", "been", "more",
        }

        tokens = [
            t for t in query.lower().split()
            if t not in STOPWORDS and len(t) > 3
        ]

        scores: dict[str, int] = defaultdict(int)
        for token in tokens:
            for eid in self._name_index.get(token, set()):
                scores[eid] += 1

        # Require at least 2 token matches — prevents single-token noise
        # (e.g. a chunk about LDAP that mentions "atomicity" in passing)
        scores = {eid: s for eid, s in scores.items() if s >= 2}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.entities[eid] for eid, _ in ranked[:top_k] if eid in self.entities]

    def neighbors(self, entity_id: str) -> list[tuple[str, str, str, str]]:
        """Return outgoing edges as (target_id, relation, context, chunk_id)."""
        return self._adj.get(entity_id, [])

    # ------------------------------------------------------------------
    # Multihop traversal
    # ------------------------------------------------------------------

    def find_paths(
        self,
        start_ids: list[str],
        end_ids: list[str],
        max_hops: int = 3,
    ) -> list[list[tuple[str, str, str]]]:
        """
        BFS to find all paths from any start entity to any end entity
        within max_hops. Returns paths as lists of (entity_id, relation, context).
        """
        paths: list[list[tuple[str, str, str]]] = []
        end_set = set(end_ids)

        for start in start_ids:
            queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque()
            queue.append((start, [(start, "", "")]))

            while queue:
                node, path = queue.popleft()

                if len(path) > max_hops + 1:
                    continue

                if node in end_set and len(path) > 1:
                    paths.append(path)
                    continue

                for target_id, relation, context, _ in self.neighbors(node):
                    if target_id not in {p[0] for p in path}:  # no cycles
                        queue.append((target_id, path + [(target_id, relation, context)]))

        return paths

    def subgraph_around(self, entity_ids: list[str], hops: int = 1) -> list[str]:
        """
        Return all entity IDs reachable within `hops` from the seed entities.
        Default reduced to 1 hop to keep the subgraph focused — the relation
        bridge boost in KnowledgeGraphRetriever still surfaces 2-hop reasoning
        by looking at edges between any two subgraph members.
        """
        visited  = set(entity_ids)
        frontier = set(entity_ids)
        for _ in range(hops):
            next_frontier: set[str] = set()
            for eid in frontier:
                for target_id, _, _, _ in self.neighbors(eid):
                    if target_id not in visited:
                        next_frontier.add(target_id)
                        visited.add(target_id)
            frontier = next_frontier
        return list(visited)

    def format_paths_as_context(self, paths: list[list[tuple[str, str, str]]]) -> str:
        """Render traversal paths as readable reasoning chains for the LLM."""
        if not paths:
            return "No direct reasoning paths found."
        lines = []
        for i, path in enumerate(paths[:5], 1):
            chain = []
            for j, (eid, relation, context) in enumerate(path):
                name = self.entities[eid].name if eid in self.entities else eid
                if j == 0:
                    chain.append(name)
                else:
                    chain.append(f"--[{relation}]--> {name}")
            lines.append(f"Path {i}: " + " ".join(chain))
            for eid, relation, context in path[1:]:
                if context:
                    lines.append(f"  Evidence: \"{context}\"")
        return "\n".join(lines)

    def format_subgraph_as_context(self, entity_ids: list[str]) -> str:
        """Format a subgraph as entity definitions + relations for LLM context."""
        lines        = []
        seen_relations: set = set()

        for eid in entity_ids:
            if eid not in self.entities:
                continue
            e = self.entities[eid]
            lines.append(f"[{e.entity_type}] {e.name}: {e.definition}")

        lines.append("\nRelationships:")
        for eid in entity_ids:
            for target_id, relation, context, _ in self.neighbors(eid):
                if target_id not in entity_ids:
                    continue
                key = (eid, target_id, relation)
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                src_name = self.entities[eid].name        if eid       in self.entities else eid
                tgt_name = self.entities[target_id].name  if target_id in self.entities else target_id
                lines.append(f"  {src_name} --[{relation}]--> {tgt_name}")
                if context:
                    lines.append(f"    \"{context}\"")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Multihop query entry point
    # ------------------------------------------------------------------

    def multihop_query(
        self,
        question: str,
        llm_fn: Callable[[str], str],
        max_hops: int = 3,
        top_k_entities: int = 5,
    ) -> dict:
        """
        Answer a question using multihop graph traversal + LLM.

        Steps:
          1. Find seed entities from the question
          2. BFS to find reasoning paths between seed entities
          3. Expand to a local subgraph for broader context
          4. Pull supporting raw text from stored chunks
          5. Feed paths + subgraph + chunks to LLM
        """
        seed_entities = self.find_entities_by_name(question, top_k=top_k_entities)
        if not seed_entities:
            return {
                "answer":        "No relevant entities found in the knowledge graph for this question.",
                "paths":         [],
                "entities_used": [],
                "graph_context": "",
            }

        seed_ids = [e.id for e in seed_entities]

        paths = []
        for i, start in enumerate(seed_ids):
            for end in seed_ids[i + 1:]:
                found = self.find_paths([start], [end], max_hops=max_hops)
                paths.extend(found)

        subgraph_ids    = self.subgraph_around(seed_ids, hops=min(max_hops, 2))
        graph_context   = self.format_subgraph_as_context(subgraph_ids)
        path_context    = self.format_paths_as_context(paths)
        full_graph_context = f"{path_context}\n\nSubgraph:\n{graph_context}"

        chunk_ids = set()
        for eid in subgraph_ids:
            if eid in self.entities:
                chunk_ids.update(self.entities[eid].source_chunks[:2])
        chunk_texts  = [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]
        chunk_context = "\n---\n".join(chunk_texts[:4])

        prompt = MULTIHOP_PROMPT.format(
            question=question,
            graph_context=full_graph_context,
            chunk_context=chunk_context,
        )
        answer = llm_fn(prompt)

        return {
            "answer":        answer,
            "paths":         [[step[0] for step in p] for p in paths],
            "entities_used": [e.name for e in seed_entities],
            "graph_context": full_graph_context,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "entities": {
                eid: {
                    "id":           e.id,
                    "name":         e.name,
                    "type":         e.entity_type,
                    "definition":   e.definition,
                    "source_chunks": e.source_chunks,
                }
                for eid, e in self.entities.items()
            },
            "relations": [
                {
                    "source":   src,
                    "target":   tgt,
                    "relation": rel,
                    "context":  ctx,
                    "chunk":    chk,
                }
                for src, edges in self._adj.items()
                for tgt, rel, ctx, chk in edges
            ],
            "chunks": self._chunks,
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(
            f"Knowledge graph saved: {len(self.entities)} entities, "
            f"{sum(len(v) for v in self._adj.values())} relations -> {path}"
        )

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        data = json.loads(Path(path).read_text())
        kg   = cls()

        for eid, e in data["entities"].items():
            kg.add_entity(Entity(
                id=e["id"],
                name=e["name"],
                entity_type=e["type"],
                definition=e["definition"],
                source_chunks=e["source_chunks"],
            ))

        for r in data["relations"]:
            kg._adj[r["source"]].append(
                (r["target"], r["relation"], r["context"], r["chunk"])
            )

        kg._chunks = data.get("chunks", {})

        # Rebuild name index from loaded entities
        for eid, e in kg.entities.items():
            for token in e.name.lower().split():
                kg._name_index[token].add(eid)

        print(
            f"Knowledge graph loaded: {len(kg.entities)} entities, "
            f"{sum(len(v) for v in kg._adj.values())} relations"
        )
        return kg


# ---------------------------------------------------------------------------
# JSON repair + parsing helpers
# ---------------------------------------------------------------------------

def _repair_json(raw: str) -> str:
    """Fix common LLM JSON formatting mistakes before parsing."""
    raw = re.sub(r'\}\s*\{',     '}, {',  raw)
    raw = re.sub(r'"\s*\n\s*"',  '",\n"', raw)
    raw = re.sub(r',\s*\]',      ']',     raw)
    raw = re.sub(r',\s*\}',      '}',     raw)
    raw = re.sub(r"(?<![\\])'",  '"',     raw)
    return raw


def _parse_llm_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(_repair_json(raw))
    except json.JSONDecodeError:
        pass

    # Extract first balanced {} block
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(_repair_json(raw[start: i + 1]))
                except json.JSONDecodeError:
                    break

    return {"entities": [], "relations": []}


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------

def extract_from_chunk(
    chunk: Chunk,
    llm_fn: Callable[[str], str],
) -> tuple[list[Entity], list[Relation]]:
    """Call the LLM to extract entities and relations from a single chunk."""
    prompt = EXTRACTION_PROMPT.format(
        source_doc=chunk.source_doc or "unknown",
        page=chunk.page or "?",
        text=textwrap.shorten(chunk.text, width=3000, placeholder="..."),
    )

    raw    = llm_fn(prompt)
    parsed = _parse_llm_json(raw)

    entities: list[Entity] = []
    for e in parsed.get("entities", []):
        eid = re.sub(r"[^a-z0-9_]", "_", e.get("id", e.get("name", "unknown")).lower())
        entities.append(Entity(
            id=eid,
            name=e.get("name", eid),
            entity_type=e.get("type", "concept"),
            definition=e.get("definition", ""),
            source_chunks=[chunk.id],
        ))

    entity_id_set = {e.id for e in entities}
    relations: list[Relation] = []
    for r in parsed.get("relations", []):
        src = re.sub(r"[^a-z0-9_]", "_", r.get("source", "").lower())
        tgt = re.sub(r"[^a-z0-9_]", "_", r.get("target", "").lower())
        if src in entity_id_set and tgt in entity_id_set:
            relations.append(Relation(
                source_id=src,
                target_id=tgt,
                relation=r.get("relation", "related_to"),
                context=r.get("context", ""),
                source_chunk=chunk.id,
            ))

    return entities, relations


def build_graph_from_chunks(
    chunks: list[Chunk],
    llm_fn: Callable[[str], str],
    verbose: bool = True,
) -> KnowledgeGraph:
    """Build a KnowledgeGraph by running extraction over every chunk."""
    kg = KnowledgeGraph()

    for i, chunk in enumerate(chunks):
        if verbose:
            print(f"Extracting chunk {i + 1}/{len(chunks)}: {chunk.id}")

        kg.add_chunk_text(chunk.id, chunk.text)

        try:
            entities, relations = extract_from_chunk(chunk, llm_fn)
        except Exception as exc:
            print(f"  Warning: extraction failed for chunk {chunk.id}: {exc}")
            continue

        for e in entities:
            kg.add_entity(e)
        for r in relations:
            kg.add_relation(r)

        if verbose:
            print(f"  -> {len(entities)} entities, {len(relations)} relations")

    if verbose:
        print(
            f"\nGraph complete: {len(kg.entities)} entities, "
            f"{sum(len(v) for v in kg._adj.values())} relations"
        )
    return kg


# ---------------------------------------------------------------------------
# TokenSmith integration helpers
# ---------------------------------------------------------------------------

def chunks_from_tokensmith_index(index_dir: str) -> list[Chunk]:
    """Load chunks from TokenSmith's index/sections/ directory."""
    chunks      = []
    index_path  = Path(index_dir)
    chunks_path = index_path / "textbook_index_chunks.pkl"

    with open(chunks_path, "rb") as f:
        chunks_temp = pickle.load(f)

    print(chunks_temp[0])

    for i, item in enumerate(chunks_temp):
        match = re.search(r"Content:\s*Page\s+(\d+)\s+(.*)", item)
        chunks.append(Chunk(
            id=f"chunk_{i:04d}",
            text=match.group(2) if match else item,
            source_doc="",
            page=match.group(1) if match else 0,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def fake_llm(prompt: str) -> str:
        if "extractor" in prompt.lower() or "chunk" in prompt.lower():
            return json.dumps({
                "entities": [
                    {
                        "id": "virtual_memory", "name": "Virtual Memory",
                        "type": "concept",
                        "definition": "An abstraction that gives each process its own address space.",
                    },
                    {
                        "id": "page_fault", "name": "Page Fault",
                        "type": "process",
                        "definition": "An interrupt raised when a process accesses a page not in physical memory.",
                    },
                    {
                        "id": "demand_paging", "name": "Demand Paging",
                        "type": "algorithm",
                        "definition": "A strategy that loads pages into memory only when they are needed.",
                    },
                ],
                "relations": [
                    {
                        "source": "virtual_memory", "target": "page_fault",
                        "relation": "causes",
                        "context": "Accessing unmapped virtual addresses triggers a page fault.",
                    },
                    {
                        "source": "page_fault", "target": "demand_paging",
                        "relation": "enables",
                        "context": "Page faults are the mechanism that makes demand paging work.",
                    },
                ],
            })
        return (
            "Virtual Memory causes Page Fault which enables Demand Paging. "
            "Therefore, demand paging relies on virtual memory indirectly through "
            "the page fault mechanism."
        )

    chunks = [
        Chunk(
            id="os_ch9_c1",
            text=(
                "Virtual memory gives each process its own address space. "
                "When a process accesses a page not currently in physical memory, "
                "a page fault interrupt is raised. This mechanism enables demand "
                "paging, where pages are loaded only when needed."
            ),
            source_doc="os_textbook",
            page=9,
        )
    ]

    print("Building graph from chunks...")
    kg = build_graph_from_chunks(chunks, llm_fn=fake_llm)

    print("\nEntities:")
    for e in kg.entities.values():
        print(f"  {e.name} ({e.entity_type}): {e.definition}")

    print("\nRunning multihop query...")
    result = kg.multihop_query(
        question="How does virtual memory relate to demand paging?",
        llm_fn=fake_llm,
        max_hops=3,
    )
    print("\nReasoning paths found:")
    print(result["graph_context"])
    print("\nAnswer:")
    print(result["answer"])

    kg.save("/tmp/test_kg.json")
    kg2 = KnowledgeGraph.load("/tmp/test_kg.json")
    assert len(kg2.entities) == len(kg.entities)
    print("\nSave/load: OK")