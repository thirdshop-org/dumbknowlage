from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.retriever import HybridRetriever, RetrievalResult


@dataclass
class QueryResult:
    context: str
    sources: list[dict] = field(default_factory=list)
    retrieval: RetrievalResult | None = None


def build_context(retrieval: RetrievalResult) -> str:
    parts = []
    for i, chunk in enumerate(retrieval.chunks, 1):
        source = chunk.metadata.get("source", "?")
        parts.append(f"--- DOCUMENT {i} (source: {source}) ---\n{chunk.text}")
    context = "\n\n".join(parts)

    if retrieval.graph_context:
        graph = retrieval.graph_context
        extra = []
        if graph.entities:
            entity_lines = [f"  - {e['name']} ({e['type']})" for e in graph.entities]
            extra.append("Entités détectées:\n" + "\n".join(entity_lines))
        if graph.relations:
            rel_lines = [f"  - {r.get('name', r.get('entity', '?'))} → {r.get('relation', '?')}" for r in graph.relations[:10]]
            extra.append("Relations dans le graphe:\n" + "\n".join(rel_lines))
        if extra:
            context += "\n\nContexte supplémentaire (graphe de connaissances):\n" + "\n".join(extra)

    return context


def query(question: str, top_k: int | None = None) -> QueryResult:
    retriever = HybridRetriever()
    retrieval = retriever.retrieve(question, top_k)

    context = build_context(retrieval)

    sources = []
    for chunk in retrieval.chunks:
        sources.append({
            "id": chunk.id,
            "score": round(chunk.score, 3),
            "source": chunk.metadata.get("source", ""),
            "source_type": chunk.metadata.get("source_type", ""),
            "text_preview": chunk.text[:200],
        })

    return QueryResult(context=context, sources=sources, retrieval=retrieval)
