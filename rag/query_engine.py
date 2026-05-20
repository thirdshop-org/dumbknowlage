from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests

from config import config
from rag.retriever import HybridRetriever, RetrievalResult


@dataclass
class QueryResult:
    answer: str
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

    prompt = (
        "Tu es un assistant spécialisé dans l'analyse de transcriptions et documents. "
        "Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.\n"
        "Si le contexte ne contient pas la réponse, dis-le clairement.\n\n"
        f"Contexte :\n{context}\n\n"
        f"Question : {question}\n"
        "Réponse :"
    )

    resp = requests.post(
        f"{config.rag.ollama_base_url}/api/generate",
        json={
            "model": config.rag.llm_model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    answer = data.get("response", "")

    sources = []
    for chunk in retrieval.chunks:
        sources.append({
            "id": chunk.id,
            "score": round(chunk.score, 3),
            "source": chunk.metadata.get("source", ""),
            "text_preview": chunk.text[:200],
        })

    return QueryResult(answer=answer, sources=sources, retrieval=retrieval)
