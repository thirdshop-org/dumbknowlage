from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chromadb

from config import config
from rag.embedder import OllamaEmbedder
from rag.indexer import get_chroma_client, get_or_create_collection


@dataclass
class ChunkResult:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphContext:
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    neighbors: list[dict] = field(default_factory=list)


@dataclass
class RetrievalResult:
    chunks: list[ChunkResult] = field(default_factory=list)
    graph_context: GraphContext | None = None


class HybridRetriever:
    def __init__(self):
        self.embedder = OllamaEmbedder()
        self.client = get_chroma_client()
        self.collection = get_or_create_collection(self.client)

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or config.rag.top_k
        question_emb = self.embedder.embed(question)

        results = self.collection.query(
            query_embeddings=[question_emb],
            n_results=k,
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            score = results["distances"][0][i] if results["distances"] else 0.0
            chunk = ChunkResult(
                id=results["ids"][0][i],
                score=1.0 - score,
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            )
            chunks.append(chunk)

        result = RetrievalResult(chunks=chunks)

        if config.rag.use_graph_enrichment:
            result.graph_context = self._enrich_with_graph(chunks)

        return result

    def _enrich_with_graph(self, chunks: list[ChunkResult]) -> GraphContext | None:
        try:
            from graph.arango_client import GraphManager
            from nlp.spacy_analyzer import SpacyAnalyzer

            gm = GraphManager()
            if not gm.connect():
                return None

            full_text = " ".join(c.text for c in chunks)
            analyzer = SpacyAnalyzer()
            spacy_result = analyzer.analyze(full_text, lang="fr")

            lemmas = [t["lemma"] for t in spacy_result["tokens"] if not t["is_punct"] and not t["is_stop"]]
            lemmas = list(set(lemmas))[:20]

            entities = []
            relations = []
            neighbors = []

            for lemma in lemmas:
                try:
                    word_neighbors = gm.get_word_neighbors(lemma)
                    neighbors.extend(word_neighbors)
                except Exception:
                    pass

            seen_pairs = set()
            for n in neighbors:
                pair = (n.get("neighbor", ""), n.get("relation", ""))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)

            # Entity-level enrichment
            for ent in spacy_result["entities"]:
                try:
                    from graph.entity_models import sanitize_key, entity_from_label

                    active_rules: list[dict] = []
                    try:
                        store = gm.get_correction_store()
                        active_rules = store.get_rules(auto_apply_only=True)
                    except Exception:
                        pass

                    entity = entity_from_label(ent["label"], ent["text"],
                                               active_rules=active_rules)
                    if entity is None:
                        continue

                    docs = gm.get_entity_documents(entity.collection, entity._key)
                    network = gm.get_entity_network(entity.collection, entity._key, depth=1)
                    entities.append({"name": entity.name, "type": entity.collection, "documents": docs})
                    relations.extend(network)
                except Exception:
                    continue

            gm.close()

            return GraphContext(
                entities=entities[:10],
                relations=relations[:20],
                neighbors=neighbors[:20],
            )
        except Exception:
            return None

    def search_only(self, question: str, top_k: int | None = None) -> list[ChunkResult]:
        k = top_k or config.rag.top_k
        question_emb = self.embedder.embed(question)

        results = self.collection.query(
            query_embeddings=[question_emb],
            n_results=k,
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            score = results["distances"][0][i] if results["distances"] else 0.0
            chunk = ChunkResult(
                id=results["ids"][0][i],
                score=1.0 - score,
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            )
            chunks.append(chunk)
        return chunks
