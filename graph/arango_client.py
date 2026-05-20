from __future__ import annotations

import time
from typing import Any

from arango import ArangoClient as ArangoHTTPClient
from arango.database import StandardDatabase
from arango.exceptions import DatabaseCreateError

from config import config
from graph.models import Edge, WordNode, SentenceNode, TopicNode, DocumentNode, sanitize_key


class GraphManager:
    def __init__(self):
        self.cfg = config.arango
        self._client: ArangoHTTPClient | None = None
        self._db: StandardDatabase | None = None

    def connect(self, retries: int = 5, delay: float = 2.0) -> bool:
        self._client = ArangoHTTPClient(self.cfg.host)
        for attempt in range(retries):
            try:
                sys_db = self._client.db("_system", username=self.cfg.username,
                                          password=self.cfg.password)
                databases = sys_db.databases()
                if self.cfg.database not in databases:
                    sys_db.create_database(self.cfg.database)
                self._db = self._client.db(self.cfg.database, username=self.cfg.username,
                                            password=self.cfg.password)
                self._ensure_collections()
                return True
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise ConnectionError(f"Impossible de connecter ArangoDB: {e}")
        return False

    def _ensure_collections(self):
        vertex_collections = [
            self.cfg.word_collection,
            self.cfg.sentence_collection,
            self.cfg.topic_collection,
            self.cfg.document_collection,
        ]

        edge_definitions = [
            self.cfg.edge_co_occurs,
            self.cfg.edge_next_to,
            self.cfg.edge_has_topic,
            self.cfg.edge_has_dependency,
            self.cfg.edge_is_similar,
        ]

        # Ensure vertex collections exist
        for col_name in vertex_collections:
            if not self._db.has_collection(col_name):
                self._db.create_collection(col_name)

        graph_name = "word_graph"
        graphs = {g["name"]: g for g in self._db.graphs()}

        if graph_name in graphs:
            graph = self._db.graph(graph_name)
            existing_edge_defs = {ed["edge_collection"] for ed in graph.edge_definitions()}
        else:
            graph = self._db.create_graph(graph_name)
            existing_edge_defs = set()

        # Add missing edge definitions
        for edge_name in edge_definitions:
            if edge_name not in existing_edge_defs:
                graph.create_edge_definition(
                    edge_collection=edge_name,
                    from_vertex_collections=vertex_collections,
                    to_vertex_collections=vertex_collections,
                )

        # Register vertex collections in the graph if missing
        vc = set(graph.vertex_collections())
        for v in vertex_collections:
            if v not in vc:
                try:
                    graph.create_vertex_collection(v)
                except Exception:
                    pass

    @property
    def db(self) -> StandardDatabase:
        if self._db is None:
            raise RuntimeError("GraphManager non connecté. Appelez connect() d'abord.")
        return self._db

    # --- CRUD Nœuds ---

    def upsert_word(self, word: WordNode) -> str:
        col = self.db.collection(self.cfg.word_collection)
        try:
            existing = col.get(word._key)
            if existing:
                col.update_match({"_key": word._key}, {"frequency": existing["frequency"] + 1})
                return word._key
        except Exception:
            pass
        col.insert(word.to_dict(), overwrite=True)
        return word._key

    def insert_sentence(self, sentence: SentenceNode) -> str:
        col = self.db.collection(self.cfg.sentence_collection)
        col.insert(sentence.to_dict(), overwrite=True)
        return sentence._key

    def upsert_topic(self, topic: TopicNode) -> str:
        col = self.db.collection(self.cfg.topic_collection)
        col.insert(topic.to_dict(), overwrite=True)
        return topic._key

    def insert_document(self, doc: DocumentNode) -> str:
        col = self.db.collection(self.cfg.document_collection)
        col.insert(doc.to_dict(), overwrite=True)
        return doc._key

    # --- CRUD Arêtes ---

    def insert_edge(self, edge: Edge) -> str:
        col = self.db.collection(edge.collection)
        result = col.insert(edge.to_dict(), overwrite=True)
        return result["_id"]

    def create_co_occurrence(self, lemma_a: str, lemma_b: str, weight: int = 1):
        edge = Edge(
            collection=self.cfg.edge_co_occurs,
            _from=f"{self.cfg.word_collection}/{sanitize_key(lemma_a)}",
            _to=f"{self.cfg.word_collection}/{sanitize_key(lemma_b)}",
            relation_type="CO_OCCURS_WITH",
            weight=float(weight),
        )
        self.insert_edge(edge)

    def create_dependency(self, word_lemma: str, head_lemma: str, dep: str):
        edge = Edge(
            collection=self.cfg.edge_has_dependency,
            _from=f"{self.cfg.word_collection}/{sanitize_key(word_lemma)}",
            _to=f"{self.cfg.word_collection}/{sanitize_key(head_lemma)}",
            relation_type=dep,
        )
        self.insert_edge(edge)

    def create_sentence_word_link(self, sentence_key: str, word_lemma: str):
        edge = Edge(
            collection=self.cfg.edge_next_to,
            _from=f"{self.cfg.sentence_collection}/{sentence_key}",
            _to=f"{self.cfg.word_collection}/{sanitize_key(word_lemma)}",
            relation_type="NEXT_TO",
        )
        self.insert_edge(edge)

    def create_sentence_topic_link(self, sentence_key: str, topic_key: str, weight: float = 1.0):
        edge = Edge(
            collection=self.cfg.edge_has_topic,
            _from=f"{self.cfg.sentence_collection}/{sentence_key}",
            _to=f"{self.cfg.topic_collection}/{topic_key}",
            relation_type="HAS_TOPIC",
            weight=weight,
        )
        self.insert_edge(edge)

    # --- Requêtes ---

    def query(self, aql: str, bind_vars: dict | None = None) -> list[dict]:
        cursor = self.db.aql.execute(aql, bind_vars=bind_vars or {})
        return [doc for doc in cursor]

    def get_top_words(self, limit: int = 20) -> list[dict]:
        aql = f"""
        FOR w IN {self.cfg.word_collection}
            SORT w.frequency DESC
            LIMIT {limit}
            RETURN {{word: w.lemma, pos: w.pos, frequency: w.frequency, is_entity: w.is_entity}}
        """
        return self.query(aql)

    def get_word_neighbors(self, lemma: str) -> list[dict]:
        aql = f"""
        FOR v, e IN 1..1 OUTBOUND '{self.cfg.word_collection}/{sanitize_key(lemma)}'
            GRAPH 'word_graph'
            RETURN {{neighbor: v.lemma, relation: e.relation_type, weight: e.weight}}
        """
        return self.query(aql)

    def get_hot_topics(self, limit: int = 10) -> list[dict]:
        aql = f"""
        FOR t IN {self.cfg.topic_collection}
            SORT t.weight DESC
            LIMIT {limit}
            RETURN {{topic: t.label, weight: t.weight, keywords: t.keywords}}
        """
        return self.query(aql)

    def close(self):
        if self._client:
            self._client = None
            self._db = None
