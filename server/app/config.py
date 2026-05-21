from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class WhisperConfig:
    model: str = os.getenv("WHISPER_MODEL", "turbo")
    device: str = "cpu"
    compute_type: str = os.getenv("COMPUTE_TYPE", "float32")
    language: str | None = None
    beam_size: int = 1
    temperature: float = 0.0
    condition_on_previous_text: bool = True


@dataclass
class ChunkConfig:
    chunk_duration: float = 8.0
    overlap: float = 1.5
    sample_rate: int = 16000


@dataclass
class NLPConfig:
    spacy_models: list = field(default_factory=lambda: ["fr_core_news_lg", "en_core_web_lg"])
    camembert_model: str = "camembert-base"
    device: str = "cpu"


@dataclass
class ArangoConfig:
    host: str = os.getenv("ARANGO_HOST", "http://localhost:8529")
    username: str = "root"
    password: str = os.getenv("ARANGO_ROOT_PASSWORD", "whispernlp")
    database: str = os.getenv("ARANGO_DATABASE", "whisper_nlp_graph")
    word_collection: str = "Word"
    sentence_collection: str = "Sentence"
    topic_collection: str = "Topic"
    document_collection: str = "Document"
    edge_co_occurs: str = "co_occurs_with"
    edge_next_to: str = "next_to"
    edge_has_topic: str = "has_topic"
    edge_has_dependency: str = "has_dependency"
    edge_is_similar: str = "is_similar_to"
    edge_appears_in: str = "appears_in"
    edge_works_for: str = "works_for"
    edge_located_in: str = "located_in"
    edge_related_to: str = "related_to"


@dataclass
class SQLiteConfig:
    db_path: str = os.getenv(
        "SQLITE_PATH",
        "/app/data/sessions.db",
    )


@dataclass
class ExtractionConfig:
    co_occurrence_window: int = 5
    burst_window_size: int = 6
    burst_threshold: float = 1.5
    top_n_frequencies: int = 20
    top_n_hot_topics: int = 10


@dataclass
class RAGConfig:
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.21:11434")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "/app/data/chroma_db")
    chroma_collection: str = "whisper_nlp_chunks"
    top_k: int = 3
    use_graph_enrichment: bool = True
    similarity_threshold: float = 0.3

    def __post_init__(self):
        if not self.chroma_persist_dir:
            self.chroma_persist_dir = "/app/data/chroma_db"


@dataclass
class MCPConfig:
    server_name: str = "whisper-nlp-graph"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
    data_dir: str = "/app/data"


@dataclass
class AppConfig:
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    arango: ArangoConfig = field(default_factory=ArangoConfig)
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    data_dir: str = "/app/data"


config = AppConfig()
