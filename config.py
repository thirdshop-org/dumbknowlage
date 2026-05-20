import os
from dataclasses import dataclass, field


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass
class WhisperConfig:
    model: str = "turbo"
    device: str = "cpu"
    compute_type: str = "float32"
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
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024


@dataclass
class NLPConfig:
    spacy_models: list = field(default_factory=lambda: ["fr_core_news_lg", "en_core_web_lg"])
    camembert_model: str = "camembert-base"
    device: str = "cpu"


@dataclass
class ArangoConfig:
    host: str = "http://localhost:8529"
    username: str = "root"
    password: str = "whispernlp"
    database: str = "whisper_nlp_graph"
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
    db_path: str = os.path.join(os.path.dirname(__file__), "data", "sessions.db")


@dataclass
class ExtractionConfig:
    co_occurrence_window: int = 5
    burst_window_size: int = 6
    burst_threshold: float = 1.5
    top_n_frequencies: int = 20
    top_n_hot_topics: int = 10


@dataclass
class AppConfig:
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    arango: ArangoConfig = field(default_factory=ArangoConfig)
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    data_dir: str = os.path.join(os.path.dirname(__file__), "data")


def _apply_device():
    device = detect_device()
    config.whisper.device = device
    config.nlp.device = device
    if device == "cuda":
        config.whisper.compute_type = "float16"
    config.whisper.compute_type = "float32"


config = AppConfig()
_apply_device()
