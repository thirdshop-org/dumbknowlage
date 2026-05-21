from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkData:
    text: str
    chunk_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class SessionCreateResponse:
    session_id: str
    chunks_count: int


@dataclass
class EntityInfo:
    name: str
    type: str
    key: str
    confidence: float = 0.0
    mentions: int = 1


@dataclass
class SearchResult:
    text: str
    score: float
    source: str
    source_type: str
    session_id: str


@dataclass
class ContextResult:
    context: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    entities: list[EntityInfo] = field(default_factory=list)


@dataclass
class HealthStatus:
    status: str
    arango: bool = False
    chroma: bool = False
    ollama: bool = False
    whisper_model: str = ""
