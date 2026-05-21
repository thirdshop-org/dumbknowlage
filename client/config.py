from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024


@dataclass
class ChunkConfig:
    chunk_duration: float = 8.0
    overlap: float = 1.5
    sample_rate: int = 16000


@dataclass
class ClientConfig:
    server_url: str = os.getenv("NOTES_GRAPH_SERVER", "http://localhost:8000")
    cache_dir: str = os.getenv(
        "NOTES_GRAPH_CACHE",
        os.path.expanduser("~/.cache/notes-graph"),
    )
    default_duration: float = 30.0
    default_language: str = "fr"
    timeout: int = 300
    audio: AudioConfig = field(default_factory=AudioConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)


config = ClientConfig()
