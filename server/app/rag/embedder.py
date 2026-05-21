from __future__ import annotations

from typing import Any

import requests

from config import config


class OllamaEmbedder:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or config.rag.embed_model
        self.base_url = (base_url or config.rag.ollama_base_url).rstrip("/")
        self._dim = config.rag.embed_dim

    def embed(self, text: str) -> list[float]:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            return [0.0] * self._dim
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("embeddings", [])
