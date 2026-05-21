from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from config import config


class ApiClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or config.server_url).rstrip("/")
        self.timeout = config.timeout

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> dict:
        r = requests.get(self.url("/api/health"), timeout=10)
        r.raise_for_status()
        return r.json()

    def transcribe(self, audio_path: str | Path, language: str = "fr",
                   build_graph: bool = True, defer: bool = False) -> dict:
        path = Path(audio_path)
        with open(path, "rb") as f:
            r = requests.post(
                self.url(
                    f"/api/sessions/transcribe?language={language}"
                    f"&build_graph={str(build_graph).lower()}"
                    f"&defer={str(defer).lower()}"
                ),
                files={"file": (path.name, f, "audio/mpeg")},
                timeout=30 if defer else self.timeout,
            )
        r.raise_for_status()
        return r.json()

    def ingest(self, text: str, filename: str = "document.txt",
               language: str = "fr", build_graph: bool = True,
               defer: bool = False) -> dict:
        # Use the JSON-body endpoint: query-string ingest blows the URL size
        # limit on documents larger than a few KB.
        r = requests.post(
            self.url("/api/sessions/ingest/json"),
            json={
                "text": text,
                "filename": filename,
                "language": language,
                "build_graph": build_graph,
                "defer": defer,
            },
            timeout=30 if defer else self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def ingest_json(self, payload: dict) -> dict:
        r = requests.post(
            self.url("/api/sessions/ingest/json"),
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        r = requests.post(
            self.url(f"/api/search?query={requests.utils.quote(query)}&top_k={top_k}"),
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def context(self, question: str, top_k: int = 5) -> dict:
        r = requests.post(
            self.url(f"/api/context?question={requests.utils.quote(question)}&top_k={top_k}"),
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def list_sessions(self) -> list[dict]:
        r = requests.get(self.url("/api/sessions"), timeout=10)
        r.raise_for_status()
        return r.json()

    def get_session(self, session_id: str) -> dict:
        r = requests.get(self.url(f"/api/sessions/{session_id}"), timeout=10)
        r.raise_for_status()
        return r.json()

    def list_entities(self, type: str | None = None, q: str = "",
                      limit: int = 50, offset: int = 0) -> list[dict]:
        params = f"limit={limit}&offset={offset}"
        if type:
            params += f"&type={type}"
        if q:
            params += f"&q={requests.utils.quote(q)}"
        r = requests.get(self.url(f"/api/entities?{params}"), timeout=10)
        r.raise_for_status()
        return r.json()

    def get_entity(self, type: str, key: str, depth: int = 2) -> dict:
        r = requests.get(
            self.url(f"/api/entities/{type}/{key}?depth={depth}"),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def confirm_entity(self, type: str, key: str) -> dict:
        r = requests.post(self.url(f"/api/entities/{type}/{key}/confirm"), timeout=10)
        r.raise_for_status()
        return r.json()

    def deny_entity(self, type: str, key: str, reason: str = "") -> dict:
        r = requests.post(
            self.url(f"/api/entities/{type}/{key}/deny?reason={requests.utils.quote(reason)}"),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def rename_entity(self, type: str, key: str, new_name: str) -> dict:
        r = requests.post(
            self.url(f"/api/entities/{type}/{key}/rename?new_name={requests.utils.quote(new_name)}"),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def graph_aql(self, query: str) -> dict:
        r = requests.post(
            self.url(f"/api/graph/aql?query={requests.utils.quote(query)}"),
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def list_rules(self) -> list[dict]:
        r = requests.get(self.url("/api/rules"), timeout=10)
        r.raise_for_status()
        return r.json()

    def list_corrections(self, limit: int = 20) -> dict:
        r = requests.get(self.url(f"/api/corrections?limit={limit}"), timeout=10)
        r.raise_for_status()
        return r.json()
