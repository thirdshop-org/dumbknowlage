from __future__ import annotations

import json
from pathlib import Path


def export_session(session_data: dict, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)
    return str(path)


def export_analysis(analysis: dict, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    return str(path)


def build_export_payload(
    session: dict,
    chunks: list[dict],
    frequencies: list | None = None,
    co_occurrences: list | None = None,
    tfidf: list | None = None,
    burst_topics: list | None = None,
    graph_data: dict | None = None,
) -> dict:
    return {
        "session": session,
        "chunks": chunks,
        "analysis": {
            "frequencies": frequencies or [],
            "co_occurrences": co_occurrences or [],
            "tfidf": tfidf or [],
            "burst_topics": burst_topics or [],
        },
        "graph": graph_data or {},
    }
