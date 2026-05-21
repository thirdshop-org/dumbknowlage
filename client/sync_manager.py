from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from config import config


class SyncManager:
    """Gère le cache local et la synchronisation avec le serveur."""

    def __init__(self):
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir = self.cache_dir / "pending"
        self.pending_dir.mkdir(exist_ok=True)

    def save_pending(self, payload: dict, source_type: str = "audio") -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        key = f"{source_type}_{ts}"
        path = self.pending_dir / f"{key}.json"
        payload["_sync"] = {"key": key, "created": time.time(), "source_type": source_type}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return key

    def list_pending(self) -> list[dict]:
        items = []
        for f in sorted(self.pending_dir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
                items.append({**data.get("_sync", {}), "path": str(f), "payload": data})
        return items

    def sync_all(self, api_client) -> tuple[int, int]:
        from api_client import ApiClient

        client = api_client or ApiClient()
        success = 0
        failed = 0

        for item in self.list_pending():
            payload = item["payload"]
            path = item["path"]
            try:
                if payload.get("_sync", {}).get("source_type") == "audio":
                    client.ingest_json(payload)
                else:
                    client.ingest_json(payload)
                os.remove(path)
                success += 1
            except Exception as e:
                failed += 1

        return success, failed
