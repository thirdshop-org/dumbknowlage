from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from config import config


class SQLiteStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = Path(db_path or config.sqlite.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self):
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                duration REAL,
                language TEXT,
                model TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_time REAL,
                end_time REAL,
                text TEXT NOT NULL,
                language TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE TABLE IF NOT EXISTS analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
            CREATE INDEX IF NOT EXISTS idx_analysis_session ON analysis(session_id);
        """)
        self._conn.commit()

    def create_session(self, source: str, language: str | None = None,
                       model: str | None = None, duration: float | None = None) -> str:
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO sessions (id, source, duration, language, model, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, source, duration, language, model, now),
        )
        self._conn.commit()
        return session_id

    def insert_chunk(self, session_id: str, chunk_index: int,
                     start_time: float, end_time: float,
                     text: str, language: str | None = None):
        self._conn.execute(
            "INSERT INTO chunks (session_id, chunk_index, start_time, end_time, text, language) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, chunk_index, start_time, end_time, text, language),
        )
        self._conn.commit()

    def insert_chunks_batch(self, session_id: str, chunks: list[dict]):
        self._conn.executemany(
            "INSERT INTO chunks (session_id, chunk_index, start_time, end_time, text, language) VALUES (?, ?, ?, ?, ?, ?)",
            [(session_id, c["chunk_index"], c["start_time"], c["end_time"], c["text"], c.get("language")) for c in chunks],
        )
        self._conn.commit()

    def save_analysis(self, session_id: str, analysis_type: str, result: dict):
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO analysis (session_id, analysis_type, result, created_at) VALUES (?, ?, ?, ?)",
            (session_id, analysis_type, json.dumps(result, ensure_ascii=False), now),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_chunks(self, session_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE session_id = ? ORDER BY chunk_index", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_sessions(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, source, duration, language, model, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_analysis(self, session_id: str, analysis_type: str | None = None) -> list[dict]:
        if analysis_type:
            rows = self._conn.execute(
                "SELECT * FROM analysis WHERE session_id = ? AND analysis_type = ? ORDER BY created_at",
                (session_id, analysis_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM analysis WHERE session_id = ? ORDER BY created_at", (session_id,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["result"] = json.loads(d["result"])
            yield d

    def close(self):
        if self._conn:
            self._conn.close()
