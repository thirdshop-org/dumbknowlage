from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings
from rich.console import Console

from config import config
from rag.embedder import OllamaEmbedder
from storage.sqlite_store import SQLiteStore

console = Console()


def get_chroma_client() -> chromadb.PersistentClient:
    persist_dir = Path(config.rag.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(),
    )


def get_or_create_collection(client: chromadb.PersistentClient | None = None):
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=config.rag.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def index_all(force: bool = False) -> int:
    store = SQLiteStore()
    store.connect()

    client = get_chroma_client()
    collection = get_or_create_collection(client)
    embedder = OllamaEmbedder()

    sessions = store.get_all_sessions()
    indexed = 0

    existing_ids = set(collection.get()["ids"]) if not force else set()

    for session in sessions:
        session_id = session["id"]
        chunks = store.get_chunks(session_id)
        for chunk in chunks:
            chunk_id = f"{session_id}_{chunk['chunk_index']}"
            if not force and chunk_id in existing_ids:
                continue

            text = chunk["text"]
            if not text.strip():
                continue

            embedding = embedder.embed(text)
            metadata = {
                "session_id": session_id,
                "chunk_index": str(chunk["chunk_index"]),
                "source": session["source"],
                "created_at": session["created_at"],
            }

            if force and chunk_id in existing_ids:
                collection.update(ids=[chunk_id], embeddings=[embedding], metadatas=[metadata], documents=[text])
            else:
                collection.add(ids=[chunk_id], embeddings=[embedding], metadatas=[metadata], documents=[text])
            indexed += 1

    store.close()
    return indexed


def index_session(session_id: str) -> int:
    store = SQLiteStore()
    store.connect()

    session = store.get_session(session_id)
    if not session:
        store.close()
        return 0

    client = get_chroma_client()
    collection = get_or_create_collection(client)
    embedder = OllamaEmbedder()

    chunks = store.get_chunks(session_id)
    indexed = 0

    for chunk in chunks:
        chunk_id = f"{session_id}_{chunk['chunk_index']}"
        text = chunk["text"]
        if not text.strip():
            continue

        embedding = embedder.embed(text)
        metadata = {
            "session_id": session_id,
            "chunk_index": str(chunk["chunk_index"]),
            "source": session["source"],
            "created_at": session["created_at"],
        }
        collection.upsert(ids=[chunk_id], embeddings=[embedding], metadatas=[metadata], documents=[text])
        indexed += 1

    store.close()
    return indexed
