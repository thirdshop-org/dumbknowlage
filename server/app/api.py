from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse

from config import config
from storage.sqlite_store import SQLiteStore
from shared.schemas import (
    ChunkData,
    ContextResult,
    EntityInfo,
    HealthStatus,
    SearchResult,
    SessionCreateResponse,
)

router = APIRouter()


def _get_store() -> SQLiteStore:
    store = SQLiteStore()
    store.connect()
    return store


def _get_gm():
    from graph.arango_client import GraphManager
    gm = GraphManager()
    if not gm.connect():
        raise HTTPException(503, "ArangoDB indisponible")
    return gm


def _get_retriever():
    from rag.retriever import HybridRetriever
    return HybridRetriever()


def _run_nlp_pipeline(store, session_id, chunks, lang, build_graph, doc_metadata=None):
    from pipeline import run_nlp_pipeline as _run
    _run(store, session_id, chunks, lang, build_graph, doc_metadata)


# ─── Health ─────────────────────────────────────────────────────────────────


@router.get("/api/health")
async def health():
    status = HealthStatus(status="ok")
    try:
        gm = _get_gm()
        status.arango = True
        gm.close()
    except Exception:
        status.arango = False
        status.status = "degraded"

    try:
        from rag.indexer import get_chroma_client, get_or_create_collection
        col = get_or_create_collection(get_chroma_client())
        col.count()
        status.chroma = True
    except Exception:
        status.chroma = False
        status.status = "degraded"

    try:
        import requests
        r = requests.get(f"{config.rag.ollama_base_url}/api/tags", timeout=5)
        status.ollama = r.status_code == 200
    except Exception:
        status.ollama = False
        status.status = "degraded"

    status.whisper_model = config.whisper.model
    return status


# ─── Sessions ────────────────────────────────────────────────────────────────


@router.get("/api/sessions")
async def list_sessions():
    store = _get_store()
    sessions = store.get_all_sessions()
    store.close()
    return sessions


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    store = _get_store()
    session = store.get_session(session_id)
    if not session:
        store.close()
        raise HTTPException(404, "Session introuvable")
    chunks = list(store.get_chunks(session_id))
    store.close()
    return {"session": dict(session), "chunks": chunks}


# ─── Transcription ───────────────────────────────────────────────────────────


@router.post("/api/sessions/transcribe", response_model=SessionCreateResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query("fr"),
    build_graph: bool = Query(True),
):
    import traceback
    import numpy as np
    import soundfile as sf
    from io import BytesIO
    from audio.chunker import chunk_audio
    from transcription.transcriber import Transcriber

    try:
        # Convert to WAV via ffmpeg (handles mp3, m4a, etc.)
        raw = await file.read()
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            proc = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-ac", "1", "-ar", str(config.chunk.sample_rate),
                 "-sample_fmt", "s16", "-y", wav_path],
                input=raw, capture_output=True, timeout=120,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg error: {proc.stderr.decode()}")
            audio, sr = sf.read(wav_path)
        finally:
            os.unlink(wav_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        duration = len(audio) / sr

        store = _get_store()
        session_id = store.create_session(
            source=file.filename or "audio_upload",
            language=language,
            model=f"whisper-{config.whisper.model}",
            duration=duration,
        )

        transcriber = Transcriber(model_name=config.whisper.model)
        chunks = transcriber.transcribe_chunks(audio, language=language)
        store.insert_chunks_batch(session_id, chunks)

        try:
            _run_nlp_pipeline(store, session_id, chunks, language, build_graph)
        except Exception as e:
            store.close()
            raise HTTPException(500, f"Pipeline error: {type(e).__name__}: {e}")
        store.close()

        return SessionCreateResponse(session_id=session_id, chunks_count=len(chunks))
    except Exception as e:
        raise HTTPException(500, f"Transcribe error: {type(e).__name__}: {e}\n{traceback.format_exc()}")


# ─── Ingest ──────────────────────────────────────────────────────────────────


@router.post("/api/sessions/ingest", response_model=SessionCreateResponse)
async def ingest_text(
    text: str = Query(..., description="Texte du document"),
    filename: str = Query("document.txt"),
    language: str = Query("fr"),
    build_graph: bool = Query(True),
):
    from document.reader import chunk_text

    store = _get_store()
    session_id = store.create_session(
        source=filename,
        language=language,
        model="document/ingest",
    )

    doc_chunks = chunk_text(text)
    chunks = [
        {"text": c["text"], "chunk_index": i, "start_time": 0.0, "end_time": 0.0}
        for i, c in enumerate(doc_chunks)
    ]
    store.insert_chunks_batch(session_id, chunks)

    try:
        _run_nlp_pipeline(store, session_id, chunks, language, build_graph,
                          doc_metadata={"filename": filename, "file_type": Path(filename).suffix})
    except Exception as e:
        store.close()
        raise HTTPException(500, f"Pipeline error: {type(e).__name__}: {e}")
    store.close()

    return SessionCreateResponse(session_id=session_id, chunks_count=len(chunks))


@router.post("/api/sessions/ingest/json")
async def ingest_json(payload: dict):
    text = payload.get("text", "")
    filename = payload.get("filename", "document.txt")
    language = payload.get("language", "fr")
    build_graph = payload.get("build_graph", True)
    return await ingest_text(text, filename, language, build_graph)


# ─── Search ──────────────────────────────────────────────────────────────────


@router.post("/api/search")
async def search(
    query: str = Query(...),
    top_k: int = Query(5),
):
    retriever = _get_retriever()
    chunks = retriever.search_only(query, top_k=top_k)
    results = [
        SearchResult(
            text=c.text[:500],
            score=c.score,
            source=c.metadata.get("source", "?"),
            source_type=c.metadata.get("source_type", "?"),
            session_id=c.metadata.get("session_id", ""),
        )
        for c in chunks
    ]
    return results


@router.post("/api/context")
async def context(
    question: str = Query(...),
    top_k: int = Query(5),
):
    from rag.query_engine import query as rag_query
    result = rag_query(question, top_k=top_k)
    entities = []
    if result.retrieval and result.retrieval.graph_context:
        seen = set()
        for ent in result.retrieval.graph_context.entities:
            key = ent.get("name", "") + ent.get("type", "")
            if key not in seen:
                seen.add(key)
                entities.append(EntityInfo(name=ent.get("name", ""), type=ent.get("type", ""), key=""))
    return ContextResult(
        context=result.context,
        sources=result.sources,
        entities=entities,
    )


# ─── Entities ────────────────────────────────────────────────────────────────


@router.get("/api/entities")
async def list_entities(
    type: str | None = Query(None, alias="type"),
    q: str = "",
    limit: int = 50,
    offset: int = 0,
):
    gm = _get_gm()
    if q:
        entities = gm.search_entities(q, limit=limit)
    else:
        entities = gm.get_entities(entity_type=type, limit=limit, offset=offset)
    gm.close()
    return entities


@router.get("/api/entities/{entity_type}/{entity_key}")
async def get_entity_detail(entity_type: str, entity_key: str, depth: int = 2):
    gm = _get_gm()
    col = gm.db.collection(entity_type)
    ent = col.get(entity_key)
    if not ent:
        gm.close()
        raise HTTPException(404, "Entité introuvable")
    docs = gm.get_entity_documents(entity_type, entity_key)
    network = gm.get_entity_network(entity_type, entity_key, depth=depth)
    gm.close()
    return {
        "entity": dict(ent),
        "documents": docs,
        "network": network[:20],
    }


@router.post("/api/entities/{entity_type}/{entity_key}/confirm")
async def confirm_entity(entity_type: str, entity_key: str):
    gm = _get_gm()
    gm.confirm_entity(entity_type, entity_key)
    gm.close()
    return {"status": "confirmed", "confidence": 0.95}


@router.post("/api/entities/{entity_type}/{entity_key}/deny")
async def deny_entity(entity_type: str, entity_key: str, reason: str = ""):
    gm = _get_gm()
    gm.deny_entity(entity_type, entity_key, reason=reason)
    gm.close()
    return {"status": "denied"}


@router.post("/api/entities/{entity_type}/{entity_key}/rename")
async def rename_entity(entity_type: str, entity_key: str, new_name: str = Query(...)):
    gm = _get_gm()
    gm.rename_entity(entity_type, entity_key, new_name)
    gm.close()
    return {"status": "renamed", "new_name": new_name}


# ─── Graph ───────────────────────────────────────────────────────────────────


@router.post("/api/graph/aql")
async def graph_aql(query: str = Query(...)):
    gm = _get_gm()
    try:
        results = gm.query(query)
        gm.close()
        return {"count": len(results), "results": results[:50]}
    except Exception as e:
        gm.close()
        raise HTTPException(400, str(e))


@router.post("/api/graph/revalidate")
async def graph_revalidate(dry_run: bool = False):
    gm = _get_gm()
    stats = gm.revalidate_entities(dry_run=dry_run)
    gm.close()
    return stats


@router.post("/api/graph/cleanup")
async def graph_cleanup(dry_run: bool = False):
    gm = _get_gm()
    stats = gm.cleanup_dangling_edges(dry_run=dry_run)
    gm.close()
    return stats


# ─── Rules / Corrections ─────────────────────────────────────────────────────


@router.get("/api/rules")
async def list_rules():
    gm = _get_gm()
    store = gm.get_correction_store()
    rules = store.get_rules()
    gm.close()
    return rules


@router.get("/api/corrections")
async def list_corrections(limit: int = 20):
    gm = _get_gm()
    store = gm.get_correction_store()
    corrections = store.get_recent_corrections(limit=limit)
    stats = store.get_correction_stats()
    gm.close()
    return {"stats": stats, "corrections": corrections}


@router.delete("/api/rules/{rule_key}")
async def delete_rule(rule_key: str):
    gm = _get_gm()
    store = gm.get_correction_store()
    store.delete_rule(rule_key)
    gm.close()
    return {"status": "deleted"}


# ─── MCP SSE endpoint ────────────────────────────────────────────────────────


from mcp.server.sse import SseServerTransport as _SseTransport
_mcp_sse = _SseTransport("/mcp")


def _get_mcp_server():
    from mcp_handler import create_mcp_server
    return create_mcp_server()


@router.get("/mcp")
async def mcp_sse(request: Request):
    server = _get_mcp_server()
    async with _mcp_sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


@router.post("/mcp")
async def mcp_post(request: Request):
    await _mcp_sse.handle_post_message(request.scope, request.receive, request._send)
