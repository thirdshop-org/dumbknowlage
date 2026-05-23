from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, Request, UploadFile
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

# Serialize deferred NLP pipelines: Whisper + spaCy + embeddings + Arango
# writes are heavy. Running several in parallel saturates RAM/CPU and (more
# critically) makes the SQLite analysis table fight over the writer lock.
_PIPELINE_SEM = threading.Semaphore(1)

_MCP_SERVER: Any | None = None


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


@router.get("/api/debug/whisper")
async def debug_whisper():
    import traceback, sys
    try:
        from transcription.whisper_model import WhisperModel
        m = WhisperModel()
        m.load()
        return {"status": "loaded", "model": config.whisper.model}
    except Exception as e:
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "python": sys.version,
        }


# ─── Sessions ────────────────────────────────────────────────────────────────


@router.get("/api/sessions")
async def list_sessions():
    store = _get_store()
    try:
        return store.get_all_sessions()
    finally:
        store.close()


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    store = _get_store()
    try:
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session introuvable")
        chunks = list(store.get_chunks(session_id))
        return {"session": dict(session), "chunks": chunks}
    finally:
        store.close()


# ─── Transcription ───────────────────────────────────────────────────────────


@router.post("/api/sessions/transcribe", response_model=SessionCreateResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Query("fr"),
    build_graph: bool = Query(True),
    defer: bool = Query(False, description="Return session_id immediately; run Whisper + NLP after response"),
):
    raw = await file.read()
    filename = file.filename or "audio_upload"
    if defer:
        store = _get_store()
        try:
            session_id = store.create_session(
                source=filename,
                language=language,
                model=f"whisper-{config.whisper.model}",
                duration=0.0,
            )
        finally:
            store.close()
        background_tasks.add_task(
            _transcribe_pipeline_background, raw, session_id, language, build_graph
        )
        return SessionCreateResponse(session_id=session_id, chunks_count=0)
    return await asyncio.to_thread(
        _transcribe_sync, raw, filename, language, build_graph
    )


def _transcribe_pipeline_background(raw: bytes, session_id: str, language: str,
                                    build_graph: bool):
    """Run ffmpeg + Whisper + NLP for a previously-created session.
    Best-effort: failures are logged, no exception is propagated.
    Serialized via _PIPELINE_SEM so only one pipeline runs at a time."""
    with _PIPELINE_SEM:
        import traceback
        import numpy as np
        import soundfile as sf
        from transcription.transcriber import Transcriber

        try:
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
                    raise RuntimeError(f"ffmpeg: {proc.stderr.decode(errors='replace')}")
                audio, sr = sf.read(wav_path)
            except subprocess.TimeoutExpired:
                raise RuntimeError("ffmpeg a dépassé le timeout de 120s")
            finally:
                os.unlink(wav_path)

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            store = _get_store()
            try:
                transcriber = Transcriber(model_name=config.whisper.model)
                chunks = transcriber.transcribe_chunks(audio, language=language)
                store.insert_chunks_batch(session_id, chunks)
                store.update_session_duration(session_id, len(audio) / sr) if hasattr(store, "update_session_duration") else None
                try:
                    _run_nlp_pipeline(store, session_id, chunks, language, build_graph)
                except Exception as e:
                    print(f"[transcribe defer] NLP pipeline error for session {session_id}: "
                          f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            finally:
                store.close()
        except Exception as e:
            print(f"[transcribe defer] Transcription error for session {session_id}: "
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def _transcribe_sync(raw: bytes, filename: str, language: str, build_graph: bool):
    import traceback
    import numpy as np
    import soundfile as sf
    from transcription.transcriber import Transcriber

    try:
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
                raise RuntimeError(f"ffmpeg: {proc.stderr.decode(errors='replace')}")
            audio, sr = sf.read(wav_path)
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg a dépassé le timeout de 120s")
        finally:
            os.unlink(wav_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        duration = len(audio) / sr

        store = _get_store()
        try:
            session_id = store.create_session(
                source=filename,
                language=language,
                model=f"whisper-{config.whisper.model}",
                duration=duration,
            )

            transcriber = Transcriber(model_name=config.whisper.model)
            chunks = transcriber.transcribe_chunks(audio, language=language)
            store.insert_chunks_batch(session_id, chunks)

            with _PIPELINE_SEM:
                _run_nlp_pipeline(store, session_id, chunks, language, build_graph)
        finally:
            store.close()

        return SessionCreateResponse(session_id=session_id, chunks_count=len(chunks))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Transcribe error: {type(e).__name__}: {e}\n{traceback.format_exc()}")


# ─── Ingest ──────────────────────────────────────────────────────────────────


@router.post("/api/sessions/ingest", response_model=SessionCreateResponse)
async def ingest_text(
    background_tasks: BackgroundTasks,
    text: str = Query(..., description="Texte du document"),
    filename: str = Query("document.txt"),
    language: str = Query("fr"),
    build_graph: bool = Query(True),
    defer: bool = Query(False, description="Return session_id immediately; run NLP pipeline after response"),
):
    if defer:
        from document.reader import chunk_text
        store = _get_store()
        try:
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
        finally:
            store.close()
        background_tasks.add_task(
            _ingest_pipeline_background,
            session_id, chunks, language, build_graph,
            {"filename": filename, "file_type": Path(filename).suffix},
        )
        return SessionCreateResponse(session_id=session_id, chunks_count=len(chunks))
    return await asyncio.to_thread(
        _ingest_text_sync, text, filename, language, build_graph
    )


def _ingest_pipeline_background(session_id: str, chunks: list, language: str,
                                build_graph: bool, doc_metadata: dict):
    """Run the NLP pipeline for an existing session. Best-effort, logs on failure.
    Serialized via _PIPELINE_SEM so only one pipeline runs at a time."""
    with _PIPELINE_SEM:
        import traceback
        store = _get_store()
        try:
            _run_nlp_pipeline(store, session_id, chunks, language, build_graph,
                              doc_metadata=doc_metadata)
        except Exception as e:
            print(f"[ingest defer] Pipeline error for session {session_id}: "
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            store.close()


def _ingest_text_sync(text: str, filename: str, language: str, build_graph: bool):
    from document.reader import chunk_text

    store = _get_store()
    try:
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

        with _PIPELINE_SEM:
            _run_nlp_pipeline(store, session_id, chunks, language, build_graph,
                              doc_metadata={"filename": filename, "file_type": Path(filename).suffix})
    finally:
        store.close()

    return SessionCreateResponse(session_id=session_id, chunks_count=len(chunks))


@router.post("/api/sessions/ingest/json")
async def ingest_json(payload: dict, background_tasks: BackgroundTasks):
    text = payload.get("text", "")
    filename = payload.get("filename", "document.txt")
    language = payload.get("language", "fr")
    build_graph = payload.get("build_graph", True)
    defer = payload.get("defer", False)
    if defer:
        from document.reader import chunk_text
        store = _get_store()
        try:
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
        finally:
            store.close()
        background_tasks.add_task(
            _ingest_pipeline_background,
            session_id, chunks, language, build_graph,
            {"filename": filename, "file_type": Path(filename).suffix},
        )
        return SessionCreateResponse(session_id=session_id, chunks_count=len(chunks))
    return await asyncio.to_thread(_ingest_text_sync, text, filename, language, build_graph)


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
    try:
        if q:
            return gm.search_entities(q, limit=limit)
        return gm.get_entities(entity_type=type, limit=limit, offset=offset)
    finally:
        gm.close()


@router.get("/api/entities/{entity_type}/{entity_key}")
async def get_entity_detail(entity_type: str, entity_key: str, depth: int = 2):
    gm = _get_gm()
    try:
        col = gm.db.collection(entity_type)
        ent = col.get(entity_key)
        if not ent:
            raise HTTPException(404, "Entité introuvable")
        docs = gm.get_entity_documents(entity_type, entity_key)
        network = gm.get_entity_network(entity_type, entity_key, depth=depth)
        return {
            "entity": dict(ent),
            "documents": docs,
            "network": network[:20],
        }
    finally:
        gm.close()


@router.post("/api/entities/{entity_type}/{entity_key}/confirm")
async def confirm_entity(entity_type: str, entity_key: str):
    gm = _get_gm()
    try:
        gm.confirm_entity(entity_type, entity_key)
        return {"status": "confirmed", "confidence": 0.95}
    finally:
        gm.close()


@router.post("/api/entities/{entity_type}/{entity_key}/deny")
async def deny_entity(entity_type: str, entity_key: str, reason: str = ""):
    gm = _get_gm()
    try:
        gm.deny_entity(entity_type, entity_key, reason=reason)
        return {"status": "denied"}
    finally:
        gm.close()


@router.post("/api/entities/{entity_type}/{entity_key}/rename")
async def rename_entity(entity_type: str, entity_key: str, new_name: str = Query(...)):
    gm = _get_gm()
    try:
        gm.rename_entity(entity_type, entity_key, new_name)
        return {"status": "renamed", "new_name": new_name}
    finally:
        gm.close()


# ─── Graph ───────────────────────────────────────────────────────────────────


@router.post("/api/graph/aql")
async def graph_aql(query: str = Query(...)):
    gm = _get_gm()
    try:
        results = gm.query(query)
        return {"count": len(results), "results": results[:50]}
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        gm.close()


@router.post("/api/graph/revalidate")
async def graph_revalidate(dry_run: bool = False):
    gm = _get_gm()
    try:
        stats = gm.revalidate_entities(dry_run=dry_run)
        return stats
    finally:
        gm.close()


@router.post("/api/graph/cleanup")
async def graph_cleanup(dry_run: bool = False):
    gm = _get_gm()
    try:
        stats = gm.cleanup_dangling_edges(dry_run=dry_run)
        return stats
    finally:
        gm.close()


# ─── Rules / Corrections ─────────────────────────────────────────────────────


@router.get("/api/rules")
async def list_rules():
    gm = _get_gm()
    try:
        store = gm.get_correction_store()
        return store.get_rules()
    finally:
        gm.close()


@router.get("/api/corrections")
async def list_corrections(limit: int = 20):
    gm = _get_gm()
    try:
        store = gm.get_correction_store()
        corrections = store.get_recent_corrections(limit=limit)
        stats = store.get_correction_stats()
        return {"stats": stats, "corrections": corrections}
    finally:
        gm.close()


@router.delete("/api/rules/{rule_key}")
async def delete_rule(rule_key: str):
    gm = _get_gm()
    try:
        store = gm.get_correction_store()
        store.delete_rule(rule_key)
        return {"status": "deleted"}
    finally:
        gm.close()


# ─── MCP SSE endpoint ────────────────────────────────────────────────────────


from mcp.server.sse import SseServerTransport as _SseTransport
from starlette.responses import Response as _StarletteResponse
_mcp_sse = _SseTransport("/mcp")


class _AlreadySentResponse(_StarletteResponse):
    """No-op response: the SSE transport already wrote everything via the raw
    ASGI send callable. Without this, FastAPI auto-wraps the handler's `None`
    return as a JSONResponse and tries to send a second `http.response.start`,
    which raises RuntimeError in uvicorn."""

    async def __call__(self, scope, receive, send):
        return


def _get_mcp_server():
    global _MCP_SERVER
    if _MCP_SERVER is None:
        from mcp_handler import create_mcp_server
        _MCP_SERVER = create_mcp_server()
    return _MCP_SERVER


@router.get("/mcp")
async def mcp_sse(request: Request):
    server = _get_mcp_server()
    async with _mcp_sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
    return _AlreadySentResponse()


@router.post("/mcp")
async def mcp_post(request: Request):
    await _mcp_sse.handle_post_message(request.scope, request.receive, request._send)
    return _AlreadySentResponse()
