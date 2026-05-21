from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import config
from api import router

app = FastAPI(
    title=config.mcp.server_name,
    version="2.0.0",
    description="Serveur backend NLP + Graphe + RAG pour notes-graph",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup():
    import asyncio
    asyncio.create_task(_preload_models())


async def _preload_models():
    import logging
    logger = logging.getLogger("uvicorn")
    try:
        logger.info("Preloading spaCy models...")
        import spacy
        spacy.load("fr_core_news_lg")
        spacy.load("en_core_web_lg")
        logger.info("spaCy models loaded")
    except Exception as e:
        logger.warning(f"spaCy preload failed: {e}")

    try:
        logger.info("Preloading Whisper model...")
        from transcription.whisper_model import WhisperModel
        m = WhisperModel()
        m.load()
        logger.info("Whisper model loaded")
    except Exception as e:
        logger.warning(f"Whisper preload failed: {e}")


@app.on_event("shutdown")
async def shutdown():
    pass


def main():
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug,
        log_level="info" if not config.server.debug else "debug",
    )


if __name__ == "__main__":
    main()
