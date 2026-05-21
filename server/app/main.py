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
    pass


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
