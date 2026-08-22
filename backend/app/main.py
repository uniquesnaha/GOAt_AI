from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_health, routes_query, routes_stt
from app.config import settings

logger = logging.getLogger("goat_ai")

app = FastAPI(title="GOAt AI backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_query.router)
app.include_router(routes_stt.router)


@app.on_event("startup")
def load_engine() -> None:
    """Load the RAG engine once at startup.

    Loading FullRAG requires CUDA, the Qwen models, and the indexed data
    (see DATA_SETUP.md) to all be present. If any of that isn't ready yet
    (e.g. data hasn't been dropped in), the process stays up and /readyz
    reports why, instead of crash-looping the container.
    """
    from app.rag.engine import FullRAG

    try:
        engine = FullRAG()
        engine.warmup()
        app.state.engine = engine
        app.state.engine_load_error = None
        logger.info("RAG engine loaded and warmed up.")
    except Exception as exc:  # noqa: BLE001 - startup readiness gate, not a RAG-logic path
        app.state.engine = None
        app.state.engine_load_error = str(exc)
        logger.error("RAG engine failed to load: %s", exc)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
