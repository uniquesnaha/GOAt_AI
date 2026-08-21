from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Infrastructure/deployment configuration only.

    The RAG hyperparameters (CFG, DENSE_CHILD_K, CONTEXT_CHAR_BUDGET, the
    system prompt, generation settings, etc.) intentionally live nowhere but
    backend/app/rag/engine.py, copied verbatim from the golden reference in
    scripts/benchmark_full_rag_t4_latency_winner.py. Duplicating those numbers
    here would risk one copy drifting from the other; this file only carries
    the things that must differ between a Colab notebook and a deployed VM.
    """

    model_config = SettingsConfigDict(extra="ignore")

    data_root: Path = Field(default=REPO_ROOT / "data", validation_alias="GOAT_DATA_ROOT")
    qdrant_url: str = Field(default="http://127.0.0.1:6333", validation_alias="QDRANT_URL")
    qdrant_grpc_port: int = Field(default=6334, validation_alias="QDRANT_GRPC_PORT")
    qdrant_prefer_grpc: bool = Field(default=False, validation_alias="QDRANT_PREFER_GRPC")

    sarvam_api_key: str = Field(default="", validation_alias="SARVAM_API_KEY")
    sarvam_api_base: str = Field(default="https://api.sarvam.ai", validation_alias="SARVAM_API_BASE")

    host: str = Field(default="0.0.0.0", validation_alias="GOAT_HOST")
    port: int = Field(default=8000, validation_alias="GOAT_PORT")

    cors_origins: list[str] = Field(default_factory=lambda: ["*"], validation_alias="GOAT_CORS_ORIGINS")
    corpus_profile: str = Field(default="25k", validation_alias="GOAT_CORPUS_PROFILE")


settings = Settings()

