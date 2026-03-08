from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DatabaseSettings:
    dbname: str
    user: str
    password: str
    host: str
    port: int


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    openrouter_site_url: str | None
    openrouter_app_name: str | None


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    dims: int


@dataclass(frozen=True)
class OpenBrainSettings:
    database: DatabaseSettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    capture_mode: str
    ingest_workers: int
    ingest_poll_ms: int
    memory_retention_months: int
    graph_name: str
    log_level: str

    @classmethod
    def from_env(cls) -> "OpenBrainSettings":
        llm_provider = (os.environ.get("OPENBRAIN_LLM_PROVIDER") or "openai").strip().lower()
        if llm_provider not in {"openai", "openrouter", "ollama"}:
            raise ValueError(f"Unsupported OPENBRAIN_LLM_PROVIDER: {llm_provider!r}")

        llm_model = (
            os.environ.get("OPENBRAIN_LLM_MODEL")
            or ("gpt-4o-mini" if llm_provider != "ollama" else "llama3.1")
        ).strip()
        llm_api_key = (os.environ.get("OPENBRAIN_LLM_API_KEY") or "").strip() or None
        llm_base_url = (os.environ.get("OPENBRAIN_LLM_BASE_URL") or "").strip() or None
        if llm_provider == "ollama" and not llm_base_url:
            llm_base_url = "http://localhost:11434"
        if llm_provider == "openrouter" and not llm_base_url:
            llm_base_url = "https://openrouter.ai/api/v1"
        if llm_provider in {"openai", "openrouter"} and not llm_api_key:
            raise ValueError(f"API key is required for LLM provider {llm_provider!r}")

        embedding_provider = (
            os.environ.get("OPENBRAIN_EMBEDDING_PROVIDER") or llm_provider
        ).strip().lower()
        if embedding_provider not in {"openai", "openrouter", "ollama"}:
            raise ValueError(f"Unsupported OPENBRAIN_EMBEDDING_PROVIDER: {embedding_provider!r}")

        embedding_model = (
            os.environ.get("OPENBRAIN_EMBEDDING_MODEL")
            or ("text-embedding-3-small" if embedding_provider != "ollama" else "nomic-embed-text")
        ).strip()
        embedding_api_key = (os.environ.get("OPENBRAIN_EMBEDDING_API_KEY") or "").strip() or None
        if not embedding_api_key and embedding_provider == llm_provider:
            embedding_api_key = llm_api_key
        embedding_base_url = (os.environ.get("OPENBRAIN_EMBEDDING_BASE_URL") or "").strip() or None
        if not embedding_base_url and embedding_provider == llm_provider:
            embedding_base_url = llm_base_url
        if embedding_provider == "ollama" and not embedding_base_url:
            embedding_base_url = "http://localhost:11434"
        if embedding_provider == "openrouter" and not embedding_base_url:
            embedding_base_url = "https://openrouter.ai/api/v1"
        if embedding_provider in {"openai", "openrouter"} and not embedding_api_key:
            raise ValueError(f"API key is required for embedding provider {embedding_provider!r}")

        dims_default = "1536" if embedding_provider == "openai" else ""
        dims_raw = (os.environ.get("OPENBRAIN_EMBEDDING_DIMS") or dims_default).strip()
        if not dims_raw:
            raise ValueError(
                f"OPENBRAIN_EMBEDDING_DIMS is required for embedding provider {embedding_provider!r}"
            )
        try:
            embedding_dims = int(dims_raw)
        except ValueError as exc:
            raise ValueError("OPENBRAIN_EMBEDDING_DIMS must be an integer") from exc
        if embedding_dims < 1 or embedding_dims > 2000:
            raise ValueError("OPENBRAIN_EMBEDDING_DIMS must be between 1 and 2000 for pgvector HNSW")

        capture_mode = (os.environ.get("OPENBRAIN_CAPTURE_MODE") or "async").strip().lower()
        if capture_mode not in {"async", "sync"}:
            capture_mode = "async"

        return cls(
            database=DatabaseSettings(
                dbname=os.environ.get("POSTGRES_DB", "open_brain"),
                user=os.environ.get("POSTGRES_USER", "brain_user"),
                password=os.environ["POSTGRES_PASSWORD"],
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
            ),
            llm=LLMSettings(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
                openrouter_site_url=(os.environ.get("OPENBRAIN_OPENROUTER_SITE_URL") or "").strip() or None,
                openrouter_app_name=(os.environ.get("OPENBRAIN_OPENROUTER_APP_NAME") or "").strip() or None,
            ),
            embedding=EmbeddingSettings(
                provider=embedding_provider,
                model=embedding_model,
                api_key=embedding_api_key,
                base_url=embedding_base_url,
                dims=embedding_dims,
            ),
            capture_mode=capture_mode,
            ingest_workers=_env_int("OPENBRAIN_INGEST_WORKERS", 1),
            ingest_poll_ms=_env_int("OPENBRAIN_INGEST_POLL_MS", 250, minimum=50),
            memory_retention_months=_env_int("MEMORY_RETENTION_MONTHS", 12, minimum=0),
            graph_name=(os.environ.get("OPENBRAIN_GRAPH_NAME") or "brain_graph_v2").strip(),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )
