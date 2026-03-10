from __future__ import annotations

from typing import Any, Protocol

from .models import CaptureJob, CapturePersistResult, GraphExtraction, MemoryChunk, RawCaptureMatch


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    embedding_dims: int

    def embed(self, text: str) -> list[float]:
        ...


class StructuredGenerationProvider(Protocol):
    provider_name: str
    model_name: str

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        ...


class RawCaptureRepository(Protocol):
    def capture_and_enqueue(
        self,
        thought: str,
        *,
        user_id: str,
        source: str,
        ingest_strategy: str,
        external_id: str | None,
    ) -> CapturePersistResult:
        ...

    def search(self, query: str, *, user_id: str, limit: int = 8) -> list[RawCaptureMatch]:
        ...


class CaptureJobRepository(Protocol):
    def claim(self, *, job_id: int | None = None) -> CaptureJob | None:
        ...

    def mark_done(self, job: CaptureJob) -> None:
        ...

    def mark_retry(self, job: CaptureJob, *, error: Exception, retry_delays: tuple[int, ...]) -> None:
        ...

    def stats(self) -> dict[str, int]:
        ...


class VectorMemoryRepository(Protocol):
    def upsert_chunks(self, chunks: list[MemoryChunk]) -> int:
        ...

    def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        limit: int = 10,
        ingest_modes: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        ...


class GraphRepository(Protocol):
    def add_nodes(self, nodes: list[dict[str, Any]]) -> None:
        ...

    def add_edges(self, edges: list[dict[str, Any]]) -> None:
        ...

    def search(self, query: str, *, limit: int = 5) -> dict[str, list[dict[str, Any]]]:
        ...


class GraphExtractor(Protocol):
    def extract(self, text: str, *, user_id: str) -> GraphExtraction:
        ...


class BulletExtractor(Protocol):
    def extract(self, text: str, *, max_bullets: int = 3) -> list[str]:
        ...
