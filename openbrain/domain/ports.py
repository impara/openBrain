from __future__ import annotations

from typing import Any, Protocol

from .models import (
    CaptureJob,
    CapturePersistResult,
    GraphExtraction,
    ManagedMemoryCandidate,
    ManagedMemoryMatch,
    ManagedMemoryRecord,
    ManagedMemoryResolution,
    MemoryChunk,
    RawCaptureMatch,
    StoredRawCapture,
)


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
        managed_kind_override: str | None,
    ) -> CapturePersistResult:
        ...

    def search(self, query: str, *, user_id: str, limit: int = 8) -> list[RawCaptureMatch]:
        ...

    def list_for_backfill(self, *, user_id: str) -> list[StoredRawCapture]:
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

    def search_vectors(
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


class ManagedMemoryExtractor(Protocol):
    def extract(self, text: str, *, forced_kind: str | None = None) -> list[ManagedMemoryCandidate]:
        ...


class ManagedMemoryResolver(Protocol):
    def resolve(
        self,
        *,
        candidate: ManagedMemoryCandidate,
        existing: list[ManagedMemoryRecord],
    ) -> ManagedMemoryResolution:
        ...


class ManagedMemoryRepository(Protocol):
    def find_active_by_topic(self, *, user_id: str, kind: str, topic_key: str) -> list[ManagedMemoryRecord]:
        ...

    def insert_managed_memory(
        self,
        *,
        user_id: str,
        kind: str,
        topic: str,
        topic_key: str,
        canonical_text: str,
        embedding: list[float],
        embedding_provider: str,
        embedding_model: str,
        embedding_dims: int,
        metadata: dict[str, Any],
        raw_capture_ids: list[int],
    ) -> int:
        ...

    def update_managed_memory(
        self,
        *,
        managed_memory_id: int,
        canonical_text: str,
        embedding: list[float],
        embedding_provider: str,
        embedding_model: str,
        embedding_dims: int,
        metadata: dict[str, Any],
        raw_capture_ids: list[int],
    ) -> None:
        ...

    def supersede_managed_memories(
        self,
        *,
        managed_memory_ids: list[int],
        superseded_by: int | None,
    ) -> None:
        ...

    def search_managed_memories(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        limit: int = 5,
        kind: str | None = None,
    ) -> list[ManagedMemoryMatch]:
        ...

    def list_active_managed_memories(
        self,
        *,
        user_id: str,
        kind: str | None = None,
        limit: int = 50,
    ) -> list[ManagedMemoryRecord]:
        ...

    def clear_managed_memories(self, *, user_id: str) -> None:
        ...
