from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class CapturePersistResult:
    raw_capture_id: int | None
    job_id: int | None
    duplicate: bool = False


@dataclass(frozen=True)
class CaptureJob:
    id: int
    raw_capture_id: int
    user_id: str
    thought: str
    attempt_count: int = 0
    source: str = "capture_thought"
    ingest_strategy: str = "personal"
    managed_kind_override: str | None = None


@dataclass(frozen=True)
class GraphNode:
    name: str
    label: str


@dataclass(frozen=True)
class GraphEdge:
    source: str
    source_label: str
    relationship: str
    target: str
    target_label: str


@dataclass(frozen=True)
class GraphExtraction:
    nodes: list[GraphNode]
    edges: list[GraphEdge]


@dataclass(frozen=True)
class MemoryChunk:
    raw_capture_id: int
    user_id: str
    source: str
    chunk_index: int
    content: str
    ingest_mode: str
    embedding: list[float]
    embedding_provider: str
    embedding_model: str
    embedding_dims: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagedMemoryCandidate:
    kind: str
    topic: str
    canonical_text: str
    evidence_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagedMemoryRecord:
    id: int
    user_id: str
    kind: str
    topic: str
    topic_key: str
    canonical_text: str
    status: str
    created_at: datetime | str | None
    updated_at: datetime | str | None
    superseded_by: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagedMemoryMatch:
    id: int
    kind: str
    topic: str
    topic_key: str
    canonical_text: str
    created_at: datetime | str | None
    updated_at: datetime | str | None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagedMemoryResolution:
    action: str
    canonical_text: str


@dataclass
class EvidenceItem:
    source: str
    text: str
    score_raw: float | None = None
    score_norm: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawCaptureMatch:
    id: int
    content: str
    created_at: datetime | str | None
    content_len: int
    source: str | None


@dataclass(frozen=True)
class StoredRawCapture:
    id: int
    user_id: str
    source: str
    content: str
    ingest_strategy: str
    managed_kind_override: str | None
    created_at: datetime | str | None
