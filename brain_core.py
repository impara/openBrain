"""
brain_core.py — Compatibility wrappers around the OpenBrain application.

This module keeps the historical public function names stable while the runtime
implementation now lives in the openbrain package.
"""

from __future__ import annotations

from openbrain import build_openbrain, get_openbrain
from openbrain.domain.models import CaptureJob, CapturePersistResult
from openbrain.settings import OpenBrainSettings


def get_app():
    return get_openbrain()


def build_app(settings: OpenBrainSettings | None = None):
    return build_openbrain(settings)


def capture_thought(thought: str) -> str:
    return get_app().capture_thought(thought)


def ingest(content: str, source: str = "import", external_id: str | None = None) -> str:
    return get_app().ingest(content, source=source, external_id=external_id)


def search_brain(query: str, debug: bool = False) -> str:
    return get_app().search_brain(query, debug=debug)


def start_background_workers() -> None:
    get_app().start_background_workers()


def stop_background_workers(timeout: float = 1.0) -> None:
    get_app().stop_background_workers(timeout=timeout)


def process_next_capture_job() -> bool:
    return get_app().process_next_capture_job()


def get_capture_job_stats() -> dict[str, int]:
    return get_app().get_capture_job_stats()


def _process_capture_job_by_id(job_id: int) -> bool:
    return get_app().process_capture_job_by_id(job_id)


__all__ = [
    "CaptureJob",
    "CapturePersistResult",
    "OpenBrainSettings",
    "build_app",
    "capture_thought",
    "get_app",
    "get_capture_job_stats",
    "ingest",
    "process_next_capture_job",
    "search_brain",
    "start_background_workers",
    "stop_background_workers",
    "_process_capture_job_by_id",
]
