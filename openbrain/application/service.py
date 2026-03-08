from __future__ import annotations

import logging
import threading
import time

from openbrain.domain.models import CaptureJob, MemoryChunk
from openbrain.domain.ports import (
    BulletExtractor,
    EmbeddingProvider,
    GraphExtractor,
    GraphRepository,
)
from openbrain.domain.search import (
    build_candidates,
    format_debug_report,
    format_search_response,
    rank_evidence,
)
from openbrain.domain.text import chunk_text_sentences, detect_auto_ingest_strategy, normalized_text
from openbrain.infrastructure.repositories import OpenBrainRepositories
from openbrain.settings import OpenBrainSettings

logger = logging.getLogger("open_brain")

_CAPTURE_ACK_TEXT = "Thought captured and queued for indexing."
_PERSIST_ERROR_HINT = " Check provider configuration and database connectivity."
_CAPTURE_RETRY_BACKOFF_SECONDS = (60, 300, 900)
_SINGLE_USER_ID = "default"


class OpenBrainApplication:
    def __init__(
        self,
        *,
        settings: OpenBrainSettings,
        repositories: OpenBrainRepositories,
        vector_repo: OpenBrainRepositories,
        graph_repo: GraphRepository,
        embedding_provider: EmbeddingProvider,
        bullet_extractor: BulletExtractor,
        graph_extractor: GraphExtractor,
    ):
        self.settings = settings
        self.repositories = repositories
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo
        self.embedding_provider = embedding_provider
        self.bullet_extractor = bullet_extractor
        self.graph_extractor = graph_extractor
        self._worker_lock = threading.Lock()
        self._worker_threads: list[threading.Thread] = []
        self._worker_stop_event = threading.Event()

    @staticmethod
    def brain_user_id() -> str:
        return _SINGLE_USER_ID

    def capture_thought(self, thought: str) -> str:
        thought = (thought or "").strip()
        if not thought:
            return "Warning: Please provide a non-empty thought."
        logger.info("Capturing thought (len=%d)", len(thought))
        started_at = time.perf_counter()
        try:
            persist_result = self.repositories.capture_and_enqueue(
                thought,
                user_id=self.brain_user_id(),
                source="capture_thought",
                ingest_strategy=detect_auto_ingest_strategy(thought, source="capture_thought"),
                external_id=None,
            )
            queue_stats = self.get_capture_job_stats()
            accept_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Capture accepted raw_capture_id=%s job_id=%s accept_ms=%.1f queue_depth=%d processing=%d failed=%d",
                persist_result.raw_capture_id,
                persist_result.job_id,
                accept_ms,
                queue_stats.get("queued", 0),
                queue_stats.get("processing", 0),
                queue_stats.get("failed", 0),
            )
        except Exception as exc:
            logger.error("Memory capture enqueue failed: %s", exc)
            return f"Warning: Failed to persist memory. Error: {exc}.{_PERSIST_ERROR_HINT}"

        if self.settings.capture_mode == "sync":
            if persist_result.job_id is None:
                return "Warning: Failed to enqueue memory for processing."
            if self.process_capture_job_by_id(persist_result.job_id):
                return "Successfully captured thought into memory."
            return "Warning: Failed to fully commit memory during synchronous processing."
        return _CAPTURE_ACK_TEXT

    def ingest(self, content: str, *, source: str = "import", external_id: str | None = None) -> str:
        content = (content or "").strip()
        if not content:
            return "Warning: Please provide non-empty content."

        logger.info("Ingest source=%s len=%d", source, len(content))
        started_at = time.perf_counter()
        try:
            persist_result = self.repositories.capture_and_enqueue(
                content,
                user_id=self.brain_user_id(),
                source=source,
                ingest_strategy=detect_auto_ingest_strategy(content, source=source),
                external_id=external_id,
            )
            if persist_result.duplicate:
                logger.info(
                    "Ingest skipped atomically (dedup) source=%s external_id=%s raw_capture_id=%s",
                    source,
                    external_id,
                    persist_result.raw_capture_id,
                )
                return "Already ingested (skipped duplicate for this source and id)."
            queue_stats = self.get_capture_job_stats()
            accept_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Ingest accepted source=%s raw_capture_id=%s job_id=%s accept_ms=%.1f queue_depth=%d",
                source,
                persist_result.raw_capture_id,
                persist_result.job_id,
                accept_ms,
                queue_stats.get("queued", 0),
            )
        except Exception as exc:
            logger.error("Ingest enqueue failed source=%s: %s", source, exc)
            return f"Warning: Failed to persist. Error: {exc}.{_PERSIST_ERROR_HINT}"

        if self.settings.capture_mode == "sync" and persist_result.job_id is not None:
            if self.process_capture_job_by_id(persist_result.job_id):
                return "Ingested and indexed."
            return "Warning: Ingest queued but synchronous processing did not complete."
        if self.settings.capture_mode == "sync":
            return "Warning: Ingest queued but synchronous processing did not complete."
        return "Ingested and queued for indexing."

    def search_brain(self, query: str, *, debug: bool = False) -> str:
        logger.info("Searching brain query=%r debug=%s", query[:80], debug)
        if not (query or "").strip():
            return "Answer: Please provide a non-empty search query."

        query_embedding = self.embedding_provider.embed(query)
        vector_results = self.vector_repo.search_vectors(
            query_embedding,
            user_id=self.brain_user_id(),
            limit=10,
        )
        raw_matches = self.repositories.search_raw(query, user_id=self.brain_user_id(), limit=8)
        relations = self.graph_repo.search(query, limit=5)
        candidates = build_candidates(query, vector_results, relations, raw_matches)
        evidence, ranking_debug = rank_evidence(query, candidates)
        if debug:
            return format_debug_report(query, ranking_debug, evidence)
        return format_search_response(query, evidence)

    def _build_memory_chunks(self, thought: str, *, raw_capture_id: int, source: str, ingest_mode: str) -> list[MemoryChunk]:
        chunks = chunk_text_sentences(thought, max_chunk_chars=900, max_chunks=6)
        results: list[MemoryChunk] = []
        for idx, chunk in enumerate(chunks, start=1):
            results.append(
                MemoryChunk(
                    raw_capture_id=raw_capture_id,
                    user_id=self.brain_user_id(),
                    source=source,
                    chunk_index=idx,
                    content=chunk,
                    ingest_mode=ingest_mode,
                    embedding=self.embedding_provider.embed(chunk),
                    embedding_provider=self.embedding_provider.provider_name,
                    embedding_model=self.embedding_provider.model_name,
                    embedding_dims=self.embedding_provider.embedding_dims,
                    metadata={
                        "origin": source,
                        "raw_capture_id": raw_capture_id,
                        "ingest_mode": ingest_mode,
                    },
                )
            )
        return results

    def _build_fact_chunks(self, thought: str, *, raw_capture_id: int, source: str) -> list[MemoryChunk]:
        bullets: list[str] = []
        seen: set[str] = set()
        for chunk in chunk_text_sentences(thought, max_chunk_chars=900, max_chunks=6):
            for bullet in self.bullet_extractor.extract(chunk, max_bullets=3):
                norm = normalized_text(bullet)
                if not norm or norm in seen or len(norm.split()) < 5:
                    continue
                seen.add(norm)
                bullets.append(bullet)
                if len(bullets) >= 10:
                    break
            if len(bullets) >= 10:
                break

        results: list[MemoryChunk] = []
        for idx, bullet in enumerate(bullets, start=1):
            results.append(
                MemoryChunk(
                    raw_capture_id=raw_capture_id,
                    user_id=self.brain_user_id(),
                    source=source,
                    chunk_index=idx,
                    content=bullet,
                    ingest_mode="fact",
                    embedding=self.embedding_provider.embed(bullet),
                    embedding_provider=self.embedding_provider.provider_name,
                    embedding_model=self.embedding_provider.model_name,
                    embedding_dims=self.embedding_provider.embedding_dims,
                    metadata={
                        "origin": source,
                        "raw_capture_id": raw_capture_id,
                        "ingest_mode": "fact",
                        "fact_index": idx,
                    },
                )
            )
        return results

    def _write_graph(self, thought: str, *, source: str) -> None:
        extraction = self.graph_extractor.extract(thought, user_id=self.brain_user_id())
        if extraction.nodes:
            self.graph_repo.add_nodes(
                [
                    {
                        "label": node.label,
                        "properties": {"name": node.name, "user_id": self.brain_user_id(), "origin": source},
                    }
                    for node in extraction.nodes
                ]
            )
        if extraction.edges:
            self.graph_repo.add_edges(
                [
                    {
                        "source": edge.source,
                        "source_label": edge.source_label,
                        "target": edge.target,
                        "target_label": edge.target_label,
                        "relationship": edge.relationship,
                    }
                    for edge in extraction.edges
                ]
            )

    def ingest_capture(
        self,
        *,
        thought: str,
        user_id: str,
        raw_capture_id: int | None,
        source: str,
        ingest_strategy: str,
    ) -> tuple[int, int]:
        del user_id
        if raw_capture_id is None:
            return 0, 0
        raw_chunks = self._build_memory_chunks(
            thought,
            raw_capture_id=raw_capture_id,
            source=source,
            ingest_mode="raw",
        )
        raw_added = self.vector_repo.upsert_chunks(raw_chunks)
        if ingest_strategy == "snippet":
            return raw_added, 0

        fact_chunks = self._build_fact_chunks(thought, raw_capture_id=raw_capture_id, source=source)
        fact_added = self.vector_repo.upsert_chunks(fact_chunks)
        try:
            self._write_graph(thought, source=source)
        except Exception as exc:
            logger.warning("Graph write failed for raw_capture_id=%s: %s", raw_capture_id, exc)
        return raw_added + fact_added, fact_added

    def _process_capture_job(self, job: CaptureJob) -> bool:
        started_at = time.perf_counter()
        try:
            added_count, fact_added = self.ingest_capture(
                thought=job.thought,
                user_id=job.user_id,
                raw_capture_id=job.raw_capture_id,
                source=job.source,
                ingest_strategy=job.ingest_strategy,
            )
            duration_ms = (time.perf_counter() - started_at) * 1000
            self.repositories.mark_done(job)
            logger.info(
                "Capture job done job_id=%s raw_capture_id=%s attempt=%d added=%d fact_added=%d ingest_ms=%.1f",
                job.id,
                job.raw_capture_id,
                job.attempt_count,
                added_count,
                fact_added,
                duration_ms,
            )
            return True
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            self.repositories.mark_retry(job, error=exc, retry_delays=_CAPTURE_RETRY_BACKOFF_SECONDS)
            logger.warning(
                "Capture job retry job_id=%s raw_capture_id=%s attempt=%d ingest_ms=%.1f error=%s",
                job.id,
                job.raw_capture_id,
                job.attempt_count,
                duration_ms,
                exc,
            )
            return False

    def process_next_capture_job(self) -> bool:
        job = self.repositories.claim()
        if not job:
            return False
        return self._process_capture_job(job)

    def process_capture_job_by_id(self, job_id: int) -> bool:
        job = self.repositories.claim(job_id=job_id)
        if not job:
            return False
        return self._process_capture_job(job)

    def get_capture_job_stats(self) -> dict[str, int]:
        return self.repositories.stats()

    def _capture_ingest_worker(self, worker_index: int) -> None:
        logger.info(
            "Capture ingest worker started worker=%d poll_ms=%d capture_mode=%s",
            worker_index,
            self.settings.ingest_poll_ms,
            self.settings.capture_mode,
        )
        poll_seconds = self.settings.ingest_poll_ms / 1000
        while not self._worker_stop_event.is_set():
            try:
                processed = self.process_next_capture_job()
            except Exception:
                logger.exception("Capture ingest worker crashed during processing worker=%d", worker_index)
                processed = False
            if not processed:
                self._worker_stop_event.wait(poll_seconds)

    def start_background_workers(self) -> None:
        if self.settings.capture_mode != "async" or self.settings.ingest_workers <= 0:
            return
        with self._worker_lock:
            active_threads = [thread for thread in self._worker_threads if thread.is_alive()]
            if active_threads:
                self._worker_threads[:] = active_threads
                return
            self._worker_threads.clear()
            self._worker_stop_event.clear()
            for idx in range(self.settings.ingest_workers):
                thread = threading.Thread(
                    target=self._capture_ingest_worker,
                    args=(idx + 1,),
                    daemon=True,
                    name=f"openbrain-capture-{idx + 1}",
                )
                thread.start()
                self._worker_threads.append(thread)

    def stop_background_workers(self, timeout: float = 1.0) -> None:
        with self._worker_lock:
            if not self._worker_threads:
                return
            self._worker_stop_event.set()
            for thread in list(self._worker_threads):
                thread.join(timeout=timeout)
            self._worker_threads.clear()
