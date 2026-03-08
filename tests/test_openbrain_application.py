from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from openbrain.application.service import OpenBrainApplication
from openbrain.domain.models import CaptureJob, CapturePersistResult, GraphExtraction, GraphNode
from openbrain.domain.text import detect_auto_ingest_strategy
from openbrain.settings import DatabaseSettings, EmbeddingSettings, LLMSettings, OpenBrainSettings


class FakeRepositories:
    def __init__(self):
        self.persist_result = CapturePersistResult(raw_capture_id=11, job_id=21, duplicate=False)
        self.stats_result = {"queued": 1, "processing": 0, "done": 0, "retry": 0, "failed": 0}
        self.claim_result = None
        self.raw_matches = []
        self.done_jobs = []
        self.retry_jobs = []
        self.capture_calls = []

    def capture_and_enqueue(self, thought, *, user_id, source, ingest_strategy, external_id):
        self.capture_calls.append(
            {
                "thought": thought,
                "user_id": user_id,
                "source": source,
                "ingest_strategy": ingest_strategy,
                "external_id": external_id,
            }
        )
        return self.persist_result

    def stats(self):
        return dict(self.stats_result)

    def claim(self, *, job_id=None):
        del job_id
        return self.claim_result

    def mark_done(self, job):
        self.done_jobs.append(job)

    def mark_retry(self, job, *, error, retry_delays):
        self.retry_jobs.append((job, str(error), retry_delays))

    def search_raw(self, query, *, user_id, limit=8):
        del query, user_id, limit
        return list(self.raw_matches)


class FakeVectorRepo:
    def __init__(self):
        self.upsert_calls = []
        self.search_results = []

    def upsert_chunks(self, chunks):
        self.upsert_calls.append(chunks)
        return len(chunks)

    def search_vectors(self, query_embedding, *, user_id, limit=10):
        del query_embedding, user_id, limit
        return list(self.search_results)


class FakeGraphRepo:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.search_results = {"nodes": [], "edges": []}

    def add_nodes(self, nodes):
        self.nodes.extend(nodes)

    def add_edges(self, edges):
        self.edges.extend(edges)

    def search(self, query, *, limit=5):
        del query, limit
        return self.search_results


class FakeEmbeddingProvider:
    provider_name = "test-embed"
    model_name = "test-embed-model"
    embedding_dims = 3

    def embed(self, text: str):
        value = float(len(text.split()))
        return [value, value / 10.0, value / 100.0]


class FakeBulletExtractor:
    def __init__(self, bullets=None):
        self.bullets = bullets or ["Dark mode reduces eye strain during long coding sessions"]

    def extract(self, text, *, max_bullets=3):
        del text, max_bullets
        return list(self.bullets)


class FakeGraphExtractor:
    def __init__(self, extraction=None):
        self.extraction = extraction or GraphExtraction(
            nodes=[GraphNode(name="amer", label="Person"), GraphNode(name="dark mode", label="Preference")],
            edges=[],
        )

    def extract(self, text, *, user_id):
        del text, user_id
        return self.extraction


def make_settings(**overrides) -> OpenBrainSettings:
    settings = OpenBrainSettings(
        database=DatabaseSettings(
            dbname="open_brain",
            user="brain_user",
            password="secret",
            host="localhost",
            port=5432,
        ),
        llm=LLMSettings(
            provider="openai",
            model="gpt-4o-mini",
            api_key="key",
            base_url=None,
            openrouter_site_url=None,
            openrouter_app_name=None,
        ),
        embedding=EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="key",
            base_url=None,
            dims=1536,
        ),
        capture_mode="async",
        ingest_workers=1,
        ingest_poll_ms=250,
        memory_retention_months=12,
        graph_name="brain_graph_v2",
        log_level="INFO",
    )
    return replace(settings, **overrides)


def make_app(**settings_overrides):
    repositories = FakeRepositories()
    vector_repo = FakeVectorRepo()
    graph_repo = FakeGraphRepo()
    app = OpenBrainApplication(
        settings=make_settings(**settings_overrides),
        repositories=repositories,
        vector_repo=vector_repo,
        graph_repo=graph_repo,
        embedding_provider=FakeEmbeddingProvider(),
        bullet_extractor=FakeBulletExtractor(),
        graph_extractor=FakeGraphExtractor(),
    )
    return app, repositories, vector_repo, graph_repo


def test_capture_thought_rejects_empty_input():
    app, _, _, _ = make_app()
    assert app.capture_thought("   ") == "Warning: Please provide a non-empty thought."


def test_capture_thought_queues_with_auto_classifier():
    app, repositories, _, _ = make_app()
    result = app.capture_thought("docker run -p 8080:80 myimage")
    assert result == "Thought captured and queued for indexing."
    assert repositories.capture_calls[0]["source"] == "capture_thought"
    assert repositories.capture_calls[0]["ingest_strategy"] == "snippet"


def test_ingest_returns_duplicate_message():
    app, repositories, _, _ = make_app()
    repositories.persist_result = CapturePersistResult(raw_capture_id=9, job_id=10, duplicate=True)
    result = app.ingest("duplicate", source="chatgpt", external_id="conv-1")
    assert "Already ingested" in result


def test_ingest_sync_processes_job_inline():
    app, repositories, _, _ = make_app(capture_mode="sync")
    app.process_capture_job_by_id = MagicMock(return_value=True)
    result = app.ingest("hello", source="import")
    assert result == "Ingested and indexed."
    app.process_capture_job_by_id.assert_called_once_with(21)
    assert repositories.capture_calls[0]["ingest_strategy"] == "personal"


def test_auto_strategy_still_detects_prose_and_code():
    assert detect_auto_ingest_strategy(
        "I prefer dark mode in all my projects because it is easier on my eyes.",
        source="capture_thought",
    ) == "personal"
    assert detect_auto_ingest_strategy("docker run -p 8080:80 myimage", source="capture_thought") == "snippet"


def test_ingest_capture_snippet_stores_raw_chunks_only():
    app, _, vector_repo, graph_repo = make_app()
    added, fact_added = app.ingest_capture(
        thought="docker run -p 8080:80 myimage",
        user_id="default",
        raw_capture_id=5,
        source="capture_thought",
        ingest_strategy="snippet",
    )
    assert added == 1
    assert fact_added == 0
    assert len(vector_repo.upsert_calls) == 1
    assert vector_repo.upsert_calls[0][0].ingest_mode == "raw"
    assert graph_repo.nodes == []


def test_ingest_capture_personal_stores_raw_and_fact_chunks_and_graph():
    app, _, vector_repo, graph_repo = make_app()
    added, fact_added = app.ingest_capture(
        thought="I prefer dark mode because it reduces eye strain during long coding sessions.",
        user_id="default",
        raw_capture_id=8,
        source="capture_thought",
        ingest_strategy="personal",
    )
    assert added == 2
    assert fact_added == 1
    assert len(vector_repo.upsert_calls) == 2
    assert vector_repo.upsert_calls[0][0].ingest_mode == "raw"
    assert vector_repo.upsert_calls[1][0].ingest_mode == "fact"
    assert graph_repo.nodes[0]["properties"]["user_id"] == "default"


def test_process_next_capture_job_marks_done_on_success():
    app, repositories, _, _ = make_app()
    repositories.claim_result = CaptureJob(
        id=7,
        raw_capture_id=11,
        user_id="default",
        thought="I prefer dark mode because it reduces eye strain during long coding sessions.",
        attempt_count=1,
        source="capture_thought",
        ingest_strategy="personal",
    )
    assert app.process_next_capture_job() is True
    assert repositories.done_jobs[0].id == 7
    assert repositories.retry_jobs == []


def test_process_next_capture_job_marks_retry_on_failure():
    app, repositories, _, _ = make_app()
    repositories.claim_result = CaptureJob(
        id=7,
        raw_capture_id=11,
        user_id="default",
        thought="hello",
        attempt_count=1,
        source="capture_thought",
        ingest_strategy="personal",
    )
    app.embedding_provider = MagicMock()
    app.embedding_provider.embed.side_effect = RuntimeError("embed failed")
    assert app.process_next_capture_job() is False
    assert repositories.done_jobs == []
    assert repositories.retry_jobs[0][1] == "embed failed"


def test_search_brain_merges_vector_raw_and_graph_results():
    app, repositories, vector_repo, graph_repo = make_app()
    vector_repo.search_results = [
        {
            "id": 1,
            "content": "I prefer dark mode in all my projects.",
            "metadata": {"origin": "capture_thought", "ingest_mode": "fact"},
            "created_at": datetime.now(timezone.utc),
            "source": "capture_thought",
            "score": 0.05,
        }
    ]
    repositories.raw_matches = [
        type(
            "Raw",
            (),
            {
                "id": 9,
                "content": "I prefer dark mode in all my projects.",
                "created_at": datetime.now(timezone.utc),
                "content_len": 39,
                "source": "capture_thought",
            },
        )()
    ]
    graph_repo.search_results = {
        "nodes": [
            {"id": 1, "properties": {"name": "amer"}},
            {"id": 2, "properties": {"name": "dark mode"}},
        ],
        "edges": [{"start_id": 1, "end_id": 2, "label": "PREFERS"}],
    }

    result = app.search_brain("dark mode", debug=False)
    assert "Answer:" in result
    assert "Sources:" in result
    assert "dark mode" in result.lower()


def test_search_brain_debug_includes_threshold_and_ranking():
    app, _, _, _ = make_app()
    result = app.search_brain("unmatched query", debug=True)
    assert "=== Search Debug ===" in result
    assert "threshold:" in result
    assert "final_ranking:" in result
