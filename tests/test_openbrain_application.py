from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from openbrain.application.service import OpenBrainApplication
from openbrain.domain.models import (
    CaptureJob,
    CapturePersistResult,
    GraphExtraction,
    GraphNode,
    ManagedMemoryCandidate,
    ManagedMemoryRecord,
    ManagedMemoryResolution,
)
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

    def capture_and_enqueue(self, thought, *, user_id, source, ingest_strategy, external_id, managed_kind_override):
        self.capture_calls.append(
            {
                "thought": thought,
                "user_id": user_id,
                "source": source,
                "ingest_strategy": ingest_strategy,
                "external_id": external_id,
                "managed_kind_override": managed_kind_override,
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
        self.search_calls = []

    def upsert_chunks(self, chunks):
        self.upsert_calls.append(chunks)
        return len(chunks)

    def search_vectors(self, query_embedding, *, user_id, limit=10, ingest_modes=None):
        self.search_calls.append(
            {
                "query_embedding": list(query_embedding),
                "user_id": user_id,
                "limit": limit,
                "ingest_modes": ingest_modes,
            }
        )
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


class FakeManagedRepo:
    def __init__(self):
        self.records = {}
        self.topic_index = {}
        self.insert_calls = []
        self.update_calls = []
        self.supersede_calls = []
        self.search_results = []
        self.search_calls = []
        self.list_calls = []
        self.clear_calls = []
        self._next_id = 1

    def find_active_by_topic(self, *, user_id, kind, topic_key):
        del user_id
        record_id = self.topic_index.get((kind, topic_key))
        if record_id is None or record_id not in self.records:
            return []
        return [self.records[record_id]]

    def insert_managed_memory(
        self,
        *,
        user_id,
        kind,
        topic,
        topic_key,
        canonical_text,
        embedding,
        embedding_provider,
        embedding_model,
        embedding_dims,
        metadata,
        raw_capture_ids,
    ):
        del embedding, embedding_provider, embedding_model, embedding_dims, raw_capture_ids
        record_id = self._next_id
        self._next_id += 1
        record = ManagedMemoryRecord(
            id=record_id,
            user_id=user_id,
            kind=kind,
            topic=topic,
            topic_key=topic_key,
            canonical_text=canonical_text,
            status="active",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            superseded_by=None,
            metadata=dict(metadata),
        )
        self.records[record_id] = record
        self.topic_index[(kind, topic_key)] = record_id
        self.insert_calls.append(record)
        return record_id

    def update_managed_memory(
        self,
        *,
        managed_memory_id,
        canonical_text,
        embedding,
        embedding_provider,
        embedding_model,
        embedding_dims,
        metadata,
        raw_capture_ids,
    ):
        del embedding, embedding_provider, embedding_model, embedding_dims, raw_capture_ids
        record = self.records[managed_memory_id]
        updated = ManagedMemoryRecord(
            id=record.id,
            user_id=record.user_id,
            kind=record.kind,
            topic=record.topic,
            topic_key=record.topic_key,
            canonical_text=canonical_text,
            status=record.status,
            created_at=record.created_at,
            updated_at=datetime.now(timezone.utc),
            superseded_by=record.superseded_by,
            metadata=dict(metadata),
        )
        self.records[managed_memory_id] = updated
        self.update_calls.append(updated)

    def supersede_managed_memories(self, *, managed_memory_ids, superseded_by):
        self.supersede_calls.append((list(managed_memory_ids), superseded_by))
        for managed_memory_id in managed_memory_ids:
            if managed_memory_id not in self.records:
                continue
            record = self.records[managed_memory_id]
            self.records[managed_memory_id] = ManagedMemoryRecord(
                id=record.id,
                user_id=record.user_id,
                kind=record.kind,
                topic=record.topic,
                topic_key=record.topic_key,
                canonical_text=record.canonical_text,
                status="superseded",
                created_at=record.created_at,
                updated_at=datetime.now(timezone.utc),
                superseded_by=superseded_by,
                metadata=record.metadata,
            )
            if superseded_by is None:
                self.topic_index.pop((record.kind, record.topic_key), None)

    def search_managed_memories(self, query_embedding, *, user_id, limit=5, kind=None):
        self.search_calls.append(
            {
                "query_embedding": list(query_embedding),
                "user_id": user_id,
                "limit": limit,
                "kind": kind,
            }
        )
        return list(self.search_results)

    def list_active_managed_memories(self, *, user_id, kind=None, limit=50):
        self.list_calls.append({"user_id": user_id, "kind": kind, "limit": limit})
        records = [
            record
            for record in self.records.values()
            if record.user_id == user_id and record.status == "active" and (kind is None or record.kind == kind)
        ]
        return records[:limit]

    def clear_managed_memories(self, *, user_id):
        self.clear_calls.append(user_id)
        self.records = {record_id: record for record_id, record in self.records.items() if record.user_id != user_id}
        self.topic_index = {
            key: record_id
            for key, record_id in self.topic_index.items()
            if record_id in self.records
        }


class FakeManagedExtractor:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def extract(self, text, *, forced_kind=None):
        self.calls.append({"text": text, "forced_kind": forced_kind})
        if forced_kind:
            filtered = [result for result in self.results if result.kind == forced_kind]
            if filtered:
                return filtered
        return list(self.results)


class FakeManagedResolver:
    def __init__(self, action="merge"):
        self.action = action
        self.calls = []

    def resolve(self, *, candidate, existing):
        self.calls.append({"candidate": candidate, "existing": list(existing)})
        return ManagedMemoryResolution(action=self.action, canonical_text=candidate.canonical_text)


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
        graph_name="brain_graph_v2",
        log_level="INFO",
    )
    return replace(settings, **overrides)


def make_app(**settings_overrides):
    repositories = FakeRepositories()
    vector_repo = FakeVectorRepo()
    graph_repo = FakeGraphRepo()
    managed_repo = FakeManagedRepo()
    managed_extractor = FakeManagedExtractor()
    managed_resolver = FakeManagedResolver()
    app = OpenBrainApplication(
        settings=make_settings(**settings_overrides),
        repositories=repositories,
        vector_repo=vector_repo,
        graph_repo=graph_repo,
        managed_repo=managed_repo,
        embedding_provider=FakeEmbeddingProvider(),
        bullet_extractor=FakeBulletExtractor(),
        graph_extractor=FakeGraphExtractor(),
        managed_extractor=managed_extractor,
        managed_resolver=managed_resolver,
    )
    return app, repositories, vector_repo, graph_repo, managed_repo, managed_extractor, managed_resolver


def test_capture_thought_rejects_empty_input():
    app, _, _, _, _, _, _ = make_app()
    assert app.capture_thought("   ") == "Warning: Please provide a non-empty thought."


def test_capture_thought_queues_with_auto_classifier():
    app, repositories, _, _, _, _, _ = make_app()
    result = app.capture_thought("docker run -p 8080:80 myimage")
    assert result == "Thought captured and queued for indexing."
    assert repositories.capture_calls[0]["source"] == "capture_thought"
    assert repositories.capture_calls[0]["ingest_strategy"] == "snippet"


def test_ingest_returns_duplicate_message():
    app, repositories, _, _, _, _, _ = make_app()
    repositories.persist_result = CapturePersistResult(raw_capture_id=9, job_id=10, duplicate=True)
    result = app.ingest("duplicate", source="chatgpt", external_id="conv-1")
    assert "Already ingested" in result


def test_ingest_sync_processes_job_inline():
    app, repositories, _, _, _, _, _ = make_app(capture_mode="sync")
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
    app, _, vector_repo, graph_repo, _, managed_extractor, _ = make_app()
    added, fact_added, managed_added = app.ingest_capture(
        thought="docker run -p 8080:80 myimage",
        user_id="default",
        raw_capture_id=5,
        source="capture_thought",
        ingest_strategy="snippet",
    )
    assert added == 1
    assert fact_added == 0
    assert managed_added == 0
    assert len(vector_repo.upsert_calls) == 1
    assert vector_repo.upsert_calls[0][0].ingest_mode == "raw"
    assert graph_repo.nodes == []
    assert managed_extractor.calls == []


def test_ingest_capture_personal_stores_raw_and_fact_chunks_and_graph():
    app, _, vector_repo, graph_repo, managed_repo, managed_extractor, _ = make_app()
    managed_extractor.results = [
        ManagedMemoryCandidate(
            kind="preference",
            topic="display style",
            canonical_text="Prefer dark mode because it reduces eye strain during long coding sessions",
        )
    ]
    added, fact_added, managed_added = app.ingest_capture(
        thought="I prefer dark mode because it reduces eye strain during long coding sessions.",
        user_id="default",
        raw_capture_id=8,
        source="capture_thought",
        ingest_strategy="personal",
    )
    assert added == 3
    assert fact_added == 1
    assert managed_added == 1
    assert len(vector_repo.upsert_calls) == 2
    assert vector_repo.upsert_calls[0][0].ingest_mode == "raw"
    assert vector_repo.upsert_calls[1][0].ingest_mode == "fact"
    assert graph_repo.nodes[0]["properties"]["user_id"] == "default"
    assert managed_repo.insert_calls[0].kind == "preference"


def test_process_next_capture_job_marks_done_on_success():
    app, repositories, _, _, _, _, _ = make_app()
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
    app, repositories, _, _, _, _, _ = make_app()
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
    app, repositories, vector_repo, graph_repo, _, _, _ = make_app()
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
    app, _, _, _, _, _, _ = make_app()
    result = app.search_brain("unmatched query", debug=True)
    assert "=== Search Debug ===" in result
    assert "threshold:" in result
    assert "final_ranking:" in result


def test_search_brain_answers_quote_style_queries_with_matching_passage():
    app, repositories, vector_repo, _, managed_repo, _, _ = make_app()
    text = (
        "We have made you into nations and tribes so that you may know one another. "
        "Everything is perishing except His Face. (28:88) Diversity is not to be worshipped."
    )
    vector_repo.search_results = [
        {
            "id": 1,
            "content": text,
            "metadata": {"origin": "capture_thought", "ingest_mode": "raw"},
            "created_at": datetime.now(timezone.utc),
            "source": "capture_thought",
            "score": 0.01,
        }
    ]
    repositories.raw_matches = []

    result = app.search_brain("What does 28:88 say", debug=False)
    assert "28:88 says" in result
    assert "Everything is perishing except His Face" in result
    assert "49:13" not in result
    assert vector_repo.search_calls[0]["ingest_modes"] == ("raw",)
    assert managed_repo.search_calls == []


def test_ingest_capture_personal_preserves_paragraph_boundaries_for_raw_chunks():
    app, _, vector_repo, _, _, _, _ = make_app()
    thought = (
        "First paragraph with a complete idea.\n\n"
        "Second paragraph with another distinct idea and a citation (28:88).\n\n"
        "Third paragraph closing the thought."
    )
    added, fact_added, managed_added = app.ingest_capture(
        thought=thought,
        user_id="default",
        raw_capture_id=9,
        source="capture_thought",
        ingest_strategy="personal",
    )
    assert added >= 3
    assert fact_added >= 1
    assert managed_added == 0
    raw_chunks = vector_repo.upsert_calls[0]
    assert [chunk.content for chunk in raw_chunks][:3] == [
        "First paragraph with a complete idea.",
        "Second paragraph with another distinct idea and a citation (28:88).",
        "Third paragraph closing the thought.",
    ]


def test_ingest_accepts_explicit_managed_kind_override():
    app, repositories, _, _, _, _, _ = make_app()
    result = app.ingest(
        "Act as an intellectual sparring partner.",
        source="import",
        managed_kind="directive",
    )
    assert "Ingested" in result
    assert repositories.capture_calls[0]["managed_kind_override"] == "directive"


def test_search_brain_prefers_managed_memory_for_general_queries():
    app, repositories, vector_repo, graph_repo, managed_repo, _, _ = make_app()
    managed_repo.search_results = [
        {
            "id": 1,
            "kind": "directive",
            "topic": "conversation style",
            "topic_key": "conversation-style",
            "canonical_text": (
                "Act as an intellectual sparring partner. Analyze assumptions, provide counterpoints, "
                "test reasoning, offer alternatives, and prioritize truth over agreement"
            ),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "score": 0.01,
            "metadata": {},
        }
    ]
    vector_repo.search_results = [
        {
            "id": 2,
            "content": "Provide counterpoints.",
            "metadata": {"origin": "capture_thought", "ingest_mode": "fact"},
            "created_at": datetime.now(timezone.utc),
            "source": "capture_thought",
            "score": 0.02,
        }
    ]
    repositories.raw_matches = []
    graph_repo.search_results = {"nodes": [], "edges": []}
    result = app.search_brain("what counterpoints", debug=False)
    assert "Based on your active directive" in result
    assert "Provide counterpoints.." not in result


def test_get_active_memories_groups_records():
    app, _, _, _, managed_repo, _, _ = make_app()
    managed_repo.insert_managed_memory(
        user_id="default",
        kind="directive",
        topic="conversation style",
        topic_key="conversation-style",
        canonical_text="Act as an intellectual sparring partner",
        embedding=[1.0, 0.1, 0.01],
        embedding_provider="test",
        embedding_model="test",
        embedding_dims=3,
        metadata={},
        raw_capture_ids=[1],
    )
    managed_repo.insert_managed_memory(
        user_id="default",
        kind="preference",
        topic="response style",
        topic_key="response-style",
        canonical_text="Prefer concise answers unless I ask for detail",
        embedding=[1.0, 0.1, 0.01],
        embedding_provider="test",
        embedding_model="test",
        embedding_dims=3,
        metadata={},
        raw_capture_ids=[2],
    )
    result = app.get_active_memories()
    assert "Active directives" in result
    assert "Active preferences" in result


def test_rebuild_managed_memories_can_reset_and_replay():
    app, repositories, _, _, managed_repo, managed_extractor, _ = make_app()
    repositories.list_for_backfill = MagicMock(
        return_value=[
            type(
                "StoredRaw",
                (),
                {
                    "id": 1,
                    "user_id": "default",
                    "source": "capture_thought",
                    "content": "Act as an intellectual sparring partner.",
                    "ingest_strategy": "personal",
                    "managed_kind_override": None,
                    "created_at": datetime.now(timezone.utc),
                },
            )()
        ]
    )
    managed_extractor.results = [
        ManagedMemoryCandidate(
            kind="directive",
            topic="conversation style",
            canonical_text="Act as an intellectual sparring partner",
        )
    ]
    result = app.rebuild_managed_memories(reset=True)
    assert "Managed updates applied: 1" in result
    assert managed_repo.clear_calls == ["default"]
