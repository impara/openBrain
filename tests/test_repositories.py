from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from openbrain.domain.models import MemoryChunk
from openbrain.infrastructure.repositories import OpenBrainRepositories
from openbrain.settings import DatabaseSettings, EmbeddingSettings, LLMSettings, OpenBrainSettings


def make_settings(dims=1536):
    return OpenBrainSettings(
        database=DatabaseSettings(
            dbname="db",
            user="user",
            password="pass",
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
            dims=dims,
        ),
        capture_mode="async",
        ingest_workers=1,
        ingest_poll_ms=250,
        graph_name="brain_graph_v2",
        log_level="INFO",
    )


def test_ensure_infrastructure_rejects_existing_dimension_mismatch():
    cursor = MagicMock()
    cursor.fetchone.side_effect = [
        ("memory_store.memory_chunks",),
        ("vector(768)",),
        (None,),
    ]

    @contextmanager
    def fake_cursor(*, commit=False):
        del commit
        yield None, cursor

    db = MagicMock()
    db.cursor = fake_cursor
    repo = OpenBrainRepositories(db, make_settings(dims=1536))
    with pytest.raises(ValueError, match="expected 'vector\\(1536\\)'"):
        repo.ensure_infrastructure()


def test_upsert_chunks_uses_pgvector_literal_and_metadata_json():
    cursor = MagicMock()

    @contextmanager
    def fake_cursor(*, commit=False):
        del commit
        yield None, cursor

    db = MagicMock()
    db.cursor = fake_cursor
    repo = OpenBrainRepositories(db, make_settings())
    chunk = MemoryChunk(
        raw_capture_id=1,
        user_id="default",
        source="capture_thought",
        chunk_index=1,
        content="hello world",
        ingest_mode="raw",
        embedding=[0.1, 0.2, 0.3],
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dims=1536,
        metadata={"origin": "capture_thought"},
    )
    assert repo.upsert_chunks([chunk]) == 1
    params = cursor.execute.call_args.args[1]
    assert params[6] == "[0.1,0.2,0.3]"


def test_capture_and_enqueue_persists_managed_kind_override():
    cursor = MagicMock()
    cursor.fetchone.side_effect = [(1, True), (2,)]

    @contextmanager
    def fake_cursor(*, commit=False):
        del commit
        yield None, cursor

    db = MagicMock()
    db.cursor = fake_cursor
    repo = OpenBrainRepositories(db, make_settings())
    repo.capture_and_enqueue(
        "Act as an intellectual sparring partner.",
        user_id="default",
        source="import",
        ingest_strategy="personal",
        external_id=None,
        managed_kind_override="directive",
    )
    params = cursor.execute.call_args_list[0].args[1]
    assert params[-1] == "directive"


def test_search_vectors_returns_runtime_chunk_shape():
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (
            1,
            "dark mode",
            {"origin": "capture_thought"},
            datetime.now(timezone.utc),
            "capture_thought",
            "fact",
            0.04,
        )
    ]

    @contextmanager
    def fake_cursor(*, commit=False):
        del commit
        yield None, cursor

    db = MagicMock()
    db.cursor = fake_cursor
    repo = OpenBrainRepositories(db, make_settings())
    result = repo.search_vectors([0.1, 0.2], user_id="default", limit=5)
    assert result[0]["content"] == "dark mode"
    assert result[0]["score"] == 0.04
    assert result[0]["ingest_mode"] == "fact"


def test_search_vectors_can_filter_ingest_modes():
    cursor = MagicMock()
    cursor.fetchall.return_value = []

    @contextmanager
    def fake_cursor(*, commit=False):
        del commit
        yield None, cursor

    db = MagicMock()
    db.cursor = fake_cursor
    repo = OpenBrainRepositories(db, make_settings())
    repo.search_vectors([0.1, 0.2], user_id="default", limit=5, ingest_modes=("raw",))
    sql = cursor.execute.call_args_list[1].args[0]
    params = cursor.execute.call_args_list[1].args[1]
    assert "ingest_mode = ANY(%s)" in sql
    assert params[2] == ["raw"]


def test_search_managed_memories_returns_active_record_shape():
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (
            1,
            "directive",
            "conversation style",
            "conversation-style",
            "Act as an intellectual sparring partner",
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
            0.03,
            {"topic": "conversation style"},
        )
    ]

    @contextmanager
    def fake_cursor(*, commit=False):
        del commit
        yield None, cursor

    db = MagicMock()
    db.cursor = fake_cursor
    repo = OpenBrainRepositories(db, make_settings())
    result = repo.search_managed_memories([0.1, 0.2], user_id="default", limit=5)
    assert result[0].kind == "directive"
    assert result[0].canonical_text == "Act as an intellectual sparring partner"
