"""
Unit tests for the async capture queue in brain_core.py.
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def brain_core_module(monkeypatch):
    """Import brain_core with external services mocked."""
    sys.modules.pop("brain_core", None)

    fake_memory = MagicMock()
    fake_memory.add = MagicMock(return_value=[{"id": 1}])
    fake_memory.search = MagicMock(return_value={"results": [], "relations": []})

    mem0_module = types.ModuleType("mem0")

    class FakeMemory:
        @staticmethod
        def from_config(_config):
            return fake_memory

    mem0_module.Memory = FakeMemory
    monkeypatch.setitem(sys.modules, "mem0", mem0_module)

    age_provider_module = types.ModuleType("age_provider")
    age_provider_module.ApacheAGEProvider = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "age_provider", age_provider_module)

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    monkeypatch.setenv("POSTGRES_PASSWORD", "test-password")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENBRAIN_CAPTURE_MODE", "async")
    monkeypatch.setenv("OPENBRAIN_INGEST_WORKERS", "1")
    monkeypatch.setenv("OPENBRAIN_INGEST_POLL_MS", "250")

    with patch("psycopg2.pool.ThreadedConnectionPool") as MockPool:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__exit__.return_value = False
        mock_conn.closed = False

        pool_instance = MockPool.return_value
        pool_instance.getconn.return_value = mock_conn

        module = importlib.import_module("brain_core")
        yield module, fake_memory

        module.stop_background_workers()
        sys.modules.pop("brain_core", None)


class TestCaptureThought:
    def test_capture_thought_rejects_empty_input(self, brain_core_module):
        module, fake_memory = brain_core_module
        fake_memory.add.reset_mock()

        result = module.capture_thought("   ")

        assert result == "Warning: Please provide a non-empty thought."
        fake_memory.add.assert_not_called()

    def test_capture_thought_queues_without_inline_ingest(self, brain_core_module):
        module, fake_memory = brain_core_module
        fake_memory.add.reset_mock()

        with patch.object(module, "_CAPTURE_MODE", "async"), patch.object(
            module,
            "_capture_raw_and_enqueue",
            return_value=module.CapturePersistResult(raw_capture_id=101, job_id=202),
        ) as mock_capture, patch.object(
            module,
            "get_capture_job_stats",
            return_value={"queued": 1, "processing": 0, "failed": 0},
        ), patch.object(module, "_process_capture_job_by_id") as mock_process:
            result = module.capture_thought("remember this")

        assert result == "Thought captured and queued for indexing."
        mock_capture.assert_called_once_with("remember this", user_id="default", source="capture_thought")
        mock_process.assert_not_called()
        fake_memory.add.assert_not_called()

    def test_capture_thought_auto_detects_shell_command_as_snippet(self, brain_core_module):
        module, _ = brain_core_module

        with patch.object(module, "_CAPTURE_MODE", "async"), patch.object(
            module,
            "_capture_raw_and_enqueue",
            return_value=module.CapturePersistResult(raw_capture_id=11, job_id=22),
        ) as mock_capture, patch.object(
            module,
            "get_capture_job_stats",
            return_value={"queued": 1, "processing": 0, "failed": 0},
        ):
            module.capture_thought("docker run -p 8080:80 myimage")

        assert mock_capture.call_args[1]["source"] == "capture_thought"
        assert "ingest_strategy" not in mock_capture.call_args[1]


class TestIngest:
    def test_ingest_rejects_empty_content(self, brain_core_module):
        module, _ = brain_core_module
        assert "Warning" in module.ingest("  ", source="import")

    def test_ingest_queues_with_source(self, brain_core_module):
        module, _ = brain_core_module
        with patch.object(module, "_CAPTURE_MODE", "async"), patch.object(
            module,
            "_capture_raw_and_enqueue",
            return_value=module.CapturePersistResult(raw_capture_id=301, job_id=401),
        ) as mock_enqueue, patch.object(
            module,
            "get_capture_job_stats",
            return_value={"queued": 1, "processing": 0, "failed": 0},
        ):
            result = module.ingest(
                "Chat export line",
                source="chatgpt",
                external_id="conv-123",
            )
        assert "queued" in result or "Ingested" in result
        mock_enqueue.assert_called_once()
        call_kw = mock_enqueue.call_args[1]
        assert call_kw["user_id"] == "default"
        assert call_kw["source"] == "chatgpt"
        assert "ingest_strategy" not in call_kw
        assert call_kw["external_id"] == "conv-123"

    def test_ingest_uses_auto_classifier_path(self, brain_core_module):
        module, _ = brain_core_module
        with patch.object(module, "_CAPTURE_MODE", "async"), patch.object(
            module,
            "_capture_raw_and_enqueue",
            return_value=module.CapturePersistResult(raw_capture_id=12, job_id=13),
        ) as mock_enqueue, patch.object(
            module,
            "get_capture_job_stats",
            return_value={"queued": 1, "processing": 0, "failed": 0},
        ):
            module.ingest("git status", source="import")

        assert "ingest_strategy" not in mock_enqueue.call_args[1]

    def test_ingest_skips_when_external_id_already_exists(self, brain_core_module):
        module, _ = brain_core_module
        with patch.object(
            module,
            "_capture_raw_and_enqueue",
            return_value=module.CapturePersistResult(raw_capture_id=77, job_id=88, duplicate=True),
        ) as mock_enqueue, patch.object(module, "get_capture_job_stats") as mock_stats:
            result = module.ingest(
                "Duplicate",
                source="claude",
                external_id="conv-456",
            )
        assert "Already ingested" in result or "duplicate" in result.lower()
        mock_enqueue.assert_called_once()
        mock_stats.assert_not_called()

    def test_capture_raw_and_enqueue_marks_duplicate_atomically(self, brain_core_module):
        module, _ = brain_core_module
        cursor = MagicMock()
        cursor.fetchone.return_value = (55, False)

        @contextmanager
        def fake_db_cursor(*, commit: bool = False):
            yield None, cursor

        with patch.object(module, "_db_cursor", fake_db_cursor), patch.object(
            module, "_enqueue_capture_job", return_value=66
        ) as mock_enqueue:
            result = module._capture_raw_and_enqueue(
                "Duplicate",
                user_id="default",
                source="chatgpt",
                ingest_strategy="personal",
                external_id="conv-1",
            )

        assert result.raw_capture_id == 55
        assert result.job_id == 66
        assert result.duplicate is True
        mock_enqueue.assert_called_once_with(55, user_id="default", cur=cursor)

    def test_auto_strategy_detects_prose_as_personal(self, brain_core_module):
        module, _ = brain_core_module
        result = module._detect_auto_ingest_strategy(
            "I prefer dark mode in all my projects because it is easier on my eyes.",
            source="capture_thought",
        )
        assert result == "personal"

    def test_auto_strategy_detects_code_as_snippet(self, brain_core_module):
        module, _ = brain_core_module
        result = module._detect_auto_ingest_strategy(
            "docker run -p 8080:80 myimage",
            source="capture_thought",
        )
        assert result == "snippet"


class TestWorkers:
    def test_worker_success_path_marks_job_done(self, brain_core_module):
        module, _ = brain_core_module
        job = module.CaptureJob(id=7, raw_capture_id=11, user_id="default", thought="remember this", attempt_count=1)

        with patch.object(module, "_claim_capture_job", return_value=job), patch.object(
            module,
            "_ingest_capture",
            return_value=(1, 0),
        ) as mock_ingest, patch.object(module, "_mark_capture_job_done") as mock_done, patch.object(
            module,
            "_mark_capture_job_retry",
        ) as mock_retry:
            assert module.process_next_capture_job() is True

        mock_ingest.assert_called_once_with(
            thought="remember this",
            user_id="default",
            raw_capture_id=11,
            source="capture_thought",
            ingest_strategy="personal",
        )
        mock_done.assert_called_once()
        mock_retry.assert_not_called()

    def test_zero_add_triggers_fallback_only_in_worker_path(self, brain_core_module):
        module, fake_memory = brain_core_module
        fake_memory.add.reset_mock()
        fake_memory.add.return_value = {}

        with patch.object(module, "_fallback_ingest_facts", return_value=2) as mock_fallback:
            added_count, fallback_added = module._ingest_capture(
                "I prefer dark mode in all my projects.",
                user_id="default",
                raw_capture_id=44,
            )

        assert added_count == 0
        assert fallback_added == 2
        mock_fallback.assert_called_once_with(
            "I prefer dark mode in all my projects.",
            user_id="default",
            raw_capture_id=44,
            origin="capture_thought",
        )

    def test_snippet_ingest_uses_infer_false_and_skips_fallback(self, brain_core_module):
        module, fake_memory = brain_core_module
        fake_memory.add.reset_mock()
        fake_memory.add.return_value = [{"id": "s1"}]

        with patch.object(module, "_fallback_ingest_facts") as mock_fallback:
            added_count, fallback_added = module._ingest_capture(
                "docker run -p 8080:80 myimage",
                user_id="default",
                raw_capture_id=99,
                ingest_strategy="snippet",
            )

        assert added_count == 1
        assert fallback_added == 0
        mock_fallback.assert_not_called()
        # Unified: snippet = single raw store (no enrichment)
        assert fake_memory.add.call_count == 1
        call_kw = fake_memory.add.call_args[1]
        assert call_kw.get("infer") is False
        assert call_kw.get("metadata", {}).get("ingest_mode") == "raw"

    def test_retry_schedules_backoff_then_terminal_failure(self, brain_core_module):
        module, _ = brain_core_module
        calls: list[tuple[str, tuple]] = []

        @contextmanager
        def fake_db_cursor(*, commit: bool = False):
            cursor = MagicMock()

            def record(sql, params=None):
                calls.append((sql, params))

            cursor.execute.side_effect = record
            yield None, cursor

        with patch.object(module, "_db_cursor", fake_db_cursor):
            module._mark_capture_job_retry(
                module.CaptureJob(id=1, raw_capture_id=2, user_id="default", thought="x", attempt_count=1),
                RuntimeError("transient"),
                12.5,
            )
            module._mark_capture_job_retry(
                module.CaptureJob(id=2, raw_capture_id=3, user_id="default", thought="y", attempt_count=4),
                RuntimeError("permanent"),
                15.0,
            )

        assert calls[0][1][0] == "retry"
        assert calls[0][1][1] == 60
        assert calls[1][1][0] == "failed"

    def test_duplicate_enqueue_uses_upsert_by_raw_capture_id(self, brain_core_module):
        module, _ = brain_core_module
        cursor = MagicMock()
        cursor.fetchone.return_value = (11,)

        first = module._enqueue_capture_job(5, user_id="default", cur=cursor)
        second = module._enqueue_capture_job(5, user_id="default", cur=cursor)

        assert first == 11
        assert second == 11
        assert cursor.execute.call_count == 2
        assert "ON CONFLICT (raw_capture_id)" in cursor.execute.call_args_list[0][0][0]


class TestSearch:
    def test_search_uses_single_brain_id_for_mem0(self, brain_core_module):
        module, fake_memory = brain_core_module
        fake_memory.search.return_value = {"results": [], "relations": []}

        with patch.object(module, "_search_raw_captures", return_value=[]):
            module.search_brain("dark mode", debug=False)

        fake_memory.search.assert_called_once_with(query="dark mode", user_id="default", limit=10)

    def test_search_finds_recent_raw_capture_before_async_ingest(self, brain_core_module):
        module, fake_memory = brain_core_module
        fake_memory.search.return_value = {"results": [], "relations": []}

        with patch.object(
            module,
            "_search_raw_captures",
            return_value=[
                {
                    "id": 9,
                    "content": "I prefer dark mode in all my projects.",
                    "created_at": datetime.now(timezone.utc),
                    "content_len": 39,
                }
            ],
        ):
            result = module.search_brain("dark mode", debug=False)

        assert "dark mode" in result.lower()
        # Source line shows evidence type and origin (e.g. [raw] or [raw • import])
        assert "raw" in result and "Sources:" in result

    def test_retrieval_sanity_golden_query_expected_memory_in_results(self, brain_core_module):
        """Retrieval sanity: for a golden query, expected memory text appears in answer or sources (recall-style)."""
        module, fake_memory = brain_core_module
        # Vector result: one semantic memory we expect to retrieve for the query
        fake_memory.search.return_value = {
            "results": [
                {
                    "memory": "Alice uses Python for data science and ML pipelines.",
                    "score": 0.12,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": {"origin": "mcp"},
                }
                ],
            "relations": [],
        }
        with patch.object(
            module,
            "_search_raw_captures",
            return_value=[
                {
                    "id": 1,
                    "content": "Project X deadline is March 15.",
                    "created_at": datetime.now(timezone.utc),
                    "content_len": 30,
                    "source": "notes_pc",
                }
            ],
        ):
            result = module.search_brain("What does Alice use for data science?", debug=False)
        # Recall-style: the expected memory (Python / Alice) should appear in the response
        assert "Python" in result or "Alice" in result
        assert "Sources:" in result

    def test_retrieval_sanity_debug_output_includes_threshold_and_ranking(self, brain_core_module):
        """Retrieval sanity: debug mode exposes threshold and candidate/ranking info for manual inspection."""
        module, fake_memory = brain_core_module
        fake_memory.search.return_value = {"results": [], "relations": []}
        with patch.object(module, "_search_raw_captures", return_value=[]):
            result = module.search_brain("test query", debug=True)
        assert "threshold" in result.lower()
        assert "candidates" in result.lower() or "final_ranking" in result.lower() or "dropped" in result.lower()
