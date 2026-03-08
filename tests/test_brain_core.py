from __future__ import annotations

from unittest.mock import MagicMock, patch

import brain_core


def test_capture_thought_delegates_to_application():
    app = MagicMock()
    app.capture_thought.return_value = "ok"
    with patch.object(brain_core, "get_app", return_value=app):
        assert brain_core.capture_thought("remember this") == "ok"
    app.capture_thought.assert_called_once_with("remember this")


def test_ingest_delegates_to_application():
    app = MagicMock()
    app.ingest.return_value = "queued"
    with patch.object(brain_core, "get_app", return_value=app):
        assert brain_core.ingest("hello", source="chatgpt", external_id="conv-1") == "queued"
    app.ingest.assert_called_once_with("hello", source="chatgpt", external_id="conv-1")


def test_search_brain_delegates_to_application():
    app = MagicMock()
    app.search_brain.return_value = "result"
    with patch.object(brain_core, "get_app", return_value=app):
        assert brain_core.search_brain("dark mode", debug=True) == "result"
    app.search_brain.assert_called_once_with("dark mode", debug=True)


def test_worker_helpers_delegate_to_application():
    app = MagicMock()
    app.process_next_capture_job.return_value = True
    app.get_capture_job_stats.return_value = {"queued": 1}
    with patch.object(brain_core, "get_app", return_value=app):
        brain_core.start_background_workers()
        assert brain_core.process_next_capture_job() is True
        assert brain_core.get_capture_job_stats() == {"queued": 1}
        assert brain_core._process_capture_job_by_id(7) is app.process_capture_job_by_id.return_value
        brain_core.stop_background_workers(timeout=2.5)
    app.start_background_workers.assert_called_once_with()
    app.process_next_capture_job.assert_called_once_with()
    app.get_capture_job_stats.assert_called_once_with()
    app.process_capture_job_by_id.assert_called_once_with(7)
    app.stop_background_workers.assert_called_once_with(timeout=2.5)
