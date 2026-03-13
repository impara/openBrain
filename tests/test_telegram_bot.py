"""
test_telegram_bot.py — Unit tests for the Telegram bot handlers.

All external dependencies (brain_core, telegram) are mocked so tests
run without the full stack or a real bot token.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ── Mock brain_core before importing telegram_bot ──
brain_core_mock = MagicMock()
brain_core_mock.capture_thought = MagicMock(
    return_value="Thought captured and queued for indexing."
)
brain_core_mock.get_active_memories = MagicMock(
    return_value="Active directives:\n- Act as an intellectual sparring partner"
)
brain_core_mock.search_brain = MagicMock(
    return_value="=== Retrieved Brain Context ===\n- Test memory\n"
)
brain_core_mock.start_background_workers = MagicMock()
sys.modules.setdefault("brain_core", brain_core_mock)

# Must set token before import
with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token", "POSTGRES_PASSWORD": "test"}):
    from telegram_bot import (
        start_handler,
        help_handler,
        remember_handler,
        profile_handler,
        search_handler,
        search_debug_handler,
        text_handler,
    )


# ── Fixtures ──────────────────────────────────

@pytest.fixture
def mock_update():
    """Create a mock Telegram Update object."""
    update = MagicMock()
    update.effective_user.id = 12345
    update.message.reply_text = AsyncMock()
    update.message.text = "Hello world"
    return update


@pytest.fixture
def mock_context():
    """Create a mock Telegram context."""
    ctx = MagicMock()
    ctx.args = []
    return ctx


# ── Tests ─────────────────────────────────────

class TestStartHandler:
    """Test /start command."""

    @pytest.mark.asyncio
    async def test_sends_welcome(self, mock_update, mock_context):
        await start_handler(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once()
        text = mock_update.message.reply_text.call_args[0][0]
        assert "OpenBrain" in text
        assert "/remember" in text
        assert "/snippet" not in text
        assert "/search" in text
        assert "/search_debug" in text
        assert "/profile" in text


class TestHelpHandler:
    """Test /help command."""

    @pytest.mark.asyncio
    async def test_sends_help(self, mock_update, mock_context):
        await help_handler(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once()
        text = mock_update.message.reply_text.call_args[0][0]
        assert "/remember" in text
        assert "/snippet" not in text
        assert "/search" in text
        assert "/search_debug" in text
        assert "/profile" in text


class TestRememberHandler:
    """Test /remember command."""

    @pytest.mark.asyncio
    async def test_captures_thought(self, mock_update, mock_context):
        mock_context.args = ["I", "prefer", "dark", "mode"]
        with patch("telegram_bot.asyncio.to_thread", new=AsyncMock(return_value="Thought captured and queued for indexing.")) as mock_to_thread:
            await remember_handler(mock_update, mock_context)
        mock_to_thread.assert_awaited_once_with(remember_handler.__globals__["capture_thought"], "I prefer dark mode")
        text = mock_update.message.reply_text.call_args[0][0]
        assert "✅" in text

    @pytest.mark.asyncio
    async def test_empty_thought_warns(self, mock_update, mock_context):
        mock_context.args = []
        await remember_handler(mock_update, mock_context)
        text = mock_update.message.reply_text.call_args[0][0]
        assert "⚠️" in text

    @pytest.mark.asyncio
    async def test_calls_single_brain_capture_api(self, mock_update, mock_context):
        mock_context.args = ["test", "thought"]
        with patch("telegram_bot.asyncio.to_thread", new=AsyncMock(return_value="ok")) as mock_to_thread:
            await remember_handler(mock_update, mock_context)
            mock_to_thread.assert_awaited_once_with(remember_handler.__globals__["capture_thought"], "test thought")

    @pytest.mark.asyncio
    async def test_handles_error(self, mock_update, mock_context):
        mock_context.args = ["test"]
        with patch("telegram_bot.asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("DB down"))):
            await remember_handler(mock_update, mock_context)
        text = mock_update.message.reply_text.call_args[0][0]
        assert "❌" in text

    @pytest.mark.asyncio
    async def test_offloads_capture_to_thread(self, mock_update, mock_context):
        mock_context.args = ["slow", "capture"]
        with patch("telegram_bot.asyncio.to_thread", new=AsyncMock(return_value="queued")) as mock_to_thread:
            await remember_handler(mock_update, mock_context)
        mock_to_thread.assert_awaited_once_with(remember_handler.__globals__["capture_thought"], "slow capture")

    @pytest.mark.asyncio
    async def test_preserves_multiline_payload_from_original_message(self, mock_update, mock_context):
        mock_update.message.text = "/remember Manifesto line 1\n\nManifesto line 2"
        command_entity = MagicMock()
        command_entity.type = "bot_command"
        command_entity.offset = 0
        command_entity.length = len("/remember")
        mock_update.message.entities = [command_entity]
        mock_context.args = ["Manifesto", "line", "1", "Manifesto", "line", "2"]
        with patch("telegram_bot.asyncio.to_thread", new=AsyncMock(return_value="queued")) as mock_to_thread:
            await remember_handler(mock_update, mock_context)
        mock_to_thread.assert_awaited_once_with(
            remember_handler.__globals__["capture_thought"],
            "Manifesto line 1\n\nManifesto line 2",
        )


class TestSearchHandler:
    """Test /search command."""

    @pytest.mark.asyncio
    async def test_searches_brain(self, mock_update, mock_context):
        mock_context.args = ["dark", "mode"]
        with patch(
            "telegram_bot.asyncio.to_thread",
            new=AsyncMock(return_value="=== Retrieved Brain Context ===\n- dark mode\n"),
        ) as mock_to_thread:
            await search_handler(mock_update, mock_context)
        mock_to_thread.assert_awaited_once_with(
            search_handler.__globals__["search_brain"], "dark mode", debug=False
        )
        text = mock_update.message.reply_text.call_args[0][0]
        assert "Brain Context" in text

    @pytest.mark.asyncio
    async def test_empty_query_warns(self, mock_update, mock_context):
        mock_context.args = []
        await search_handler(mock_update, mock_context)
        text = mock_update.message.reply_text.call_args[0][0]
        assert "⚠️" in text

    @pytest.mark.asyncio
    async def test_calls_single_brain_search_api(self, mock_update, mock_context):
        mock_context.args = ["query"]
        with patch("telegram_bot.asyncio.to_thread", new=AsyncMock(return_value="results")) as mock_to_thread:
            await search_handler(mock_update, mock_context)
            mock_to_thread.assert_awaited_once_with(
                search_handler.__globals__["search_brain"], "query", debug=False
            )

    @pytest.mark.asyncio
    async def test_handles_error(self, mock_update, mock_context):
        mock_context.args = ["test"]
        with patch("telegram_bot.asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("DB down"))):
            await search_handler(mock_update, mock_context)
        text = mock_update.message.reply_text.call_args[0][0]
        assert "❌" in text


class TestProfileHandler:
    """Test /profile command."""

    @pytest.mark.asyncio
    async def test_reads_active_managed_memories(self, mock_update, mock_context):
        mock_context.args = ["counterpoints"]
        with patch(
            "telegram_bot.asyncio.to_thread",
            new=AsyncMock(return_value="Active directives:\n- Act as an intellectual sparring partner"),
        ) as mock_to_thread:
            await profile_handler(mock_update, mock_context)
        mock_to_thread.assert_awaited_once_with(
            profile_handler.__globals__["get_active_memories"],
            query="counterpoints",
        )
        text = mock_update.message.reply_text.call_args[0][0]
        assert "Active directives" in text


class TestAutoCapture:
    """Test plain text auto-capture behavior."""

    @pytest.mark.asyncio
    async def test_no_capture_when_disabled(self, mock_update, mock_context):
        with patch("telegram_bot.AUTO_CAPTURE", False):
            await text_handler(mock_update, mock_context)
        mock_update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_captures_when_enabled(self, mock_update, mock_context):
        with patch("telegram_bot.AUTO_CAPTURE", True), \
             patch("telegram_bot.asyncio.to_thread", new=AsyncMock(return_value="ok")) as mock_to_thread:
            await text_handler(mock_update, mock_context)
        mock_to_thread.assert_awaited_once_with(text_handler.__globals__["capture_thought"], "Hello world")
        mock_update.message.reply_text.assert_called_once()
        text = mock_update.message.reply_text.call_args[0][0]
        assert "💭" in text


class TestSearchDebugHandler:
    """Test /search_debug command."""

    @pytest.mark.asyncio
    async def test_searches_brain_in_debug_mode(self, mock_update, mock_context):
        mock_context.args = ["dark", "mode"]
        with patch("telegram_bot.asyncio.to_thread", new=AsyncMock(return_value="=== Search Debug ===")) as mock_to_thread:
            await search_debug_handler(mock_update, mock_context)
        mock_to_thread.assert_awaited_once_with(
            search_debug_handler.__globals__["search_brain"], "dark mode", debug=True
        )
        text = mock_update.message.reply_text.call_args[0][0]
        assert "Search Debug" in text

    @pytest.mark.asyncio
    async def test_empty_query_warns(self, mock_update, mock_context):
        mock_context.args = []
        await search_debug_handler(mock_update, mock_context)
        text = mock_update.message.reply_text.call_args[0][0]
        assert "⚠️" in text

    @pytest.mark.asyncio
    async def test_calls_single_brain_debug_search_api(self, mock_update, mock_context):
        mock_context.args = ["query"]
        with patch("telegram_bot.asyncio.to_thread", new=AsyncMock(return_value="debug")) as mock_to_thread:
            await search_debug_handler(mock_update, mock_context)
            mock_to_thread.assert_awaited_once_with(
                search_debug_handler.__globals__["search_brain"], "query", debug=True
            )
