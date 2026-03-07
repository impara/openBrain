"""
telegram_bot.py — Telegram interface for OpenBrain.

Allows users to capture thoughts and search their brain from Telegram.
Uses long-polling (no public URL needed — works behind NAT / homelab).
"""

import asyncio
import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from brain_core import capture_thought, search_brain, start_background_workers

# ── Bootstrap ─────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("open_brain.telegram")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
AUTO_CAPTURE = os.environ.get("TELEGRAM_AUTO_CAPTURE", "false").lower() == "true"


def _user_id(update: Update) -> str:
    """Map Telegram user to OpenBrain user_id.
    
    Currently single-user mode — shares the same brain as the IDE.
    For multi-user isolation, change to: return f"telegram_{update.effective_user.id}"
    """
    return "default"


# ── Command Handlers ──────────────────────────

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — welcome message."""
    await update.message.reply_text(
        "🧠 *OpenBrain — Your AI Memory*\n\n"
        "I store and recall your thoughts using semantic search "
        "and a knowledge graph.\n\n"
        "*Commands:*\n"
        "• `/remember <text>` — Save a thought\n"
        "• `/search <query>` — Search your memories\n"
        "• `/search_debug <query>` — Search with debug trace\n"
        "• `/help` — Show this message\n\n"
        "Everything is tied to your Telegram account, "
        "so your memories are private to you.",
        parse_mode="Markdown",
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — usage instructions."""
    await update.message.reply_text(
        "🧠 *OpenBrain Commands*\n\n"
        "• `/remember <text>` — Save a thought or decision\n"
        "• `/search <query>` — Recall related memories\n"
        "• `/search_debug <query>` — Recall with scoring/debug trace\n"
        "• `/help` — Show this message",
        parse_mode="Markdown",
    )


async def remember_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /remember <text> — capture a thought."""
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text(
            "⚠️ Please provide a thought to remember.\n"
            "Example: `/remember I prefer dark mode in all my projects`",
            parse_mode="Markdown",
        )
        return

    uid = _user_id(update)
    logger.info("Telegram /remember from %s (len=%d)", uid, len(text))

    try:
        result = await asyncio.to_thread(capture_thought, text, user_id=uid)
        await update.message.reply_text(f"✅ {result}")
    except Exception as e:
        logger.exception("Failed to capture thought")
        await update.message.reply_text(f"❌ Error: {e}")


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search <query> — search the brain."""
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text(
            "⚠️ Please provide a search query.\n"
            "Example: `/search dark mode preferences`",
            parse_mode="Markdown",
        )
        return

    uid = _user_id(update)
    logger.info("Telegram /search from %s query=%r", uid, query[:80])

    try:
        result = search_brain(query, user_id=uid, debug=False)
        await update.message.reply_text(result)
    except Exception as e:
        logger.exception("Failed to search brain")
        await update.message.reply_text(f"❌ Error: {e}")


async def search_debug_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search_debug <query> — search with full trace."""
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text(
            "⚠️ Please provide a search query.\n"
            "Example: `/search_debug dark mode preferences`",
            parse_mode="Markdown",
        )
        return

    uid = _user_id(update)
    logger.info("Telegram /search_debug from %s query=%r", uid, query[:80])

    try:
        result = search_brain(query, user_id=uid, debug=True)
        await update.message.reply_text(result)
    except Exception as e:
        logger.exception("Failed to debug-search brain")
        await update.message.reply_text(f"❌ Error: {e}")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text — auto-capture if enabled."""
    if not AUTO_CAPTURE:
        return

    text = update.message.text
    if not text or text.startswith("/"):
        return

    uid = _user_id(update)
    logger.info("Telegram auto-capture from %s (len=%d)", uid, len(text))

    try:
        await asyncio.to_thread(capture_thought, text, user_id=uid)
        await update.message.reply_text("💭 Thought captured.")
    except Exception as e:
        logger.exception("Failed to auto-capture thought")


# ── Main ──────────────────────────────────────

def main() -> None:
    """Start the Telegram bot with long-polling."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN environment variable is required. "
            "Get one from @BotFather on Telegram."
        )

    logger.info(
        "Starting OpenBrain Telegram bot (auto_capture=%s)", AUTO_CAPTURE
    )
    start_background_workers()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("remember", remember_handler))
    app.add_handler(CommandHandler("search", search_handler))
    app.add_handler(CommandHandler("search_debug", search_debug_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
