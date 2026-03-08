import os
import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from brain_core import (
    capture_thought,
    ingest,
    search_brain,
    start_background_workers,
)

# ── Bootstrap ─────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("open_brain")

# ── MCP Server ────────────────────────────────
mcp = FastMCP("OpenBrain", host="0.0.0.0")


@mcp.tool(name="capture_thought")
def mcp_capture_thought(thought: str) -> str:
    """
    Saves a new memory into the local OpenBrain instance.
    OpenBrain automatically decides whether to enrich it or store it raw-only.

    Args:
        thought: The memory content to save (non-empty).
    """
    return capture_thought(thought)


@mcp.tool(name="ingest")
def mcp_ingest(
    content: str,
    source: str = "import",
    external_id: str | None = None,
) -> str:
    """
    Flexible ingest for any channel: chat exports (ChatGPT, Claude, Antigravity),
    PC/phone notes, or other sources. Content is queued into the same memory pipeline;
    search_brain searches across all sources. Use external_id to avoid re-ingesting
    the same conversation or note (dedup).

    Args:
        content: Text to remember (one message, note, or merged chunk).
        source: Origin label (e.g. chatgpt, claude, notes_pc).
        external_id: Optional stable id for dedup (e.g. conversation or note id).
    """
    return ingest(
        content=content,
        source=source,
        external_id=external_id,
    )


@mcp.tool(name="search_brain")
def mcp_search_brain(query: str) -> str:
    """
    Searches stored memories via vector similarity and knowledge graph.
    Call this BEFORE answering questions about the user's projects or history.

    Args:
        query: Natural language search query.
    """
    return search_brain(query, debug=False)


@mcp.tool(name="search_brain_debug")
def mcp_search_brain_debug(query: str) -> str:
    """
    Searches the user's past memories and returns scoring/debug trace details.
    Use for retrieval quality troubleshooting.

    Args:
        query: Natural language search query.
    """
    return search_brain(query, debug=True)


if __name__ == "__main__":
    # Allow switching transport via environment variable for local testing/IDE integration
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    logger.info("Starting OpenBrain MCP server with transport=%s", transport)
    start_background_workers()
    mcp.run(transport=transport)
