import os
import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from brain_core import capture_thought, search_brain

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
def mcp_capture_thought(thought: str, user_id: str = "default") -> str:
    """
    Saves a new memory into the user's Open Brain.
    Call this when the user makes a decision, meets someone, or specifies a constraint.
    """
    return capture_thought(thought, user_id)


@mcp.tool(name="search_brain")
def mcp_search_brain(query: str, user_id: str = "default") -> str:
    """
    Searches the user's past memories via vector similarity and knowledge graph.
    Call this BEFORE answering questions about the user's projects or history.
    """
    return search_brain(query, user_id)


if __name__ == "__main__":
    # Allow switching transport via environment variable for local testing/IDE integration
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    logger.info("Starting OpenBrain MCP server with transport=%s", transport)
    mcp.run(transport=transport)