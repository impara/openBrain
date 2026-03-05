import os
import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mem0 import Memory
from mem0.configs.base import MemoryConfig, VectorStoreConfig, LlmConfig
from age_provider import ApacheAGEProvider

# ── Bootstrap ─────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("open_brain")

# ── MCP Server ────────────────────────────────
mcp = FastMCP("OpenBrain", host="0.0.0.0")

# ── Database Config ───────────────────────────
DB_CONFIG = {
    "dbname": os.environ.get("POSTGRES_DB", "open_brain"),
    "user": os.environ.get("POSTGRES_USER", "brain_user"),
    "password": os.environ["POSTGRES_PASSWORD"],
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
}

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required.")

# ── Mem0 + AGE Dual-Stack ─────────────────────
age_graph = ApacheAGEProvider(**DB_CONFIG)

config_dict = {
    "vector_store": {
        "provider": "pgvector",
        "config": {
            **DB_CONFIG,
            "collection_name": "memories",
        },
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini"},
    },
}

m = Memory.from_config(config_dict)
m.graph = age_graph
m.enable_graph = True

logger.info("OpenBrain initialized — Mem0 + AGE dual-stack ready")


@mcp.tool()
def capture_thought(thought: str, user_id: str = "default") -> str:
    """
    Saves a new memory into the user's Open Brain.
    Call this when the user makes a decision, meets someone, or specifies a constraint.
    """
    logger.info("Capturing thought for user=%s (len=%d)", user_id, len(thought))
    result = m.add(messages=[{"role": "user", "content": thought}], user_id=user_id)
    logger.debug("Mem0 add result: %s", result)
    return "Successfully captured thought into memory."


@mcp.tool()
def search_brain(query: str, user_id: str = "default") -> str:
    """
    Searches the user's past memories via vector similarity and knowledge graph.
    Call this BEFORE answering questions about the user's projects or history.
    """
    logger.info("Searching brain for user=%s query=%r", user_id, query[:80])
    memories = m.search(query=query, user_id=user_id)
    
    # Mem0 1.0.5 with graph enabled returns a dict {"results": [...], "relations": [...]}
    results = memories
    if isinstance(memories, dict):
        results = memories.get("results", [])
        # We could also include relations here if we wanted
    
    if not results:
        return "No relevant memories found in the Open Brain."

    context = "=== Retrieved Brain Context ===\n"
    for memory in results:
        if isinstance(memory, dict):
            text = memory.get("text", memory.get("content", str(memory)))
        else:
            text = str(memory)
        context += f"- {text}\n"

    logger.debug("Returning %d context items", len(results))
    return context


if __name__ == "__main__":
    # Allow switching transport via environment variable for local testing/IDE integration
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    logger.info("Starting OpenBrain MCP server with transport=%s", transport)
    mcp.run(transport=transport)