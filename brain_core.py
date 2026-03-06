"""
brain_core.py — Shared memory layer for OpenBrain.

Initializes the Mem0 + Apache AGE dual-stack and exposes
capture_thought() / search_brain() for any client (MCP, Telegram, etc.).
"""

import os
import logging
from dotenv import load_dotenv
from mem0 import Memory
from age_provider import ApacheAGEProvider

load_dotenv()

logger = logging.getLogger("open_brain")

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
# Note: Mem0 1.0.5 does not natively support injecting a live 'instance' of a custom provider
# via config_dict cleanly. We monkey-patch m.graph here. If Mem0 updates to support
# {"graph_store": {"provider": "custom", "instance": age_graph}}, this should be migrated.
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


def _extract_added_count(result) -> int:
    """Best-effort count of memories persisted by Mem0 add()."""
    if result is None:
        return 0
    if isinstance(result, list):
        return len(result)
    if isinstance(result, dict):
        for key in ("results", "memories", "data", "added"):
            value = result.get(key)
            if isinstance(value, list):
                return len(value)
        return 1 if result else 0
    return 1


def _format_relation(relation) -> str:
    """Normalize graph relation payloads into readable text."""
    if isinstance(relation, str):
        return relation
    if not isinstance(relation, dict):
        return str(relation)

    source = relation.get("source") or relation.get("from") or relation.get("subject")
    target = relation.get("target") or relation.get("to") or relation.get("object")
    rel = relation.get("relationship") or relation.get("relation") or relation.get("type")

    if isinstance(source, dict):
        source = source.get("name") or source.get("id") or str(source)
    if isinstance(target, dict):
        target = target.get("name") or target.get("id") or str(target)

    if source and rel and target:
        return f"{source} -[{rel}]-> {target}"

    return relation.get("memory") or relation.get("text") or relation.get("content") or str(relation)


def capture_thought(thought: str, user_id: str = "default") -> str:
    """
    Saves a new memory into the user's Open Brain.
    Call this when the user makes a decision, meets someone, or specifies a constraint.
    """
    logger.info("Capturing thought for user=%s (len=%d)", user_id, len(thought))
    try:
        result = m.add(messages=[{"role": "user", "content": thought}], user_id=user_id)
        logger.debug("Mem0 add result: %s", result)
        added_count = _extract_added_count(result)
        if added_count == 0:
            logger.warning("No searchable memories were extracted for user=%s", user_id)
            return (
                "Input was captured, but no searchable memory was extracted. "
                "Try sending shorter factual chunks."
            )
        return "Successfully captured thought into memory."
    except Exception as e:
        logger.error("Memory capture failed (potential ghost memory): %s", e)
        return f"Warning: Failed to fully commit memory. Error: {e}"


def search_brain(query: str, user_id: str = "default") -> str:
    """
    Searches the user's past memories via vector similarity and knowledge graph.
    Call this BEFORE answering questions about the user's projects or history.
    """
    logger.info("Searching brain for user=%s query=%r", user_id, query[:80])
    memories = m.search(query=query, user_id=user_id, limit=10)

    # Mem0 1.0.5 with graph enabled returns a dict {"results": [...], "relations": [...]}
    results = memories
    relations = []
    if isinstance(memories, dict):
        results = memories.get("results", [])
        relations = memories.get("relations", [])

    # Filter by relevance — pgvector cosine distance: lower = more similar
    SCORE_THRESHOLD = 0.75
    relevant = [
        r for r in results
        if not isinstance(r, dict) or r.get("score", 0) <= SCORE_THRESHOLD
    ]

    if not relevant and results:
        # Fallback: avoid hiding all memories when provider score semantics differ.
        relevant = results[:3]

    context_parts = []
    for memory in relevant:
        if isinstance(memory, dict):
            text = memory.get("memory", memory.get("text", memory.get("content", str(memory))))
        else:
            text = str(memory)
        if text:
            context_parts.append(f"• {text}")

    for relation in relations:
        relation_text = _format_relation(relation)
        if relation_text:
            context_parts.append(f"• {relation_text}")

    if not context_parts:
        return "No relevant memories found in the Open Brain."

    logger.debug(
        "Returning %d memory items and %d relation items",
        len(relevant),
        len(relations),
    )
    return "\n\n".join(context_parts).strip()
