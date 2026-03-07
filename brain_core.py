"""
brain_core.py — Shared memory layer for OpenBrain.

Initializes the Mem0 + Apache AGE dual-stack and exposes
capture_thought() / search_brain() for any client (MCP, Telegram, etc.).
"""

import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI
from psycopg2 import pool

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


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


_LLM_MODEL = os.environ.get("OPENBRAIN_SUMMARY_MODEL", "gpt-4o-mini")
_CAPTURE_MODE = os.environ.get("OPENBRAIN_CAPTURE_MODE", "async").strip().lower()
if _CAPTURE_MODE not in {"async", "sync"}:
    _CAPTURE_MODE = "async"
_INGEST_WORKERS = _env_int("OPENBRAIN_INGEST_WORKERS", 1)
_INGEST_POLL_MS = _env_int("OPENBRAIN_INGEST_POLL_MS", 250, minimum=50)
_CAPTURE_RETRY_BACKOFF_SECONDS = (60, 300, 900)
_CAPTURE_ACK_TEXT = "Thought captured and queued for indexing."
_MAX_EVIDENCE = 5
_SCORE_THRESHOLD = 0.52
_SOURCE_WEIGHT = {"vector": 1.0, "graph": 0.82, "raw": 0.68}
_MAX_DEBUG_CHARS = 3800
_MAX_SOURCE_TEXT = 180
_MAX_ANSWER_TEXT = 520
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9@._-]{1,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "your",
    "you",
    "does",
    "which",
    "what",
    "when",
    "where",
    "why",
    "how",
    "into",
    "about",
    "have",
    "has",
    "had",
    "was",
    "were",
    "are",
    "is",
    "to",
    "of",
    "in",
    "on",
}


@dataclass
class EvidenceItem:
    source: str
    text: str
    score_raw: float | None = None
    score_norm: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureJob:
    id: int
    raw_capture_id: int
    user_id: str
    thought: str
    attempt_count: int = 0


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
        "config": {"model": _LLM_MODEL},
    },
}

m = Memory.from_config(config_dict)
m.graph = age_graph
m.enable_graph = True
_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
_db_pool = pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=max(4, _INGEST_WORKERS + 4),
    **DB_CONFIG,
)
_worker_lock = threading.Lock()
_worker_threads: list[threading.Thread] = []
_worker_stop_event = threading.Event()

logger.info(
    "OpenBrain initialized — Mem0 + AGE dual-stack ready (capture_mode=%s, ingest_workers=%d)",
    _CAPTURE_MODE,
    _INGEST_WORKERS,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def _db_cursor(*, commit: bool = False):
    """Yield a pooled DB cursor and return the connection to the pool cleanly."""
    conn = _db_pool.getconn()
    try:
        with conn.cursor() as cur:
            yield conn, cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            if not conn.closed:
                conn.rollback()
        except Exception:
            pass
        _db_pool.putconn(conn)


def _ensure_storage_infrastructure() -> None:
    """Create capture tables and indexes for fresh and upgraded deployments."""
    with _db_cursor(commit=True) as (_, cur):
        cur.execute("CREATE SCHEMA IF NOT EXISTS memory_store;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_store.raw_captures (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'capture_thought',
                content TEXT NOT NULL,
                content_len INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_raw_captures_user_created
            ON memory_store.raw_captures (user_id, created_at DESC);
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_store.capture_jobs (
                id BIGSERIAL PRIMARY KEY,
                raw_capture_id BIGINT NOT NULL UNIQUE REFERENCES memory_store.raw_captures(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                available_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ,
                CONSTRAINT capture_jobs_status_check
                    CHECK (status IN ('pending', 'processing', 'done', 'retry', 'failed'))
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_capture_jobs_status_available_created
            ON memory_store.capture_jobs (status, available_at, created_at);
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_capture_jobs_raw_capture_id
            ON memory_store.capture_jobs (raw_capture_id);
            """
        )


def _enqueue_capture_job(raw_capture_id: int, user_id: str, cur: Any | None = None) -> int | None:
    """Create or reuse one ingest job per raw capture."""
    if raw_capture_id is None:
        return None

    def _run(cursor: Any) -> int | None:
        cursor.execute(
            """
            INSERT INTO memory_store.capture_jobs (raw_capture_id, user_id)
            VALUES (%s, %s)
            ON CONFLICT (raw_capture_id)
            DO UPDATE SET user_id = EXCLUDED.user_id
            RETURNING id;
            """,
            (raw_capture_id, user_id),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    if cur is not None:
        return _run(cur)

    with _db_cursor(commit=True) as (_, cursor):
        return _run(cursor)


def _capture_raw_and_enqueue(
    thought: str,
    user_id: str,
    source: str = "capture_thought",
) -> tuple[int | None, int | None]:
    """Persist raw text and enqueue ingestion atomically."""
    if not thought:
        return None, None

    with _db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            INSERT INTO memory_store.raw_captures (user_id, source, content, content_len)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (user_id, source, thought, len(thought)),
        )
        row = cur.fetchone()
        raw_capture_id = row[0] if row else None
        job_id = _enqueue_capture_job(raw_capture_id, user_id=user_id, cur=cur)
        return raw_capture_id, job_id


def _tokenize(text: str) -> list[str]:
    return sorted(
        {
            token
            for token in _TOKEN_RE.findall((text or "").lower())
            if len(token) > 1 and token not in _STOPWORDS
        }
    )


def _normalized_text(text: str) -> str:
    return " ".join(_tokenize(text))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = f"{raw[:-1]}+00:00"
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _recency_bonus(created_at: Any) -> float:
    dt = _parse_dt(created_at)
    if not dt:
        return 0.0
    age_hours = (_utcnow() - dt.astimezone(timezone.utc)).total_seconds() / 3600
    if age_hours <= 0:
        return 0.08
    if age_hours >= 72:
        return 0.0
    return max(0.0, min(0.08, (72 - age_hours) / 72 * 0.08))


def _query_overlap(query_tokens: list[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return 0.0
    matched = 0
    for q in set(query_tokens):
        for t in text_tokens:
            if q == t:
                matched += 1
                break
            if len(q) >= 4 and (q in t or t in q):
                matched += 1
                break
    overlap = matched / len(set(query_tokens))
    return max(0.0, min(1.0, overlap))


def _vector_similarity(distance: Any) -> float:
    if distance is None:
        return 0.55
    d = _safe_float(distance, default=1.0)
    # Mem0 pgvector score is cosine distance (lower is better, usually [0, 2]).
    similarity = 1.0 - min(max(d, 0.0), 1.0)
    return max(0.0, min(1.0, similarity))


def _extract_relevant_snippet(content: str, query_tokens: list[str], max_len: int = 260) -> str:
    clean = " ".join((content or "").split())
    if len(clean) <= max_len:
        return clean
    if not query_tokens:
        return f"{clean[:max_len].rstrip()}..."

    lowered = clean.lower()
    for token in query_tokens:
        idx = lowered.find(token)
        if idx >= 0:
            start = max(0, idx - max_len // 3)
            end = min(len(clean), start + max_len)
            snippet = clean[start:end].strip()
            if start > 0:
                snippet = f"...{snippet}"
            if end < len(clean):
                snippet = f"{snippet}..."
            return snippet
    return f"{clean[:max_len].rstrip()}..."


def _search_raw_captures(query: str, user_id: str, limit: int = 8) -> list[dict]:
    """Find recent raw captures matching query terms."""
    terms = _tokenize(query)[:8]

    with _db_cursor() as (_, cur):
        if terms:
            where_parts = " OR ".join(["LOWER(content) LIKE %s"] * len(terms))
            params = [user_id, *[f"%{t}%" for t in terms], limit]
            cur.execute(
                f"""
                SELECT id, content, created_at, content_len
                FROM memory_store.raw_captures
                WHERE user_id = %s AND ({where_parts})
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                params,
            )
        else:
            cur.execute(
                """
                SELECT id, content, created_at, content_len
                FROM memory_store.raw_captures
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (user_id, limit),
            )

        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "content_len": row[3],
            }
            for row in rows
        ]


def _extract_added_count(result: Any) -> int:
    """Best-effort count of memories persisted by Mem0 add()."""
    if result is None:
        return 0
    if isinstance(result, list):
        return len(result)
    if isinstance(result, Mapping) or hasattr(result, "get"):
        for key in ("results", "memories", "data", "added", "nodes", "edges"):
            value = result.get(key)
            if isinstance(value, list):
                return len(value)
        return 1 if result else 0
    return 1


def _format_relation(relation: Any) -> str:
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


def _relation_lines(relations: Any) -> list[str]:
    """Normalize Mem0 relation payloads into readable lines."""
    if isinstance(relations, Mapping) or hasattr(relations, "get"):
        nodes = relations.get("nodes", [])
        edges = relations.get("edges", [])

        if isinstance(nodes, list) and isinstance(edges, list):
            node_names = {}
            for node in nodes:
                if isinstance(node, dict):
                    node_id = node.get("id")
                    props = node.get("properties") or {}
                    name = props.get("name") or node.get("name") or node.get("label") or str(node_id)
                    if node_id is not None:
                        node_names[node_id] = str(name)

            lines = []
            for edge in edges:
                if not isinstance(edge, dict):
                    lines.append(_format_relation(edge))
                    continue
                src = node_names.get(edge.get("start_id"), edge.get("start_id"))
                dst = node_names.get(edge.get("end_id"), edge.get("end_id"))
                rel = edge.get("label") or edge.get("relationship") or "RELATED_TO"
                lines.append(f"{src} -[{rel}]-> {dst}")

            return [line for line in lines if line]

    if isinstance(relations, list):
        return [line for line in (_format_relation(r) for r in relations) if line]

    single = _format_relation(relations)
    return [single] if single else []


def _extract_vector_and_relations(payload: Any) -> tuple[list[Any], Any]:
    results: list[Any] = []
    relations: Any = []

    if isinstance(payload, list):
        results = payload
        return results, relations

    if isinstance(payload, Mapping) or hasattr(payload, "get"):
        results = payload.get("results", []) or []
        relations = payload.get("relations", []) or []
        if not results:
            nodes_value = payload.get("nodes", None)
            if isinstance(nodes_value, list):
                results = nodes_value
        if not relations:
            edges_value = payload.get("edges", None)
            if isinstance(edges_value, list):
                relations = edges_value
        return results, relations

    return results, relations


def _build_candidates(
    query: str,
    vector_results: list[Any],
    relations: Any,
    raw_matches: list[dict],
) -> list[EvidenceItem]:
    query_tokens = _tokenize(query)
    candidates: list[EvidenceItem] = []

    for row in vector_results:
        if isinstance(row, dict):
            text = row.get("memory") or row.get("text") or row.get("content") or str(row)
            created_at = row.get("created_at")
            score_raw = _safe_float(row.get("score"), default=1.0)
            metadata = {
                "id": row.get("id"),
                "created_at": created_at,
                "vector_distance": score_raw,
            }
        else:
            text = str(row)
            score_raw = 1.0
            metadata = {}
        if text:
            candidates.append(
                EvidenceItem(
                    source="vector",
                    text=str(text).strip(),
                    score_raw=score_raw,
                    metadata=metadata,
                )
            )

    for relation_text in _relation_lines(relations):
        candidates.append(
            EvidenceItem(
                source="graph",
                text=relation_text.strip(),
                score_raw=None,
                metadata={},
            )
        )

    for raw in raw_matches:
        snippet = _extract_relevant_snippet(raw.get("content", ""), query_tokens=query_tokens)
        if snippet:
            candidates.append(
                EvidenceItem(
                    source="raw",
                    text=snippet,
                    score_raw=None,
                    metadata={
                        "raw_id": raw.get("id"),
                        "created_at": raw.get("created_at"),
                        "content_len": raw.get("content_len"),
                    },
                )
            )
    return candidates


def _score_candidate(item: EvidenceItem, query_tokens: list[str]) -> dict[str, float]:
    overlap = _query_overlap(query_tokens, item.text)
    source_weight = _SOURCE_WEIGHT.get(item.source, 0.5)
    recency = _recency_bonus(item.metadata.get("created_at"))

    vector_sim = 0.0
    if item.source == "vector":
        vector_sim = _vector_similarity(item.score_raw)
        score = (0.55 * vector_sim) + (0.30 * overlap) + (0.15 * source_weight) + recency
    else:
        score = (0.65 * overlap) + (0.35 * source_weight) + recency

    score = max(0.0, min(1.0, score))
    return {
        "overlap": overlap,
        "source_weight": source_weight,
        "vector_similarity": vector_sim,
        "recency_bonus": recency,
        "final": score,
    }


def _rank_evidence(
    query: str,
    candidates: list[EvidenceItem],
    limit: int = _MAX_EVIDENCE,
    threshold: float = _SCORE_THRESHOLD,
) -> tuple[list[EvidenceItem], dict[str, Any]]:
    query_tokens = _tokenize(query)
    selected: list[EvidenceItem] = []
    dropped: list[dict[str, Any]] = []
    debug_candidates: list[dict[str, Any]] = []
    seen_norm: set[str] = set()

    for item in candidates:
        text = item.text.strip()
        if not text:
            dropped.append({"source": item.source, "text": "", "reason": "empty"})
            continue

        norm = _normalized_text(text)
        components = _score_candidate(item, query_tokens=query_tokens)
        debug_candidates.append(
            {
                "source": item.source,
                "text": text,
                "score": round(components["final"], 4),
                "overlap": round(components["overlap"], 4),
                "vector_similarity": round(components["vector_similarity"], 4),
                "recency_bonus": round(components["recency_bonus"], 4),
            }
        )

        # Precision-first: reject near-zero overlap unless vector similarity is very high.
        if components["overlap"] < 0.05 and components["vector_similarity"] < 0.72:
            dropped.append(
                {
                    "source": item.source,
                    "text": text,
                    "reason": "low_overlap",
                    "score": round(components["final"], 4),
                }
            )
            continue

        if norm in seen_norm:
            dropped.append(
                {
                    "source": item.source,
                    "text": text,
                    "reason": "duplicate",
                    "score": round(components["final"], 4),
                }
            )
            continue

        item.score_norm = components["final"]
        if item.score_norm < threshold:
            dropped.append(
                {
                    "source": item.source,
                    "text": text,
                    "reason": "low_confidence",
                    "score": round(item.score_norm, 4),
                }
            )
            continue

        seen_norm.add(norm)
        selected.append(item)

    selected.sort(key=lambda x: x.score_norm, reverse=True)
    selected = selected[: max(1, min(limit, _MAX_EVIDENCE))]

    debug = {
        "query": query,
        "query_tokens": query_tokens,
        "candidates": debug_candidates,
        "selected": [
            {
                "source": item.source,
                "text": item.text,
                "score": round(item.score_norm, 4),
                "metadata": item.metadata,
            }
            for item in selected
        ],
        "dropped": dropped,
        "threshold": threshold,
    }
    return selected, debug


def _relation_to_sentence(text: str) -> str:
    match = re.match(r"^\s*(.+?)\s*-\[([^\]]+)\]->\s*(.+?)\s*$", text)
    if not match:
        return text.strip()
    source, relation, target = match.groups()
    rel_text = relation.replace("_", " ").strip().lower()
    return f"{source.strip()} is {rel_text} {target.strip()}"


def _truncate_text(text: str, max_len: int = _MAX_SOURCE_TEXT) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_len:
        return clean
    return f"{clean[:max_len].rstrip()}..."


def _synthesize_answer(query: str, evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return (
            "I am not confident enough to answer from current memory. "
            "Try a more specific query or remember more concrete facts."
        )

    statements: list[str] = []
    source_counts = {"vector": 0, "graph": 0, "raw": 0}
    source_limits = {"vector": 2, "graph": 2, "raw": 1}
    seen_norm: set[str] = set()

    for item in evidence:
        if source_counts.get(item.source, 0) >= source_limits.get(item.source, 1):
            continue
        if item.source == "raw" and source_counts.get("vector", 0) > 0:
            continue

        text = _relation_to_sentence(item.text)
        if item.source == "raw":
            text = re.split(r"(?<=[.!?])\s+", text.strip())[0]
        text = _truncate_text(text, max_len=140)
        if not text:
            continue
        norm = _normalized_text(text)
        if norm and norm not in seen_norm:
            redundant = False
            norm_tokens = set(norm.split())
            for existing in seen_norm:
                existing_tokens = set(existing.split())
                if norm_tokens and (norm_tokens.issubset(existing_tokens) or existing_tokens.issubset(norm_tokens)):
                    redundant = True
                    break
            if redundant:
                continue
            statements.append(text)
            seen_norm.add(norm)
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
        if len(statements) >= 3:
            break

    if not statements:
        return (
            "I found some related memory, but not enough clear facts to answer confidently."
        )

    joined = "; ".join(statements)
    confidence = evidence[0].score_norm
    if confidence < 0.70:
        answer = f"I may be missing context, but your memory suggests: {joined}."
    else:
        answer = f"Based on your stored memories: {joined}."

    return _truncate_text(answer, max_len=_MAX_ANSWER_TEXT)


def _format_sources(evidence: list[EvidenceItem]) -> list[str]:
    lines: list[str] = []
    source_counts = {"vector": 0, "graph": 0, "raw": 0}
    source_limits = {"vector": 2, "graph": 2, "raw": 1}
    for item in evidence:
        if source_counts.get(item.source, 0) >= source_limits.get(item.source, 1):
            continue
        text = _truncate_text(item.text, max_len=_MAX_SOURCE_TEXT)
        lines.append(f"• [{item.source}] {text}")
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
        if len(lines) >= _MAX_EVIDENCE:
            break
    return lines


def _format_search_response(query: str, evidence: list[EvidenceItem]) -> str:
    answer = _synthesize_answer(query, evidence)
    if not evidence:
        return f"Answer: {answer}"
    source_lines = "\n".join(_format_sources(evidence))
    return f"Answer: {answer}\n\nSources:\n{source_lines}"


def _format_debug_report(query: str, ranking_debug: dict[str, Any], evidence: list[EvidenceItem]) -> str:
    lines = [
        "=== Search Debug ===",
        f"query: {query}",
        f"normalized_tokens: {', '.join(ranking_debug.get('query_tokens', [])) or '(none)'}",
        f"threshold: {ranking_debug.get('threshold')}",
        "",
        "candidates_pre_filter:",
    ]
    for idx, row in enumerate(ranking_debug.get("candidates", []), start=1):
        lines.append(
            (
                f"{idx}. [{row.get('source')}] score={row.get('score')} "
                f"overlap={row.get('overlap')} vec={row.get('vector_similarity')} "
                f"recency={row.get('recency_bonus')} text={_truncate_text(str(row.get('text', '')), 140)}"
            )
        )

    lines.append("")
    lines.append("final_ranking:")
    for idx, item in enumerate(evidence, start=1):
        lines.append(
            f"{idx}. [{item.source}] score={round(item.score_norm, 4)} text={_truncate_text(item.text, 140)}"
        )

    lines.append("")
    lines.append("dropped:")
    dropped = ranking_debug.get("dropped", [])
    if not dropped:
        lines.append("(none)")
    else:
        for idx, row in enumerate(dropped, start=1):
            lines.append(
                f"{idx}. [{row.get('source')}] {row.get('reason')} "
                f"score={row.get('score', '-')}: {_truncate_text(str(row.get('text', '')), 120)}"
            )

    lines.append("")
    lines.append("answer_preview:")
    lines.append(_synthesize_answer(query, evidence))
    report = "\n".join(lines)
    if len(report) > _MAX_DEBUG_CHARS:
        return f"{report[:_MAX_DEBUG_CHARS - 3]}..."
    return report


def _chunk_text_sentences(text: str, max_chunk_chars: int = 900, max_chunks: int = 6) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    if not sentences:
        fallback = [s.strip() for s in re.split(r"[\n;]+", text or "") if s.strip()]
        sentences = fallback if fallback else [text.strip()]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        proposed = sentence if not current else f"{current} {sentence}"
        if len(proposed) > max_chunk_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = proposed

        if len(chunks) >= max_chunks:
            break

    if current and len(chunks) < max_chunks:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk]


def _heuristic_bullets(chunk: str, max_bullets: int = 3) -> list[str]:
    sentences = [s.strip("•*- ").strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
    prioritized: list[str] = []
    keywords = ("channel", "frequency", "standard", "limit", "mode", "uses", "allowed", "reserved")
    for sentence in sentences:
        lower = sentence.lower()
        if len(sentence) < 25 or len(sentence) > 240:
            continue
        if any(k in lower for k in keywords) or re.search(r"\d", sentence):
            prioritized.append(sentence.rstrip("."))
        if len(prioritized) >= max_bullets:
            break

    if prioritized:
        return prioritized
    return [s.rstrip(".") for s in sentences[:max_bullets] if len(s) > 20]


def _extract_factual_bullets(chunk: str, max_bullets: int = 3) -> list[str]:
    try:
        response = _openai.chat.completions.create(
            model=_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract concise factual bullets from the text. "
                        f"Return JSON only with shape: {{\"bullets\": [\"...\"]}}. "
                        f"Do not output more than {max_bullets} bullets."
                    ),
                },
                {"role": "user", "content": chunk},
            ],
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        payload = json.loads(raw) if raw else {}
        bullets = payload.get("bullets", [])
        if not isinstance(bullets, list):
            return _heuristic_bullets(chunk, max_bullets=max_bullets)
        cleaned: list[str] = []
        for bullet in bullets:
            text = str(bullet).strip().lstrip("•*- ").strip()
            if 20 <= len(text) <= 240:
                cleaned.append(text.rstrip("."))
            if len(cleaned) >= max_bullets:
                break
        if cleaned:
            return cleaned
    except Exception as e:
        logger.warning("LLM bullet extraction failed, using heuristic fallback: %s", e)
    return _heuristic_bullets(chunk, max_bullets=max_bullets)


def _fallback_ingest_facts(thought: str, user_id: str, raw_capture_id: int | None) -> int:
    chunks = _chunk_text_sentences(thought, max_chunk_chars=900, max_chunks=6)
    if not chunks:
        return 0

    max_total_bullets = 10
    all_bullets: list[str] = []
    seen: set[str] = set()
    for chunk_idx, chunk in enumerate(chunks, start=1):
        bullets = _extract_factual_bullets(chunk, max_bullets=3)
        for bullet in bullets:
            norm = _normalized_text(bullet)
            if not norm or norm in seen:
                continue
            # Guardrails: skip low-information bullets.
            if len(norm.split()) < 5:
                continue
            seen.add(norm)
            all_bullets.append(bullet)
            if len(all_bullets) >= max_total_bullets:
                break
        if len(all_bullets) >= max_total_bullets:
            break

    added = 0
    for idx, bullet in enumerate(all_bullets, start=1):
        try:
            result = m.add(
                messages=[{"role": "user", "content": bullet}],
                user_id=user_id,
                metadata={
                    "ingest_mode": "fallback_fact",
                    "raw_capture_id": raw_capture_id,
                    "fact_index": idx,
                },
                infer=False,
            )
            added += _extract_added_count(result)
        except Exception as e:
            logger.warning("Fallback bullet ingestion failed for user=%s: %s", user_id, e)
            continue

    return added


def _ingest_capture(thought: str, user_id: str, raw_capture_id: int | None) -> tuple[int, int]:
    """Run the expensive memory ingestion flow for one captured thought."""
    result = m.add(
        messages=[{"role": "user", "content": thought}],
        user_id=user_id,
        metadata={"raw_capture_id": raw_capture_id, "ingest_mode": "primary"},
    )
    logger.debug("Mem0 add result for raw_capture_id=%s: %s", raw_capture_id, result)

    added_count = _extract_added_count(result)
    fallback_added = 0
    if added_count == 0:
        logger.info(
            "Mem0 add returned no explicit new items for user=%s raw_capture_id=%s, triggering fallback ingestion",
            user_id,
            raw_capture_id,
        )
        fallback_added = _fallback_ingest_facts(thought, user_id=user_id, raw_capture_id=raw_capture_id)
        if fallback_added > 0:
            logger.info(
                "Fallback ingestion added %d factual memories for user=%s raw_capture_id=%s",
                fallback_added,
                user_id,
                raw_capture_id,
            )
        else:
            logger.info(
                "Fallback ingestion produced no new vector memories for user=%s raw_capture_id=%s",
                user_id,
                raw_capture_id,
            )

    return added_count, fallback_added


def _claim_capture_job(job_id: int | None = None) -> CaptureJob | None:
    """Claim the next available capture job, or a specific job when provided."""
    filters = ["j.status IN ('pending', 'retry')"]
    params: list[Any] = []
    if job_id is None:
        filters.append("j.available_at <= CURRENT_TIMESTAMP")
    else:
        filters.append("j.id = %s")
        params.append(job_id)

    where_clause = " AND ".join(filters)
    with _db_cursor(commit=True) as (_, cur):
        cur.execute(
            f"""
            WITH next_job AS (
                SELECT j.id
                FROM memory_store.capture_jobs j
                WHERE {where_clause}
                ORDER BY j.available_at ASC, j.created_at ASC
                FOR UPDATE OF j SKIP LOCKED
                LIMIT 1
            )
            UPDATE memory_store.capture_jobs j
            SET status = 'processing',
                started_at = CURRENT_TIMESTAMP,
                finished_at = NULL,
                last_error = NULL,
                attempt_count = j.attempt_count + 1
            FROM next_job, memory_store.raw_captures r
            WHERE j.id = next_job.id AND r.id = j.raw_capture_id
            RETURNING j.id, j.raw_capture_id, j.user_id, r.content, j.attempt_count;
            """,
            params,
        )
        row = cur.fetchone()
        if not row:
            return None
        return CaptureJob(
            id=row[0],
            raw_capture_id=row[1],
            user_id=row[2],
            thought=row[3],
            attempt_count=row[4],
        )


def _mark_capture_job_done(job: CaptureJob, added_count: int, fallback_added: int, duration_ms: float) -> None:
    with _db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            UPDATE memory_store.capture_jobs
            SET status = 'done',
                finished_at = CURRENT_TIMESTAMP,
                last_error = NULL
            WHERE id = %s;
            """,
            (job.id,),
        )
    logger.info(
        "Capture job done job_id=%s raw_capture_id=%s attempt=%d added=%d fallback_added=%d ingest_ms=%.1f",
        job.id,
        job.raw_capture_id,
        job.attempt_count,
        added_count,
        fallback_added,
        duration_ms,
    )


def _mark_capture_job_retry(job: CaptureJob, error: Exception, duration_ms: float) -> None:
    message = str(error)
    if job.attempt_count <= len(_CAPTURE_RETRY_BACKOFF_SECONDS):
        delay_seconds = _CAPTURE_RETRY_BACKOFF_SECONDS[job.attempt_count - 1]
        status = "retry"
        sql = """
            UPDATE memory_store.capture_jobs
            SET status = %s,
                available_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                finished_at = NULL,
                last_error = %s
            WHERE id = %s;
        """
        params = (status, delay_seconds, message, job.id)
    else:
        delay_seconds = None
        status = "failed"
        sql = """
            UPDATE memory_store.capture_jobs
            SET status = %s,
                finished_at = CURRENT_TIMESTAMP,
                last_error = %s
            WHERE id = %s;
        """
        params = (status, message, job.id)

    with _db_cursor(commit=True) as (_, cur):
        cur.execute(sql, params)

    logger.warning(
        "Capture job %s job_id=%s raw_capture_id=%s attempt=%d retry_in=%s ingest_ms=%.1f error=%s",
        status,
        job.id,
        job.raw_capture_id,
        job.attempt_count,
        delay_seconds,
        duration_ms,
        message,
    )


def _process_capture_job(job: CaptureJob) -> bool:
    started_at = time.perf_counter()
    try:
        added_count, fallback_added = _ingest_capture(
            thought=job.thought,
            user_id=job.user_id,
            raw_capture_id=job.raw_capture_id,
        )
        duration_ms = (time.perf_counter() - started_at) * 1000
        _mark_capture_job_done(job, added_count=added_count, fallback_added=fallback_added, duration_ms=duration_ms)
        return True
    except Exception as e:
        duration_ms = (time.perf_counter() - started_at) * 1000
        _mark_capture_job_retry(job, error=e, duration_ms=duration_ms)
        return False


def process_next_capture_job() -> bool:
    """Claim and process the next available capture job."""
    job = _claim_capture_job()
    if not job:
        return False
    return _process_capture_job(job)


def _process_capture_job_by_id(job_id: int) -> bool:
    job = _claim_capture_job(job_id=job_id)
    if not job:
        return False
    return _process_capture_job(job)


def get_capture_job_stats() -> dict[str, int]:
    """Return current job counts by status for observability."""
    stats = {"pending": 0, "processing": 0, "done": 0, "retry": 0, "failed": 0}
    with _db_cursor() as (_, cur):
        cur.execute(
            """
            SELECT status, COUNT(*)
            FROM memory_store.capture_jobs
            GROUP BY status;
            """
        )
        for status, count in cur.fetchall():
            stats[str(status)] = int(count)
    stats["queued"] = stats["pending"] + stats["retry"]
    return stats


def _capture_ingest_worker(worker_index: int) -> None:
    logger.info(
        "Capture ingest worker started worker=%d poll_ms=%d capture_mode=%s",
        worker_index,
        _INGEST_POLL_MS,
        _CAPTURE_MODE,
    )
    poll_seconds = _INGEST_POLL_MS / 1000
    while not _worker_stop_event.is_set():
        try:
            processed = process_next_capture_job()
        except Exception:
            logger.exception("Capture ingest worker crashed during processing worker=%d", worker_index)
            processed = False
        if not processed:
            _worker_stop_event.wait(poll_seconds)


def start_background_workers() -> None:
    """Start idempotent in-process ingestion workers when async mode is enabled."""
    if _CAPTURE_MODE != "async" or _INGEST_WORKERS <= 0:
        return

    with _worker_lock:
        active_threads = [thread for thread in _worker_threads if thread.is_alive()]
        if active_threads:
            _worker_threads[:] = active_threads
            return

        _worker_threads.clear()
        _worker_stop_event.clear()
        for idx in range(_INGEST_WORKERS):
            thread = threading.Thread(
                target=_capture_ingest_worker,
                args=(idx + 1,),
                daemon=True,
                name=f"openbrain-capture-{idx + 1}",
            )
            thread.start()
            _worker_threads.append(thread)


def stop_background_workers(timeout: float = 1.0) -> None:
    """Stop background workers. Primarily used by tests."""
    with _worker_lock:
        if not _worker_threads:
            return
        _worker_stop_event.set()
        for thread in list(_worker_threads):
            thread.join(timeout=timeout)
        _worker_threads.clear()


_ensure_storage_infrastructure()


def capture_thought(thought: str, user_id: str = "default") -> str:
    """
    Saves a new memory into the user's Open Brain.
    Call this when the user makes a decision, meets someone, or specifies a constraint.
    """
    thought = (thought or "").strip()
    if not thought:
        return "Warning: Please provide a non-empty thought."

    logger.info("Capturing thought for user=%s (len=%d)", user_id, len(thought))
    started_at = time.perf_counter()

    try:
        raw_capture_id, job_id = _capture_raw_and_enqueue(thought, user_id=user_id, source="capture_thought")
        queue_stats = get_capture_job_stats()
        accept_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Capture accepted user=%s raw_capture_id=%s job_id=%s accept_ms=%.1f queue_depth=%d processing=%d failed=%d",
            user_id,
            raw_capture_id,
            job_id,
            accept_ms,
            queue_stats.get("queued", 0),
            queue_stats.get("processing", 0),
            queue_stats.get("failed", 0),
        )
    except Exception as e:
        logger.error("Memory capture enqueue failed for user=%s: %s", user_id, e)
        return f"Warning: Failed to persist memory. Error: {e}"

    if _CAPTURE_MODE == "sync":
        if job_id is None:
            return "Warning: Failed to enqueue memory for processing."
        if _process_capture_job_by_id(job_id):
            return "Successfully captured thought into memory."
        return "Warning: Failed to fully commit memory during synchronous processing."

    return _CAPTURE_ACK_TEXT


def search_brain(query: str, user_id: str = "default", debug: bool = False) -> str:
    """
    Searches the user's past memories via vector similarity and knowledge graph.
    Returns answer-first output with concise sources. Use debug=True for trace output.
    """
    logger.info("Searching brain for user=%s query=%r debug=%s", user_id, query[:80], debug)

    memories_payload = m.search(query=query, user_id=user_id, limit=10)
    vector_results, relations = _extract_vector_and_relations(memories_payload)

    raw_matches: list[dict] = []
    try:
        raw_matches = _search_raw_captures(query=query, user_id=user_id, limit=8)
    except Exception as e:
        logger.warning("Raw capture search failed for user=%s: %s", user_id, e)

    candidates = _build_candidates(query, vector_results=vector_results, relations=relations, raw_matches=raw_matches)
    ranked, ranking_debug = _rank_evidence(query=query, candidates=candidates)

    if debug:
        return _format_debug_report(query=query, ranking_debug=ranking_debug, evidence=ranked)

    response = _format_search_response(query=query, evidence=ranked)
    return response
