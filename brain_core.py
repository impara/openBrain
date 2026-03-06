"""
brain_core.py — Shared memory layer for OpenBrain.

Initializes the Mem0 + Apache AGE dual-stack and exposes
capture_thought() / search_brain() for any client (MCP, Telegram, etc.).
"""

import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import psycopg2
from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI

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

_LLM_MODEL = os.environ.get("OPENBRAIN_SUMMARY_MODEL", "gpt-4o-mini")
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

logger.info("OpenBrain initialized — Mem0 + AGE dual-stack ready")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_raw_capture_table() -> None:
    """Create raw capture table for debugging remember/search quality."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
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
            conn.commit()
    finally:
        conn.close()


def _save_raw_capture(thought: str, user_id: str, source: str = "capture_thought") -> int | None:
    """Persist the exact incoming text for audit/debug visibility."""
    if not thought:
        return None

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_store.raw_captures (user_id, source, content, content_len)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (user_id, source, thought, len(thought)),
            )
            row = cur.fetchone()
            conn.commit()
            return row[0] if row else None
    finally:
        conn.close()


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

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
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
    finally:
        conn.close()


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


_ensure_raw_capture_table()


def capture_thought(thought: str, user_id: str = "default") -> str:
    """
    Saves a new memory into the user's Open Brain.
    Call this when the user makes a decision, meets someone, or specifies a constraint.
    """
    logger.info("Capturing thought for user=%s (len=%d)", user_id, len(thought))
    raw_capture_id = None
    try:
        raw_capture_id = _save_raw_capture(thought, user_id=user_id, source="capture_thought")
    except Exception as e:
        logger.warning("Raw capture save failed for user=%s: %s", user_id, e)

    try:
        result = m.add(
            messages=[{"role": "user", "content": thought}],
            user_id=user_id,
            metadata={"raw_capture_id": raw_capture_id, "ingest_mode": "primary"},
        )
        logger.debug("Mem0 add result: %s", result)
        added_count = _extract_added_count(result)
        if added_count == 0:
            logger.info(
                "Mem0 add returned no explicit new items for user=%s, triggering fallback ingestion",
                user_id,
            )
            fallback_added = _fallback_ingest_facts(thought, user_id=user_id, raw_capture_id=raw_capture_id)
            if fallback_added > 0:
                logger.info(
                    "Fallback ingestion added %d factual memories for user=%s",
                    fallback_added,
                    user_id,
                )
            else:
                logger.info(
                    "Fallback ingestion produced no new vector memories for user=%s",
                    user_id,
                )
        return "Successfully captured thought into memory."
    except Exception as e:
        logger.error("Memory capture failed (potential ghost memory): %s", e)
        return f"Warning: Failed to fully commit memory. Error: {e}"


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
