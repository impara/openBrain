from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .models import EvidenceItem, RawCaptureMatch
from .text import normalized_text, tokenize


MAX_EVIDENCE = 5
SCORE_THRESHOLD = 0.52
SOURCE_WEIGHT = {"vector": 1.0, "graph": 0.82, "raw": 0.68}
MAX_DEBUG_CHARS = 3800
MAX_SOURCE_TEXT = 180
MAX_ANSWER_TEXT = 520


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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    matched = 0
    for query_token in set(query_tokens):
        for text_token in text_tokens:
            if query_token == text_token:
                matched += 1
                break
            if len(query_token) >= 4 and (query_token in text_token or text_token in query_token):
                matched += 1
                break
    overlap = matched / len(set(query_tokens))
    return max(0.0, min(1.0, overlap))


def _vector_similarity(distance: Any) -> float:
    if distance is None:
        return 0.55
    dist = _safe_float(distance, default=1.0)
    similarity = 1.0 - min(max(dist, 0.0), 1.0)
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


def _format_relation(relation: Any) -> str:
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


def relation_lines(relations: Any) -> list[str]:
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
        return [line for line in (_format_relation(item) for item in relations) if line]
    single = _format_relation(relations)
    return [single] if single else []


def build_candidates(
    query: str,
    vector_results: list[Any],
    relations: Any,
    raw_matches: list[RawCaptureMatch],
) -> list[EvidenceItem]:
    query_tokens = tokenize(query)
    candidates: list[EvidenceItem] = []

    for row in vector_results:
        if isinstance(row, dict):
            text = row.get("memory") or row.get("text") or row.get("content") or str(row)
            created_at = row.get("created_at")
            score_raw = _safe_float(row.get("score"), default=1.0)
            row_meta = row.get("metadata") or {}
            metadata = {
                "id": row.get("id"),
                "created_at": created_at,
                "vector_distance": score_raw,
                "origin": row_meta.get("origin") or row.get("source"),
                "ingest_mode": row_meta.get("ingest_mode"),
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

    for relation_text in relation_lines(relations):
        candidates.append(EvidenceItem(source="graph", text=relation_text.strip(), score_raw=None, metadata={}))

    for raw in raw_matches:
        snippet = _extract_relevant_snippet(raw.content, query_tokens=query_tokens)
        if snippet:
            candidates.append(
                EvidenceItem(
                    source="raw",
                    text=snippet,
                    score_raw=None,
                    metadata={
                        "raw_id": raw.id,
                        "created_at": raw.created_at,
                        "content_len": raw.content_len,
                        "origin": raw.source or "import",
                    },
                )
            )
    return candidates


def _score_candidate(item: EvidenceItem, query_tokens: list[str]) -> dict[str, float]:
    overlap = _query_overlap(query_tokens, item.text)
    source_weight = SOURCE_WEIGHT.get(item.source, 0.5)
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


def rank_evidence(
    query: str,
    candidates: list[EvidenceItem],
    limit: int = MAX_EVIDENCE,
    threshold: float = SCORE_THRESHOLD,
) -> tuple[list[EvidenceItem], dict[str, Any]]:
    query_tokens = tokenize(query)
    selected: list[EvidenceItem] = []
    dropped: list[dict[str, Any]] = []
    debug_candidates: list[dict[str, Any]] = []
    seen_norm: set[str] = set()

    for item in candidates:
        text = item.text.strip()
        if not text:
            dropped.append({"source": item.source, "text": "", "reason": "empty"})
            continue

        norm = normalized_text(text)
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

    selected.sort(key=lambda item: item.score_norm, reverse=True)
    selected = selected[: max(1, min(limit, MAX_EVIDENCE))]

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


def _truncate_text(text: str, max_len: int = MAX_SOURCE_TEXT) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_len:
        return clean
    return f"{clean[:max_len].rstrip()}..."


def synthesize_answer(query: str, evidence: list[EvidenceItem]) -> str:
    del query
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
        norm = normalized_text(text)
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
        return "I found some related memory, but not enough clear facts to answer confidently."

    joined = "; ".join(statements)
    confidence = evidence[0].score_norm
    if confidence < 0.70:
        answer = f"I may be missing context, but your memory suggests: {joined}."
    else:
        answer = f"Based on your stored memories: {joined}."
    return _truncate_text(answer, max_len=MAX_ANSWER_TEXT)


def format_sources(evidence: list[EvidenceItem]) -> list[str]:
    lines: list[str] = []
    source_counts = {"vector": 0, "graph": 0, "raw": 0}
    source_limits = {"vector": 2, "graph": 2, "raw": 1}
    for item in evidence:
        if source_counts.get(item.source, 0) >= source_limits.get(item.source, 1):
            continue
        text = _truncate_text(item.text, max_len=MAX_SOURCE_TEXT)
        origin = item.metadata.get("origin") if item.metadata else None
        label = f"{item.source} • {origin}" if origin else item.source
        lines.append(f"• [{label}] {text}")
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
        if len(lines) >= MAX_EVIDENCE:
            break
    return lines


def format_search_response(query: str, evidence: list[EvidenceItem]) -> str:
    answer = synthesize_answer(query, evidence)
    if not evidence:
        return f"Answer: {answer}"
    source_lines = "\n".join(format_sources(evidence))
    return f"Answer: {answer}\n\nSources:\n{source_lines}"


def format_debug_report(query: str, ranking_debug: dict[str, Any], evidence: list[EvidenceItem]) -> str:
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
    lines.append(synthesize_answer(query, evidence))
    report = "\n".join(lines)
    if len(report) > MAX_DEBUG_CHARS:
        return f"{report[:MAX_DEBUG_CHARS - 3]}..."
    return report
