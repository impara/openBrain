from __future__ import annotations

import logging

from openbrain.domain.models import (
    GraphEdge,
    GraphExtraction,
    GraphNode,
    ManagedMemoryCandidate,
    ManagedMemoryRecord,
    ManagedMemoryResolution,
)
from openbrain.domain.ports import (
    BulletExtractor,
    GraphExtractor,
    ManagedMemoryExtractor,
    ManagedMemoryResolver,
    StructuredGenerationProvider,
)
from openbrain.domain.text import heuristic_bullets, normalized_text

logger = logging.getLogger("open_brain")

_MANAGED_KINDS = {"directive", "preference"}


class LLMBulletExtractor(BulletExtractor):
    def __init__(self, provider: StructuredGenerationProvider):
        self._provider = provider

    def extract(self, text: str, *, max_bullets: int = 3) -> list[str]:
        try:
            payload = self._provider.generate_json(
                system_prompt=(
                    "Extract concise factual bullets from the text. "
                    f"Return JSON only with shape: {{\"bullets\": [\"...\"]}}. "
                    f"Do not output more than {max_bullets} bullets."
                ),
                user_prompt=text,
            )
            bullets = payload.get("bullets", [])
            if not isinstance(bullets, list):
                return heuristic_bullets(text, max_bullets=max_bullets)
            cleaned: list[str] = []
            for bullet in bullets:
                item = str(bullet).strip().lstrip("•*- ").strip()
                if 20 <= len(item) <= 240:
                    cleaned.append(item.rstrip("."))
                if len(cleaned) >= max_bullets:
                    break
            return cleaned or heuristic_bullets(text, max_bullets=max_bullets)
        except Exception as exc:
            logger.warning("Structured bullet extraction failed, using heuristic fallback: %s", exc)
            return heuristic_bullets(text, max_bullets=max_bullets)


class LLMGraphExtractor(GraphExtractor):
    def __init__(self, provider: StructuredGenerationProvider):
        self._provider = provider

    def extract(self, text: str, *, user_id: str) -> GraphExtraction:
        try:
            payload = self._provider.generate_json(
                system_prompt=(
                    "Extract entities and relationships from the text to build a knowledge graph. "
                    f"Resolve self-references like 'I' or 'my' to the local OpenBrain user '{user_id}'. "
                    "Make entities discrete and relationships concise. "
                    "Return JSON only with shape: "
                    "{\"nodes\": [{\"name\": \"...\", \"label\": \"...\"}], "
                    "\"edges\": [{\"source\": \"...\", \"source_label\": \"...\", "
                    "\"relationship\": \"...\", \"target\": \"...\", \"target_label\": \"...\"}]}"
                ),
                user_prompt=text,
            )
            raw_nodes = payload.get("nodes", [])
            raw_edges = payload.get("edges", [])
            nodes = [
                GraphNode(name=str(node.get("name", "")).strip(), label=str(node.get("label", "")).strip())
                for node in raw_nodes
                if isinstance(node, dict) and str(node.get("name", "")).strip()
            ]
            edges = [
                GraphEdge(
                    source=str(edge.get("source", "")).strip(),
                    source_label=str(edge.get("source_label", "")).strip(),
                    relationship=str(edge.get("relationship", "")).strip(),
                    target=str(edge.get("target", "")).strip(),
                    target_label=str(edge.get("target_label", "")).strip(),
                )
                for edge in raw_edges
                if isinstance(edge, dict)
                and str(edge.get("source", "")).strip()
                and str(edge.get("target", "")).strip()
                and str(edge.get("relationship", "")).strip()
            ]
            return GraphExtraction(nodes=nodes, edges=edges)
        except Exception as exc:
            logger.warning("Structured graph extraction failed: %s", exc)
            return GraphExtraction(nodes=[], edges=[])


class LLMManagedMemoryExtractor(ManagedMemoryExtractor):
    def __init__(self, provider: StructuredGenerationProvider):
        self._provider = provider

    def extract(self, text: str, *, forced_kind: str | None = None) -> list[ManagedMemoryCandidate]:
        kind_instruction = (
            f"Only return memories of kind '{forced_kind}'. "
            if forced_kind in _MANAGED_KINDS
            else "Return only stable directives or user preferences. "
        )
        try:
            payload = self._provider.generate_json(
                system_prompt=(
                    "Extract managed memories from the text. "
                    f"{kind_instruction}"
                    "Be conservative. Ignore transient requests, one-off tasks, and normal prose. "
                    "Return JSON only with shape: "
                    "{\"memories\": [{\"kind\": \"directive|preference\", "
                    "\"topic\": \"...\", \"canonical_text\": \"...\", \"evidence_text\": \"...\"}]}"
                ),
                user_prompt=text,
            )
        except Exception as exc:
            logger.warning("Managed memory extraction failed: %s", exc)
            return []

        raw_items = payload.get("memories", [])
        if not isinstance(raw_items, list):
            return []

        results: list[ManagedMemoryCandidate] = []
        seen: set[tuple[str, str, str]] = set()
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "")).strip().lower()
            topic = str(item.get("topic", "")).strip()
            canonical_text = str(item.get("canonical_text", "")).strip()
            evidence_text = str(item.get("evidence_text", "")).strip()
            if forced_kind in _MANAGED_KINDS:
                kind = forced_kind
            if kind not in _MANAGED_KINDS:
                continue
            if not topic or not canonical_text:
                continue
            if len(canonical_text) < 24 or len(canonical_text) > 420:
                continue
            if len(topic) > 120:
                topic = topic[:120].strip()
            dedupe_key = (kind, normalized_text(topic), normalized_text(canonical_text))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            results.append(
                ManagedMemoryCandidate(
                    kind=kind,
                    topic=topic,
                    canonical_text=canonical_text.rstrip("."),
                    evidence_text=evidence_text,
                    metadata={"forced_kind": forced_kind} if forced_kind else {},
                )
            )
        return results


class LLMManagedMemoryResolver(ManagedMemoryResolver):
    def __init__(self, provider: StructuredGenerationProvider):
        self._provider = provider

    def resolve(
        self,
        *,
        candidate: ManagedMemoryCandidate,
        existing: list[ManagedMemoryRecord],
    ) -> ManagedMemoryResolution:
        if not existing:
            return ManagedMemoryResolution(action="supersede", canonical_text=candidate.canonical_text)

        existing_text = "\n".join(
            f"- #{record.id}: {record.canonical_text}"
            for record in existing
        )
        try:
            payload = self._provider.generate_json(
                system_prompt=(
                    "Compare an existing managed memory with a new candidate on the same topic. "
                    "Return JSON only with shape: "
                    "{\"action\": \"merge|supersede\", \"canonical_text\": \"...\"}. "
                    "Choose merge only if they mean the same thing. "
                    "Choose supersede if the new candidate updates, replaces, or materially changes the instruction or preference."
                ),
                user_prompt=(
                    f"Kind: {candidate.kind}\n"
                    f"Topic: {candidate.topic}\n"
                    f"Existing active records:\n{existing_text}\n\n"
                    f"New candidate:\n{candidate.canonical_text}"
                ),
            )
            action = str(payload.get("action", "")).strip().lower()
            canonical_text = str(payload.get("canonical_text", "")).strip() or candidate.canonical_text
        except Exception as exc:
            logger.warning("Managed memory resolution failed, using deterministic fallback: %s", exc)
            action = ""
            canonical_text = candidate.canonical_text

        if action not in {"merge", "supersede"}:
            existing_norm = {normalized_text(record.canonical_text) for record in existing}
            action = "merge" if normalized_text(candidate.canonical_text) in existing_norm else "supersede"

        return ManagedMemoryResolution(
            action=action,
            canonical_text=canonical_text.rstrip("."),
        )
