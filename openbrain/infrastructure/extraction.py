from __future__ import annotations

import logging

from openbrain.domain.models import GraphEdge, GraphExtraction, GraphNode
from openbrain.domain.ports import BulletExtractor, GraphExtractor, StructuredGenerationProvider
from openbrain.domain.text import heuristic_bullets

logger = logging.getLogger("open_brain")


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
