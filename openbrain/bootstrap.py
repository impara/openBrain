from __future__ import annotations

import logging
from functools import lru_cache

from dotenv import load_dotenv

from openbrain.application.service import OpenBrainApplication
from openbrain.infrastructure.db import Database
from openbrain.infrastructure.extraction import (
    LLMBulletExtractor,
    LLMGraphExtractor,
    LLMManagedMemoryExtractor,
    LLMManagedMemoryResolver,
)
from openbrain.infrastructure.providers import build_embedding_provider, build_structured_generation_provider
from openbrain.infrastructure.repositories import AGEGraphRepository, OpenBrainRepositories
from openbrain.settings import OpenBrainSettings


def build_openbrain(settings: OpenBrainSettings | None = None) -> OpenBrainApplication:
    load_dotenv()
    settings = settings or OpenBrainSettings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    db = Database(settings)
    repositories = OpenBrainRepositories(db, settings)
    repositories.ensure_infrastructure()

    graph_repo = AGEGraphRepository(db, graph_name=settings.graph_name)
    graph_repo.ensure_infrastructure()

    llm_provider = build_structured_generation_provider(settings)
    embedding_provider = build_embedding_provider(settings)

    app = OpenBrainApplication(
        settings=settings,
        repositories=repositories,
        vector_repo=repositories,
        graph_repo=graph_repo,
        managed_repo=repositories,
        embedding_provider=embedding_provider,
        bullet_extractor=LLMBulletExtractor(llm_provider),
        graph_extractor=LLMGraphExtractor(llm_provider),
        managed_extractor=LLMManagedMemoryExtractor(llm_provider),
        managed_resolver=LLMManagedMemoryResolver(llm_provider),
    )
    logging.getLogger("open_brain").info(
        "OpenBrain initialized — provider-agnostic runtime ready (llm=%s/%s, embedding=%s/%s, capture_mode=%s, ingest_workers=%d)",
        settings.llm.provider,
        settings.llm.model,
        settings.embedding.provider,
        settings.embedding.model,
        settings.capture_mode,
        settings.ingest_workers,
    )
    return app


@lru_cache(maxsize=1)
def get_openbrain() -> OpenBrainApplication:
    return build_openbrain()
