from __future__ import annotations

import json
import logging
import re
from typing import Any

from psycopg2.extras import Json

from openbrain.domain.models import CaptureJob, CapturePersistResult, MemoryChunk, RawCaptureMatch
from openbrain.domain.text import tokenize
from openbrain.infrastructure.db import Database
from openbrain.settings import OpenBrainSettings

logger = logging.getLogger("open_brain")

_GRAPH_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SEARCH_TERM_RE = re.compile(r"^[A-Za-z0-9\s\-_.@]+$")
_SEARCH_TERM_MAX_LEN = 200
_PROP_VALUE_MAX_LEN = 2000
_SINGLE_USER_ID = "default"


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.12g}" for value in values) + "]"


class OpenBrainRepositories:
    def __init__(self, db: Database, settings: OpenBrainSettings):
        self.db = db
        self.settings = settings

    def ensure_infrastructure(self) -> None:
        dims = self.settings.embedding.dims
        with self.db.cursor(commit=True) as (_, cur):
            # Coolify/compose can start app containers before docker-entrypoint init
            # scripts have finished creating extensions. Ensure the runtime-critical
            # extensions exist before creating any vector or AGE-backed objects.
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS age;")
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
                ALTER TABLE memory_store.raw_captures
                ADD COLUMN IF NOT EXISTS ingest_strategy TEXT NOT NULL DEFAULT 'personal',
                ADD COLUMN IF NOT EXISTS external_id TEXT;
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
                CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_captures_user_source_external_id
                ON memory_store.raw_captures (user_id, source, external_id)
                WHERE external_id IS NOT NULL;
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

            cur.execute("SELECT to_regclass('memory_store.memory_chunks');")
            row = cur.fetchone()
            if row and row[0]:
                cur.execute(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    JOIN pg_namespace n ON c.relnamespace = n.oid
                    WHERE n.nspname = 'memory_store'
                      AND c.relname = 'memory_chunks'
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped;
                    """
                )
                type_row = cur.fetchone()
                expected_type = f"vector({dims})"
                if type_row and type_row[0] != expected_type:
                    raise ValueError(
                        f"memory_store.memory_chunks.embedding is {type_row[0]!r}, expected {expected_type!r}"
                    )
            else:
                cur.execute(
                    f"""
                    CREATE TABLE memory_store.memory_chunks (
                        id BIGSERIAL PRIMARY KEY,
                        raw_capture_id BIGINT NOT NULL REFERENCES memory_store.raw_captures(id) ON DELETE CASCADE,
                        user_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        ingest_mode TEXT NOT NULL,
                        embedding vector({dims}) NOT NULL,
                        embedding_provider TEXT NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding_dims INTEGER NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_chunks_raw_chunk_mode
                ON memory_store.memory_chunks (raw_capture_id, chunk_index, ingest_mode);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_chunks_user_created
                ON memory_store.memory_chunks (user_id, created_at DESC);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_chunks_metadata_gin
                ON memory_store.memory_chunks USING gin (metadata);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_chunks_embedding_hnsw
                ON memory_store.memory_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 128);
                """
            )

    def capture_and_enqueue(
        self,
        thought: str,
        *,
        user_id: str,
        source: str,
        ingest_strategy: str,
        external_id: str | None,
    ) -> CapturePersistResult:
        if not thought:
            return CapturePersistResult(raw_capture_id=None, job_id=None, duplicate=False)

        with self.db.cursor(commit=True) as (_, cur):
            inserted_new = True
            if external_id:
                cur.execute(
                    """
                    WITH inserted AS (
                        INSERT INTO memory_store.raw_captures
                        (user_id, source, content, content_len, ingest_strategy, external_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        RETURNING id
                    )
                    SELECT id, TRUE AS inserted_new
                    FROM inserted
                    UNION ALL
                    SELECT rc.id, FALSE AS inserted_new
                    FROM memory_store.raw_captures rc
                    WHERE rc.user_id = %s
                      AND rc.source = %s
                      AND rc.external_id = %s
                      AND NOT EXISTS (SELECT 1 FROM inserted)
                    LIMIT 1;
                    """,
                    (
                        user_id,
                        source,
                        thought,
                        len(thought),
                        ingest_strategy,
                        external_id,
                        user_id,
                        source,
                        external_id,
                    ),
                )
                row = cur.fetchone()
                raw_capture_id = row[0] if row else None
                inserted_new = bool(row[1]) if row and len(row) > 1 else False
            else:
                cur.execute(
                    """
                    INSERT INTO memory_store.raw_captures
                    (user_id, source, content, content_len, ingest_strategy, external_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (user_id, source, thought, len(thought), ingest_strategy, external_id),
                )
                row = cur.fetchone()
                raw_capture_id = row[0] if row else None

            if raw_capture_id is None:
                return CapturePersistResult(raw_capture_id=None, job_id=None, duplicate=False)

            cur.execute(
                """
                INSERT INTO memory_store.capture_jobs (raw_capture_id, user_id)
                VALUES (%s, %s)
                ON CONFLICT (raw_capture_id)
                DO UPDATE SET user_id = EXCLUDED.user_id
                RETURNING id;
                """,
                (raw_capture_id, user_id),
            )
            job_row = cur.fetchone()
            job_id = job_row[0] if job_row else None
            return CapturePersistResult(raw_capture_id=raw_capture_id, job_id=job_id, duplicate=not inserted_new)

    def claim(self, *, job_id: int | None = None) -> CaptureJob | None:
        filters = ["j.status IN ('pending', 'retry')"]
        params: list[Any] = []
        if job_id is None:
            filters.append("j.available_at <= CURRENT_TIMESTAMP")
        else:
            filters.append("j.id = %s")
            params.append(job_id)

        where_clause = " AND ".join(filters)
        with self.db.cursor(commit=True) as (_, cur):
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
                RETURNING j.id, j.raw_capture_id, j.user_id, r.content, j.attempt_count, r.source, r.ingest_strategy;
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
                source=row[5] if len(row) > 5 else "capture_thought",
                ingest_strategy=row[6] if len(row) > 6 else "personal",
            )

    def mark_done(self, job: CaptureJob) -> None:
        with self.db.cursor(commit=True) as (_, cur):
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

    def mark_retry(self, job: CaptureJob, *, error: Exception, retry_delays: tuple[int, ...]) -> None:
        message = str(error)
        with self.db.cursor(commit=True) as (_, cur):
            if job.attempt_count <= len(retry_delays):
                delay_seconds = retry_delays[job.attempt_count - 1]
                cur.execute(
                    """
                    UPDATE memory_store.capture_jobs
                    SET status = 'retry',
                        available_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                        finished_at = NULL,
                        last_error = %s
                    WHERE id = %s;
                    """,
                    (delay_seconds, message, job.id),
                )
            else:
                cur.execute(
                    """
                    UPDATE memory_store.capture_jobs
                    SET status = 'failed',
                        finished_at = CURRENT_TIMESTAMP,
                        last_error = %s
                    WHERE id = %s;
                    """,
                    (message, job.id),
                )

    def stats(self) -> dict[str, int]:
        stats = {"pending": 0, "processing": 0, "done": 0, "retry": 0, "failed": 0}
        with self.db.cursor() as (_, cur):
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

    def search_raw(self, query: str, *, user_id: str, limit: int = 8) -> list[RawCaptureMatch]:
        terms = tokenize(query)[:8]
        with self.db.cursor() as (_, cur):
            if terms:
                where_parts = " OR ".join(["LOWER(content) LIKE %s"] * len(terms))
                params = [user_id, *[f"%{term}%" for term in terms], limit]
                cur.execute(
                    f"""
                    SELECT id, content, created_at, content_len, source
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
                    SELECT id, content, created_at, content_len, source
                    FROM memory_store.raw_captures
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit),
                )
            rows = cur.fetchall()
        return [
            RawCaptureMatch(
                id=row[0],
                content=row[1],
                created_at=row[2],
                content_len=row[3],
                source=row[4] if len(row) > 4 else None,
            )
            for row in rows
        ]

    def upsert_chunks(self, chunks: list[MemoryChunk]) -> int:
        if not chunks:
            return 0
        with self.db.cursor(commit=True) as (_, cur):
            for chunk in chunks:
                cur.execute(
                    """
                    INSERT INTO memory_store.memory_chunks (
                        raw_capture_id,
                        user_id,
                        source,
                        chunk_index,
                        content,
                        ingest_mode,
                        embedding,
                        embedding_provider,
                        embedding_model,
                        embedding_dims,
                        metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s)
                    ON CONFLICT (raw_capture_id, chunk_index, ingest_mode)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        embedding_provider = EXCLUDED.embedding_provider,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dims = EXCLUDED.embedding_dims,
                        metadata = EXCLUDED.metadata;
                    """,
                    (
                        chunk.raw_capture_id,
                        chunk.user_id,
                        chunk.source,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.ingest_mode,
                        _vector_literal(chunk.embedding),
                        chunk.embedding_provider,
                        chunk.embedding_model,
                        chunk.embedding_dims,
                        Json(chunk.metadata),
                    ),
                )
        return len(chunks)

    def search_vectors(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        limit: int = 10,
        ingest_modes: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        vector = _vector_literal(query_embedding)
        with self.db.cursor() as (_, cur):
            cur.execute("SET LOCAL hnsw.ef_search = 100;")
            if ingest_modes:
                cur.execute(
                    """
                    SELECT id, content, metadata, created_at, source, ingest_mode, embedding <=> %s::vector AS score
                    FROM memory_store.memory_chunks
                    WHERE user_id = %s AND ingest_mode = ANY(%s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (vector, user_id, list(ingest_modes), vector, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, content, metadata, created_at, source, ingest_mode, embedding <=> %s::vector AS score
                    FROM memory_store.memory_chunks
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (vector, user_id, vector, limit),
                )
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "content": row[1],
                "metadata": row[2] or {},
                "created_at": row[3],
                "source": row[4],
                "ingest_mode": row[5],
                "score": row[6],
            }
            for row in rows
        ]


class AGEGraphRepository:
    def __init__(self, db: Database, *, graph_name: str):
        if not _GRAPH_NAME_RE.match(graph_name):
            raise ValueError(f"Invalid graph_name: {graph_name!r}")
        self.db = db
        self.graph_name = graph_name

    def ensure_infrastructure(self) -> None:
        with self.db.cursor(commit=True, age=True) as (conn, cur):
            cur.execute(
                f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM ag_catalog.ag_graph WHERE name = '{self.graph_name}'
                    ) THEN
                        PERFORM ag_catalog.create_graph('{self.graph_name}');
                    END IF;
                END $$;
                """
            )
            cur.execute("CREATE SCHEMA IF NOT EXISTS memory_store;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_store.graph_dlq (
                    id SERIAL PRIMARY KEY,
                    payload_type VARCHAR(50),
                    payload JSONB,
                    error_message TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE
                );
                """
            )
            conn.commit()
        logger.info("AGE graph '%s' infrastructure ready", self.graph_name)

    @staticmethod
    def _validate_label(label: str) -> str:
        if not _LABEL_RE.match(label):
            raise ValueError(f"Invalid Cypher label: {label!r}")
        return label

    @classmethod
    def _coerce_label(cls, label: str, fallback: str = "Entity") -> str:
        raw = (label or "").strip()
        if not raw:
            raw = fallback
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            cleaned = fallback
        if not cleaned[0].isalpha():
            cleaned = f"L_{cleaned}"
        cleaned = cleaned[:64]
        if not cleaned:
            cleaned = fallback
        return cls._validate_label(cleaned)

    @staticmethod
    def _escape_cypher_string(value: str) -> str:
        if len(value) > _PROP_VALUE_MAX_LEN:
            raise ValueError(f"Property value exceeds max length ({_PROP_VALUE_MAX_LEN})")
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _sanitize_props(self, props: dict[str, Any]) -> str:
        if not props:
            return ""
        parts = []
        for key, value in props.items():
            clean_key = re.sub(r"[^A-Za-z0-9_]", "", str(key))
            if not clean_key:
                continue
            clean_val = self._escape_cypher_string(str(value))
            parts.append(f"{clean_key}: '{clean_val}'")
        return ", ".join(parts)

    @staticmethod
    def _validate_search_term(term: str) -> str:
        term = term.strip()
        if len(term) > _SEARCH_TERM_MAX_LEN:
            term = term[:_SEARCH_TERM_MAX_LEN]
        if not _SEARCH_TERM_RE.match(term):
            raise ValueError(f"Invalid search term: {term!r}")
        return term.replace("\\", "\\\\").replace("'", "\\'")

    def _log_to_dlq(self, payload_type: str, payload: dict[str, Any], error: Exception) -> None:
        try:
            with self.db.cursor(commit=True, age=True) as (_, cur):
                cur.execute(
                    """
                    INSERT INTO memory_store.graph_dlq (payload_type, payload, error_message)
                    VALUES (%s, %s, %s);
                    """,
                    (payload_type, Json(payload), str(error)),
                )
        except Exception as dlq_error:
            logger.critical("DLQ write failed for payload=%s error=%s", payload, dlq_error)

    def add_nodes(self, nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            try:
                with self.db.cursor(commit=True, age=True) as (_, cur):
                    label = self._coerce_label(node.get("label", "Entity"), fallback="Entity")
                    prop_str = self._sanitize_props(node.get("properties", {}))
                    cur.execute(
                        f"""
                        SELECT * FROM cypher('{self.graph_name}', $$
                            MERGE (n:{label} {{{prop_str}}})
                            RETURN n
                        $$) AS (n agtype);
                        """
                    )
            except ValueError as exc:
                logger.warning("Skipping invalid node: %s", exc)
            except Exception as exc:
                self._log_to_dlq("node", node, exc)
                raise

    def add_edges(self, edges: list[dict[str, Any]]) -> None:
        for edge in edges:
            try:
                with self.db.cursor(commit=True, age=True) as (_, cur):
                    source = self._escape_cypher_string(edge["source"])
                    target = self._escape_cypher_string(edge["target"])
                    rel = self._coerce_label(edge.get("relationship", "RELATED_TO"), fallback="RELATED_TO")
                    source_label = edge.get("source_label", "")
                    target_label = edge.get("target_label", "")
                    s_match = f":{self._coerce_label(source_label, fallback='Entity')}" if source_label else ""
                    t_match = f":{self._coerce_label(target_label, fallback='Entity')}" if target_label else ""
                    cur.execute(
                        f"""
                        SELECT * FROM cypher('{self.graph_name}', $$
                            MATCH (a{s_match} {{name: '{source}'}}), (b{t_match} {{name: '{target}'}})
                            MERGE (a)-[r:{rel}]->(b)
                            RETURN r
                        $$) AS (r agtype);
                        """
                    )
            except ValueError as exc:
                logger.warning("Skipping invalid edge: %s", exc)
            except Exception as exc:
                self._log_to_dlq("edge", edge, exc)
                raise

    def search(self, query: str, *, limit: int = 5) -> dict[str, list[dict[str, Any]]]:
        safe_limit = max(1, min(int(limit), 100))
        safe_user = self._escape_cypher_string(_SINGLE_USER_ID)
        terms = []
        for raw_term in query.split():
            raw_term = raw_term.strip()
            if len(raw_term) <= 2:
                continue
            try:
                terms.append(self._validate_search_term(raw_term))
            except ValueError:
                continue

        if terms:
            conditions = " OR ".join(f"toLower(a.name) CONTAINS toLower('{term}')" for term in terms)
            where_clause = f"a.user_id = '{safe_user}' AND ({conditions})"
        else:
            where_clause = f"a.user_id = '{safe_user}'"

        results: dict[str, list[dict[str, Any]]] = {"nodes": [], "edges": []}
        try:
            with self.db.cursor(age=True) as (_, cur):
                cur.execute(
                    f"""
                    SELECT * FROM cypher('{self.graph_name}', $$
                        MATCH (a)-[r]->(b)
                        WHERE {where_clause}
                        RETURN a, r, b
                        LIMIT {safe_limit}
                    $$) AS (a agtype, r agtype, b agtype);
                    """
                )
                rows = cur.fetchall()
                for row in rows:
                    results["nodes"].append(self._parse_agtype(row[0]))
                    results["edges"].append(self._parse_agtype(row[1]))
                    results["nodes"].append(self._parse_agtype(row[2]))
        except Exception as exc:
            logger.error("Graph search failed: %s", exc)
        return results

    @staticmethod
    def _parse_agtype(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        raw = str(value)
        if "::" in raw:
            raw = raw[: raw.rfind("::")]
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, Exception):
            return {"raw": raw}
