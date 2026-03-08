"""
Apache AGE Graph Provider for Mem0.

Provides a Cypher-based graph store implementation on top of PostgreSQL + Apache AGE,
integrated with Mem0's BaseGraphProvider interface.

Security notes:
  - AGE's cypher() function does NOT support psycopg2 %s parameter binding inside
    Cypher queries.  All values interpolated into Cypher strings are validated/escaped
    through strict helpers below.
  - Labels are validated against a strict alphanumeric regex (^[A-Za-z][A-Za-z0-9_]{0,63}$).
  - Property string values are escaped using double-backslash then escaped-quote to
    prevent Cypher injection.
"""

import json
import logging
import os
import re
from contextlib import contextmanager

from openai import OpenAI
from psycopg2 import pool
from psycopg2.extras import Json
from pydantic import BaseModel, Field

class GraphWriteError(Exception):
    """Raised when a graph operation fails and cannot be recovered."""
    pass

class _Entity(BaseModel):
    name: str = Field(description="The name of the entity, e.g. 'Project Solar' or 'test_user'")
    label: str = Field(description="A generic node label like 'Project', 'Person', 'Concept'")

class _Edge(BaseModel):
    source: str = Field(description="Name of the source entity")
    source_label: str = Field(description="Label of the source entity")
    relationship: str = Field(description="The relationship, e.g. 'WORKING_ON', 'PREFERS'")
    target: str = Field(description="Name of the target entity")
    target_label: str = Field(description="Label of the target entity")

class _GraphExtraction(BaseModel):
    nodes: list[_Entity]
    edges: list[_Edge]

# We don't inherit from BaseGraphProvider because it doesn't exist in mem0ai 1.0.5
# We implement the interface expected by Memory.py directly instead.

logger = logging.getLogger(__name__)
_SINGLE_USER_ID = "default"

# ── Validation Constants ──────────────────────
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_GRAPH_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_PROP_VALUE_MAX_LEN = 2000
_SEARCH_TERM_MAX_LEN = 200
_SEARCH_TERM_RE = re.compile(r"^[A-Za-z0-9\s\-_.@]+$")


class ApacheAGEProvider:
    """Mem0-compatible graph provider backed by Apache AGE (Cypher-on-Postgres)."""

    def add(self, data: str, filters: dict):
        """Minimal add implementation for Mem0 compatibility."""
        logger.debug("Graph add called with data=%r filters=%r", data[:50], filters)

        user_id = _SINGLE_USER_ID
        
        try:
            extraction_instruction = (
                "Extract entities and relationships from the text to build a knowledge graph. "
                f"Resolve self-references like 'I' or 'my' to the local OpenBrain user '{user_id}'. "
                "Make entities discrete and relationships concise."
            )
            json_shape_instruction = (
                "Return ONLY valid JSON with this shape: "
                "{\"nodes\": [{\"name\": \"...\", \"label\": \"...\"}], "
                "\"edges\": [{\"source\": \"...\", \"source_label\": \"...\", "
                "\"relationship\": \"...\", \"target\": \"...\", \"target_label\": \"...\"}]}"
            )

            messages = [
                {
                    "role": "system",
                    "content": extraction_instruction
                },
                {
                    "role": "user",
                    "content": data
                }
            ]

            if hasattr(self._openai.chat.completions, "parse"):
                response = self._openai.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format=_GraphExtraction,
                )
                ext = response.choices[0].message.parsed
            else:
                # Fallback for SDKs that don't expose parse(): force JSON mode and validate.
                response = self._openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"{extraction_instruction} {json_shape_instruction}"},
                        {"role": "user", "content": data},
                    ],
                    response_format={"type": "json_object"},
                )
                raw = (response.choices[0].message.content or "").strip()
                payload = json.loads(raw) if raw else {}
                if hasattr(_GraphExtraction, "model_validate"):
                    ext = _GraphExtraction.model_validate(payload)
                else:
                    ext = _GraphExtraction.parse_obj(payload)
            
            nodes = [
                {"label": n.label, "properties": {"name": n.name, "user_id": user_id}}
                for n in ext.nodes
            ]
            edges = [
                {
                    "source": e.source, "source_label": e.source_label,
                    "target": e.target, "target_label": e.target_label,
                    "relationship": e.relationship
                }
                for e in ext.edges
            ]
            
            if nodes:
                self.add_nodes(nodes)
            if edges:
                self.add_edges(edges)
                
            return {"added_entities": nodes, "deleted_entities": []}
            
        except Exception as e:
            logger.error("Graph extraction failed, skipping graph write: %s", e)
            self._log_to_dlq("extraction", {"data": data[:200], "filters": filters}, e)
            return {"added_entities": [], "deleted_entities": []}

    def get_all(self, filters, limit=100):
        return []

    def delete_all(self, filters):
        pass

    def __init__(
        self,
        dbname: str,
        user: str,
        password: str,
        host: str = "localhost",
        port: int = 5432,
        graph_name: str = "brain_graph",
        openai_api_key: str = None,
    ):
        if not _GRAPH_NAME_RE.match(graph_name):
            raise ValueError(f"Invalid graph_name: {graph_name!r}")
        self.graph_name = graph_name

        self._openai = OpenAI(
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY")
        )

        self.pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
        )
        self._initialize_infrastructure()

    # ── Connection Management ─────────────────

    @contextmanager
    def _get_cursor(self):
        """Yield (conn, cur) with AGE loaded.  Guarantees clean connection state."""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("LOAD 'age';")
                cur.execute("SET search_path TO ag_catalog, public;")
                yield conn, cur
        except Exception:
            conn.rollback()
            raise
        finally:
            # Reset connection to a clean state before returning to pool
            try:
                if not conn.closed:
                    conn.rollback()
            except Exception:
                pass
            self.pool.putconn(conn)

    # ── Validation / Escaping Helpers ─────────

    @staticmethod
    def _validate_label(label: str) -> str:
        """Validate a Cypher node/edge label.  Must be alphanumeric + underscores."""
        if not _LABEL_RE.match(label):
            raise ValueError(f"Invalid Cypher label: {label!r}")
        return label

    @classmethod
    def _coerce_label(cls, label: str, fallback: str = "Entity") -> str:
        """Normalize LLM-provided labels into safe Cypher identifiers.

        AGE labels cannot contain spaces or punctuation. Instead of dropping entities,
        we convert common label forms like "CB Channel" -> "CB_Channel".
        """
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
        """Escape a value for embedding in a Cypher single-quoted literal.

        Strategy:
          1. Enforce max length.
          2. Escape backslashes first (\\  ->  \\\\).
          3. Escape single quotes  ('  ->  \\').
        """
        if len(value) > _PROP_VALUE_MAX_LEN:
            raise ValueError(
                f"Property value exceeds max length ({_PROP_VALUE_MAX_LEN})"
            )
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _sanitize_props(self, props: dict) -> str:
        """Build a Cypher property map string like  name: 'val', age: '30'."""
        if not props:
            return ""

        parts = []
        for key, val in props.items():
            clean_key = re.sub(r"[^A-Za-z0-9_]", "", str(key))
            if not clean_key:
                continue
            clean_val = self._escape_cypher_string(str(val))
            parts.append(f"{clean_key}: '{clean_val}'")

        return ", ".join(parts)

    @staticmethod
    def _validate_search_term(term: str) -> str:
        """Validate and return a search term safe for Cypher CONTAINS."""
        term = term.strip()
        if len(term) > _SEARCH_TERM_MAX_LEN:
            term = term[:_SEARCH_TERM_MAX_LEN]
        if not _SEARCH_TERM_RE.match(term):
            raise ValueError(f"Invalid search term: {term!r}")
        return term.replace("\\", "\\\\").replace("'", "\\'")

    # ── Dead Letter Queue ─────────────────────

    def _log_to_dlq(self, payload_type: str, payload: dict, error: Exception):
        """Log failed graph operations to a dead-letter queue table."""
        try:
            with self._get_cursor() as (conn, cur):
                cur.execute(
                    """
                    INSERT INTO memory_store.graph_dlq (payload_type, payload, error_message)
                    VALUES (%s, %s, %s);
                    """,
                    (payload_type, Json(payload), str(error)),
                )
                conn.commit()
            logger.error("Graph write failed — logged to DLQ: %s", error)
        except Exception as dlq_error:
            logger.critical(
                "DLQ WRITE FAILED. Data lost: %s. Error: %s", payload, dlq_error
            )

    # ── Infrastructure ────────────────────────

    def _initialize_infrastructure(self):
        """Create the graph (if missing) and the DLQ table on first connect."""
        with self._get_cursor() as (conn, cur):
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

    # ── Graph Operations ──────────────────────

    def add_nodes(self, nodes: list[dict]):
        """Merge nodes into the graph.  Each dict must have 'label' and 'properties'."""
        for node in nodes:
            try:
                with self._get_cursor() as (conn, cur):
                    label = self._coerce_label(node.get("label", "Entity"), fallback="Entity")
                    prop_str = self._sanitize_props(node.get("properties", {}))

                    query = f"""
                    SELECT * FROM cypher('{self.graph_name}', $$
                        MERGE (n:{label} {{{prop_str}}})
                        RETURN n
                    $$) AS (n agtype);
                    """
                    cur.execute(query)
                    conn.commit()
            except ValueError as e:
                logger.warning("Skipping invalid node: %s", e)
            except Exception as e:
                self._log_to_dlq("node", node, e)
                raise GraphWriteError(f"Failed to write node to graph: {e}") from e

    def add_edges(self, edges: list[dict]):
        """Merge edges into the graph.  Each dict needs source, target, relationship."""
        for edge in edges:
            try:
                with self._get_cursor() as (conn, cur):
                    source = self._escape_cypher_string(edge["source"])
                    target = self._escape_cypher_string(edge["target"])
                    rel = self._coerce_label(edge.get("relationship", "RELATED_TO"), fallback="RELATED_TO")

                    source_label = edge.get("source_label", "")
                    target_label = edge.get("target_label", "")

                    s_match = f":{self._coerce_label(source_label, fallback='Entity')}" if source_label else ""
                    t_match = f":{self._coerce_label(target_label, fallback='Entity')}" if target_label else ""

                    query = f"""
                    SELECT * FROM cypher('{self.graph_name}', $$
                        MATCH (a{s_match} {{name: '{source}'}}), (b{t_match} {{name: '{target}'}})
                        MERGE (a)-[r:{rel}]->(b)
                        RETURN r
                    $$) AS (r agtype);
                    """
                    cur.execute(query)
                    conn.commit()
            except ValueError as e:
                logger.warning("Skipping invalid edge: %s", e)
            except Exception as e:
                self._log_to_dlq("edge", edge, e)
                raise GraphWriteError(f"Failed to write edge to graph: {e}") from e

    def search(self, query: str, filters: dict = None, limit: int = 5):
        """Search the knowledge graph for nodes related to the query terms.

        Returns a dict with 'nodes' and 'edges' lists containing parsed dicts.
        """
        safe_limit = max(1, min(int(limit), 100))
        user_id = _SINGLE_USER_ID
        safe_user = self._escape_cypher_string(str(user_id))

        # Extract searchable terms > 2 chars, validate each
        terms = []
        for raw_term in query.split():
            raw_term = raw_term.strip()
            if len(raw_term) <= 2:
                continue
            try:
                terms.append(self._validate_search_term(raw_term))
            except ValueError:
                logger.debug("Skipping invalid search term: %r", raw_term)
                continue

        if terms:
            conditions = " OR ".join(
                f"toLower(a.name) CONTAINS toLower('{t}')" for t in terms
            )
            where_clause = f"a.user_id = '{safe_user}' AND ({conditions})"
        else:
            where_clause = f"a.user_id = '{safe_user}'"

        results: dict = {"nodes": [], "edges": []}

        try:
            with self._get_cursor() as (conn, cur):
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

        except Exception as e:
            logger.error("Graph search failed: %s", e)

        return results

    # ── Agtype Parsing ────────────────────────

    @staticmethod
    def _parse_agtype(value) -> dict:
        """Parse an AGE agtype value into a Python dict.

        AGE returns vertices as:  {...}::vertex
        and edges as:             {...}::edge
        """
        if value is None:
            return {}
        raw = str(value)
        # Strip the ::vertex or ::edge type suffix
        if "::" in raw:
            raw = raw[: raw.rfind("::")]
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, Exception):
            return {"raw": raw}

    # ── Cleanup ───────────────────────────────

    def __del__(self):
        if hasattr(self, "pool") and self.pool:
            self.pool.closeall()
