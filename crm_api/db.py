import os

import psycopg2
from psycopg2.extras import RealDictCursor


def get_connection():
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "open_brain"),
        user=os.getenv("POSTGRES_USER", "brain_user"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        cursor_factory=RealDictCursor,
    )
    return conn


def fetchall(query: str, params: tuple | None = None):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            return cur.fetchall()
    finally:
        conn.close()


def fetchone(query: str, params: tuple | None = None):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            return cur.fetchone()
    finally:
        conn.close()


def ensure_crm_schema() -> None:
    """
    Ensure the CRM tables exist.

    Rationale: OpenBrain's main runtime creates tables at startup via ensure_infrastructure(),
    but the CRM API can be deployed/restarted independently. This keeps crm-api robust.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS crm;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS crm.contacts (
                    id BIGSERIAL PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    email TEXT,
                    phone TEXT,
                    company TEXT,
                    role TEXT,
                    location TEXT,
                    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                    notes TEXT,
                    canonical_id BIGINT REFERENCES crm.contacts(id) ON DELETE SET NULL,
                    last_modified_by TEXT NOT NULL DEFAULT 'agent',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                -- Unique constraint (allows multiple NULL emails) so ON CONFLICT (email) is valid.
                CREATE UNIQUE INDEX IF NOT EXISTS idx_crm_contacts_email_unique
                ON crm.contacts (email);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_crm_contacts_company_name
                ON crm.contacts (company, full_name);
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS crm.contact_aliases (
                    id BIGSERIAL PRIMARY KEY,
                    contact_id BIGINT NOT NULL REFERENCES crm.contacts(id) ON DELETE CASCADE,
                    alias_name TEXT,
                    alias_email TEXT,
                    source TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS crm.interactions (
                    id BIGSERIAL PRIMARY KEY,
                    contact_id BIGINT NOT NULL REFERENCES crm.contacts(id) ON DELETE CASCADE,
                    raw_capture_id BIGINT REFERENCES memory_store.raw_captures(id) ON DELETE SET NULL,
                    channel TEXT NOT NULL DEFAULT 'chat',
                    direction TEXT NOT NULL DEFAULT 'outbound',
                    summary TEXT NOT NULL,
                    source TEXT,
                    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_crm_interactions_contact_occurred
                ON crm.interactions (contact_id, occurred_at DESC);
                """
            )
        conn.commit()
    finally:
        conn.close()

