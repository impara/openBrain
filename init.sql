-- ============================================================================
-- OpenBrain Database Initialization
--
-- This script runs once on first container boot via docker-entrypoint-initdb.d.
-- It sets up extensions, schemas, partitioned tables, indexes, and scheduling.
--
-- NOTE ON MEM0 INTEGRATION:
--   Mem0's pgvector provider creates its own internal tables (typically in the
--   public schema) using the configured `collection_name` as a logical grouping.
--   The `memory_store.memories` partitioned table below serves as a *parallel*
--   storage layer for direct queries, analytics, and admin access independent
--   of Mem0's abstractions. OpenBrain's runtime capture/search flow does not
--   write to this table today; the retention settings below apply only to this
--   optional partitioned store.
-- ============================================================================

-- 1. Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;
CREATE EXTENSION IF NOT EXISTS pg_partman SCHEMA public;
CREATE EXTENSION IF NOT EXISTS pg_cron SCHEMA pg_catalog;
-- AGE forcefully owns its own schema (ag_catalog), do not coerce it.
CREATE EXTENSION IF NOT EXISTS age;

-- 2. Establish Schema Discipline
CREATE SCHEMA IF NOT EXISTS memory_store;

-- 2a. Create capture audit + async ingestion tables
CREATE TABLE IF NOT EXISTS memory_store.raw_captures (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'capture_thought',
    content TEXT NOT NULL,
    content_len INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ingest_strategy TEXT NOT NULL DEFAULT 'personal' CHECK (ingest_strategy IN ('personal', 'snippet')),
    external_id TEXT
);

ALTER TABLE memory_store.raw_captures
ADD COLUMN IF NOT EXISTS ingest_strategy TEXT NOT NULL DEFAULT 'personal',
ADD COLUMN IF NOT EXISTS external_id TEXT;

CREATE INDEX IF NOT EXISTS idx_raw_captures_user_created
ON memory_store.raw_captures (user_id, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_captures_user_source_external_id
ON memory_store.raw_captures (user_id, source, external_id)
WHERE external_id IS NOT NULL;

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

CREATE INDEX IF NOT EXISTS idx_capture_jobs_status_available_created
ON memory_store.capture_jobs (status, available_at, created_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_capture_jobs_raw_capture_id
ON memory_store.capture_jobs (raw_capture_id);

-- 3. Create the Parent Partitioned Table
CREATE TABLE memory_store.memories (
    id BIGSERIAL,
    user_id TEXT NOT NULL DEFAULT 'default',
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 4. Create the Template Table (indexes here auto-apply to new partitions)
CREATE TABLE memory_store.memories_template (LIKE memory_store.memories);

-- 5. Apply Indexes to the Template
--
-- HNSW Parameter Rationale (for OpenAI text-embedding-ada-002, 1536 dimensions):
--   m = 16         → Each graph node connects to 16 neighbors.
--                    Good balance of recall vs. memory for 1536-dim vectors.
--                    Lower (8) saves memory but hurts recall; higher (32) diminishing returns.
--   ef_construction = 128  → Search breadth during index build.
--                    128 targets ~95%+ recall. Lower (64) builds faster but weaker recall.
--                    Higher (256) marginal improvement at 2x build cost.
--   vector_cosine_ops  → Cosine similarity, standard for normalized text embeddings.
--
-- For datasets > 1M vectors, consider raising ef_construction to 200 or switching to IVFFlat.
CREATE INDEX ON memory_store.memories_template
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);

CREATE INDEX ON memory_store.memories_template USING gin (metadata);
CREATE INDEX ON memory_store.memories_template (user_id);

-- 6. Hand over control to pg_partman
SELECT public.create_parent(
    p_parent_table := 'memory_store.memories',
    p_control := 'created_at',
    p_type := 'range',
    p_interval := '1 month',
    p_premake := 3,
    p_template_table := 'memory_store.memories_template'
);

-- 7. Configure Data Retention Policy
--
-- RETENTION NOTE:
--   For a "brain" system, auto-deleting old memories may be counterproductive.
--   This retention setting only affects memory_store.memories, the optional
--   partitioned admin table described above.
--   Consider adjusting or removing this based on your use case:
--     - Remove retention:       set MEMORY_RETENTION_MONTHS=0
--     - Archive instead:        SET retention_keep_table = true (detaches but preserves)
--     - Longer window:          SET retention = '60 months'
--     - Environment-controlled: MEMORY_RETENTION_MONTHS is read from
--                               custom Postgres setting openbrain.memory_retention_months
DO $$
DECLARE
    retention_months INTEGER := GREATEST(
        COALESCE(
            NULLIF(current_setting('openbrain.memory_retention_months', true), '')::INTEGER,
            12
        ),
        0
    );
BEGIN
    UPDATE public.part_config
    SET retention = CASE
            WHEN retention_months = 0 THEN NULL
            ELSE format('%s months', retention_months)
        END,
        retention_keep_table = false,
        retention_keep_index = false
    WHERE parent_table = 'memory_store.memories';
END $$;

-- 8. Automate Partition Maintenance (Midnight Cron)
SELECT cron.schedule(
    'partman-maintenance',
    '0 0 * * *',
    $$CALL public.run_maintenance_proc()$$
);
