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
--   of Mem0's abstractions.  Both write paths target the same Postgres instance.
-- ============================================================================

-- 1. Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;
CREATE EXTENSION IF NOT EXISTS pg_partman SCHEMA public;
CREATE EXTENSION IF NOT EXISTS pg_cron SCHEMA pg_catalog;
-- AGE forcefully owns its own schema (ag_catalog), do not coerce it.
CREATE EXTENSION IF NOT EXISTS age;

-- 2. Establish Schema Discipline
CREATE SCHEMA IF NOT EXISTS memory_store;

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
--   The current 12-month setting deletes partitions older than 1 year.
--   Consider adjusting or removing this based on your use case:
--     - Remove retention:       SET retention = NULL
--     - Archive instead:        SET retention_keep_table = true (detaches but preserves)
--     - Longer window:          SET retention = '60 months'
--     - Environment-controlled: Override MEMORY_RETENTION_MONTHS via an init wrapper
UPDATE public.part_config
SET retention = '12 months',
    retention_keep_table = false,
    retention_keep_index = false
WHERE parent_table = 'memory_store.memories';

-- 8. Automate Partition Maintenance (Midnight Cron)
SELECT cron.schedule(
    'partman-maintenance',
    '0 0 * * *',
    $$CALL public.run_maintenance_proc()$$
);