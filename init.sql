-- ============================================================================
-- OpenBrain Database Initialization
--
-- This script runs once on first container boot via docker-entrypoint-initdb.d.
-- The runtime-owned tables are created by application bootstrap because vector
-- dimensions are configuration-driven. This init step only guarantees the
-- extensions and base schema required by the runtime.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;
CREATE EXTENSION IF NOT EXISTS age;

CREATE SCHEMA IF NOT EXISTS memory_store;
