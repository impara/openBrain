# OpenBrain Testing & Verification Guide

This document provides all the commands necessary to verify the security, infrastructure, and logic of the OpenBrain project.

## 1. Environment & Infrastructure

### Verify Docker Build
Ensure images build without errors:
```bash
docker compose build
```

### Check Running Containers
```bash
docker compose ps
```

### Verify Database Extensions
Confirm all required plugins are loaded and initialized:
```bash
docker compose exec open-brain-db psql -U brain_user -d open_brain -c "SELECT extname, extversion FROM pg_extension;"
```
*Expected: `age` and `vector` should be present.*

---

## 2. Unit Testing
Run the full test suite covering provider settings, provider adapters, runtime repositories, Apache AGE graph behavior, async queue behavior, auto-classification, and regression fixes.

### Run Local Tests
Requires dependencies installed (`pip install -r requirements.txt`):
```bash
python -m pytest tests/ -v
```

### Run Tests inside Docker
```bash
docker compose exec open-brain-mcp python -m pytest tests/ -v
```

---

## 3. Database Verification (Manual Audit)

### Check Runtime Tables
Verify that only the runtime-owned tables are present:
```bash
docker compose exec open-brain-db psql -U brain_user -d open_brain -c "\dt memory_store.*"
```

### Check HNSW Index Choice
Verify that the runtime chunk index exists:
```bash
docker compose exec open-brain-db psql -U brain_user -d open_brain -c "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'memory_store' AND indexname LIKE '%memory_chunks%';"
```

### Inspect Knowledge Graph (Apache AGE)
Check if any nodes have been created in the graph:
```bash
docker compose exec open-brain-db psql -U brain_user -d open_brain -c "SELECT * FROM cypher('brain_graph_v2', \$\$ MATCH (n) RETURN n \$\$) AS (n agtype);"
```

---

## 4. End-to-End Verification

### Check MCP Server Logs
Monitor the MCP server startup and verify Streamable HTTP is running on `0.0.0.0:8000`:
```bash
docker compose logs -f open-brain-mcp
```
*Expected: `Uvicorn running on http://0.0.0.0:8000`*

### Testing with MCP Inspector
Since the server is now running in **Streamable HTTP mode**, you can test it using the [MCP Inspector](https://github.com/modelcontextprotocol/inspector):
```bash
npx @modelcontextprotocol/inspector http://localhost:8001/mcp
```

### Manual E2E Smoke Test (Direct Tool Call)
You can still run a python script to test the logic directly:
1. Ensure the containers are running.
2. Run:
```bash
docker compose exec open-brain-mcp python -c "from brain_core import capture_thought; print(capture_thought('Testing background brain.'))"
```

### Telegram Bot Verification
If `TELEGRAM_BOT_TOKEN` is set in `.env`:
```bash
# Check the Telegram bot container is running
docker compose ps open-brain-telegram

# Check logs for successful startup
docker compose logs --tail=10 open-brain-telegram
```
*Expected: `Starting OpenBrain Telegram bot` with no errors.*

Then in Telegram:
1. Send `/start` to your bot — should get a welcome message
2. Send `/remember I prefer dark mode` — should confirm capture or queueing
3. Send `/search dark mode` — should return the memory immediately from raw capture or shortly after background indexing

### Cleanup After Testing
To reset the environment completely:
```bash
docker compose down -v
```
*(The `-v` flag removes the Postgres data volume, allowing you to start fresh with `init.sql`).*

### Fast DB Truncate (Keep Containers + Volume)
Use this when you want to re-test retrieval from a clean database without deleting Docker volumes.

```bash
docker update --restart=no open_brain_mcp open_brain_telegram
docker stop open_brain_mcp open_brain_telegram

docker exec open_brain_postgres psql -U brain_user -d open_brain -v ON_ERROR_STOP=1 -c "
TRUNCATE TABLE memory_store.raw_captures RESTART IDENTITY CASCADE;
TRUNCATE TABLE memory_store.graph_dlq RESTART IDENTITY;
DO \$\$
DECLARE r RECORD;
BEGIN
  FOR r IN SELECT tablename FROM pg_tables WHERE schemaname='brain_graph_v2' LOOP
    EXECUTE format('TRUNCATE TABLE brain_graph_v2.%I RESTART IDENTITY CASCADE', r.tablename);
  END LOOP;
END
\$\$;
"

docker exec open_brain_postgres psql -U brain_user -d open_brain -c "
SELECT 'memory_store.raw_captures' AS table_name, COUNT(*) AS rows FROM memory_store.raw_captures
UNION ALL SELECT 'memory_store.capture_jobs', COUNT(*) FROM memory_store.capture_jobs
UNION ALL SELECT 'memory_store.memory_chunks', COUNT(*) FROM memory_store.memory_chunks
UNION ALL SELECT 'memory_store.graph_dlq', COUNT(*) FROM memory_store.graph_dlq;
"

docker update --restart=unless-stopped open_brain_mcp open_brain_telegram
docker start open_brain_mcp open_brain_telegram


```

Notes:
- The runtime retrieval path uses `memory_store.raw_captures`, `memory_store.capture_jobs`, and `memory_store.memory_chunks`.
- `TRUNCATE memory_store.raw_captures ... CASCADE` also clears `memory_store.capture_jobs` and `memory_store.memory_chunks`.
- The active Apache AGE graph schema is `brain_graph_v2`, not `brain_graph`.

If your local Postgres volume predates the current schema and reset/cleanup becomes noisy, use `docker compose down -v` for a full local reset.

> [!IMPORTANT]
> **Provider Capability Requirement**: The `capture_thought` tool needs both an embedding provider and a structured-generation provider. Ensure your configured provider/model pair supports the chosen role and that `OPENBRAIN_EMBEDDING_DIMS` matches the actual embedding output size.
