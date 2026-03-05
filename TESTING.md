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
*Expected: `age`, `vector`, `pg_partman`, `pg_cron` should be present.*

---

## 2. Unit Testing
Run the full test suite (29 tests) covering the Apache AGE provider, Cypher injection protection, and regression fixes.

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

### Check Partitioning (pg_partman)
Verify that the `memories` table is partitioned by month:
```bash
docker compose exec open-brain-db psql -U brain_user -d open_brain -c "\d+ memory_store.memories"
```

### Check HNSW Index Choice
Verify that the semantic index is created with correct parameters ($m=16, ef\_construction=128$):
```bash
docker compose exec open-brain-db psql -U brain_user -d open_brain -c "SELECT indexdef FROM pg_indexes WHERE indexname LIKE '%embedding%';"
```

### Inspect Knowledge Graph (Apache AGE)
Check if any nodes have been created in the graph:
```bash
docker compose exec open-brain-db psql -U brain_user -d open_brain -c "SELECT * FROM cypher('brain_graph', \$\$ MATCH (n) RETURN n \$\$) AS (n agtype);"
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
npx @modelcontextprotocol/inspector http://localhost:8000/mcp
```

### Manual E2E Smoke Test (Direct Tool Call)
You can still run a python script to test the logic directly:
1. Ensure the containers are running.
2. Run:
```bash
docker compose exec open-brain-mcp python -c "from open_brain_mcp import capture_thought; print(capture_thought('Testing background brain.', user_id='test1'))"
```

### Cleanup After Testing
To reset the environment completely:
```bash
docker compose down -v
```
*(The `-v` flag removes the Postgres data volume, allowing you to start fresh with `init.sql`).*

> [!IMPORTANT]
> **LLM Quota Requirement**: The `capture_thought` tool uses the LLM to extract facts from text. If your OpenAI Key only has embedding access (or is out of quota), the `m.add()` call will fail. Ensure your key has access to `gpt-4o-mini` (default) or update the config in `open_brain_mcp.py`.
