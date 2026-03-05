# 🧠 OpenBrain

**Agentic memory framework with semantic recall + relational reasoning.**

Give your AI agent a persistent memory that remembers *what* happened (vector search) and *why* it matters (knowledge graph) — all from a single Postgres instance, exposed as MCP tools.

---

## Architecture

```
┌─────────────────────────┐
│     Claude / Agent      │
│  (via MCP protocol)     │
└──────────┬──────────────┘
           │ streamable-http (/mcp)
           │ (or stdio for local)
┌──────────▼──────────────┐
│   OpenBrain MCP Server  │
│   (open_brain_mcp.py)   │
│                         │
│  ┌───────┐  ┌─────────┐ │
│  │ Mem0  │  │  AGE    │ │
│  │(embed)│  │(graph)  │ │
│  └───┬───┘  └────┬────┘ │
└──────┼───────────┼──────┘
       │           │
┌──────▼───────────▼──────┐
│   PostgreSQL 16         │
│                         │
│  pgvector  → Embeddings │
│  Apache AGE → Knowledge │
│  pg_partman → Partitions│
│  pg_cron    → Automation│
└─────────────────────────┘
```

| Component | Role |
|-----------|------|
| **pgvector** | 1536-dim HNSW vector search for semantic memory recall |
| **Apache AGE** | Cypher-based knowledge graph for entity relationships |
| **pg_partman** | Monthly time-range partitioning with auto-maintenance |
| **pg_cron** | Nightly partition creation / retention cleanup |
| **Mem0** | Embedding generation + entity extraction via OpenAI |
| **MCP** | Tool protocol for agent integration via Streamable HTTP or stdio |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- An OpenAI API key (for embeddings + entity extraction)

### 1. Configure

```bash
cp .env.example .env
# Edit .env — set your OPENAI_API_KEY and a strong POSTGRES_PASSWORD
```

### 2. Start

```bash
docker compose up -d
```

This starts two containers:
- `open_brain_postgres` — Custom Postgres 16 with AGE + pgvector + pg_partman + pg_cron
- `open_brain_mcp` — Python MCP server exposing the memory tools

### 3. Verify

```bash
# Check both services are running
docker compose ps

# Verify database extensions
docker compose exec open-brain-db psql -U brain_user -d open_brain \
  -c "SELECT extname FROM pg_extension;"
# Expected: vector, pg_partman, pg_cron, age

# Verify partitions
docker compose exec open-brain-db psql -U brain_user -d open_brain \
  -c "SELECT tablename FROM pg_tables WHERE schemaname = 'memory_store';"
```

---

## MCP Tools

### `capture_thought`

Saves a new memory into the user's brain. Call this when the user makes a decision, meets someone, or specifies a constraint.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thought` | `str` | required | The memory content to capture |
| `user_id` | `str` | `"default"` | Isolates memories per user |

### `search_brain`

Searches past memories via vector similarity and knowledge graph. Call this BEFORE answering questions about the user's projects or history.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language search query |
| `user_id` | `str` | `"default"` | Which user's brain to search |

---

## Configuration

All configuration is via environment variables (`.env` file):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `POSTGRES_DB` | No | `open_brain` | Database name |
| `POSTGRES_USER` | No | `brain_user` | Database user |
| `POSTGRES_PASSWORD` | **Yes** | — | Database password |
| `POSTGRES_HOST` | No | `localhost` | DB host (set to `open-brain-db` in Docker) |
| `POSTGRES_PORT` | No | `5432` | DB port |
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API key for Mem0 |
| `MEMORY_RETENTION_MONTHS` | No | `12` | Partition retention window |
| `LOG_LEVEL` | No | `INFO` | Python log level |

---

## IDE Integration

### Network (Streamable HTTP — recommended)

When running via Docker, the MCP server exposes a Streamable HTTP endpoint. Any MCP-compatible IDE on the network can connect:

```json
{
  "mcpServers": {
    "open-brain": {
      "serverUrl": "http://localhost:8000/mcp"
    }
  }
}
```

> Replace `localhost` with your server's IP for remote/homelab access.

### Local (stdio)

For direct local integration (e.g., Claude Desktop), pipe into the Docker container:

```json
{
  "mcpServers": {
    "open-brain": {
      "command": "docker",
      "args": ["exec", "-i", "-e", "MCP_TRANSPORT=stdio", "open_brain_mcp", "python3", "open_brain_mcp.py"]
    }
  }
}
```

---

## Development

### Local Setup (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start Postgres with extensions (requires manual setup or Docker DB only)
docker compose up open-brain-db -d

# Run the MCP server locally
python open_brain_mcp.py
```

### Running Tests

```bash
pip install pytest pytest-mock
python -m pytest tests/ -v
```

---

## Project Structure

```
openBrain/
├── age_provider.py       # Apache AGE graph provider (Mem0 BaseGraphProvider)
├── open_brain_mcp.py     # MCP server with capture_thought / search_brain tools
├── init.sql              # Database schema, partitioning, indexes, cron
├── Dockerfile            # Custom Postgres 16 + AGE + pgvector + partman + cron
├── Dockerfile.mcp        # Python container for the MCP server
├── docker-compose.yml    # Two-service orchestration (DB + MCP)
├── mcp.json              # MCP server config reference for IDE integration
├── requirements.txt      # Pinned Python dependencies
├── .env.example          # Environment variable template
└── tests/                # Unit and integration tests
```

---

## License

MIT
