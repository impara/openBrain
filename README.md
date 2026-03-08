# 🧠 OpenBrain

**Agentic memory framework with semantic recall + relational reasoning.**

This repository is intentionally **single-user**. The public API is one local brain by design: capture, ingest, and search all operate on the same memory store.

Give your AI agent a persistent memory that remembers *what* happened (vector search) and *why* it matters (knowledge graph) — all from a single Postgres instance, exposed as MCP tools.

For **multi-source ingestion** (ChatGPT/Claude/Antigravity exports, PC/phone notes, etc.), see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Architecture

```
┌─────────────────────────┐     ┌─────────────────────────┐
│     Claude / Agent      │     │     Telegram User       │
│  (via MCP protocol)     │     │  (via Telegram Bot API) │
└──────────┬──────────────┘     └──────────┬──────────────┘
           │ streamable-http (/mcp)         │ long-polling
           │ (or stdio for local)           │
┌──────────▼──────────────┐     ┌──────────▼──────────────┐
│   OpenBrain MCP Server  │     │   Telegram Bot          │
│   (open_brain_mcp.py)   │     │   (telegram_bot.py)     │
└──────────┬──────────────┘     └──────────┬──────────────┘
           │                               │
           └───────────┬───────────────────┘
                       │ brain_core.py
              ┌────────▼────────┐
              │  Mem0  │  AGE   │
              │(embed) │(graph) │
              └────┬───┴───┬───┘
                   │       │
              ┌────▼───────▼────┐
              │  PostgreSQL 16  │
              │                 │
              │ pgvector → Emb. │
              │ AGE → Knowledge │
              │ partman → Parts │
              │ pg_cron → Auto  │
              └─────────────────┘
```

| Component | Role |
|-----------|------|
| **pgvector** | 1536-dim HNSW vector search for semantic memory recall |
| **Apache AGE** | Cypher-based knowledge graph for entity relationships |
| **pg_partman** | Optional monthly partitioning for the example admin table `memory_store.memories` |
| **pg_cron** | Maintenance for that optional partitioned admin table |
| **Mem0** | Embedding generation + entity extraction via OpenAI |
| **MCP** | Tool protocol for agent integration via Streamable HTTP or stdio |

Schema: main tables (`raw_captures`, `capture_jobs`) and key indexes are summarized in [ARCHITECTURE.md §6](ARCHITECTURE.md#6-schema-summary).

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

This starts three containers:
- `open_brain_postgres` — Custom Postgres 16 with AGE + pgvector + pg_partman + pg_cron
- `open_brain_mcp` — Python MCP server exposing the memory tools
- `open_brain_telegram` — Telegram bot for chat-based memory access

### 3. Verify

```bash
# Check both services are running
docker compose ps

# Verify database extensions
docker compose exec open-brain-db psql -U brain_user -d open_brain \
  -c "SELECT extname FROM pg_extension;"
# Expected: vector, pg_partman, pg_cron, age

# Verify optional admin-table partitions
docker compose exec open-brain-db psql -U brain_user -d open_brain \
  -c "SELECT tablename FROM pg_tables WHERE schemaname = 'memory_store';"
```

---

## MCP Tools

### `capture_thought`

Saves a new memory. **Everything is stored verbatim**, and OpenBrain automatically decides whether to also enrich it with fact extraction. This is the default path for normal use.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thought` | `str` | required | The memory content to capture |

### `ingest`

Flexible entry point for any channel (chat exports, notes, imports). **Everything is stored verbatim**, and OpenBrain automatically decides whether to also enrich it with fact extraction.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | required | Text to remember (one message, note, or merged chunk) |
| `source` | `str` | `"import"` | Origin: e.g. `chatgpt`, `claude`, `antigravity`, `notes_pc`, `notes_phone` |
| `external_id` | `str` \| `null` | `null` | Optional stable id for dedup (e.g. conversation or note id) |

### `search_brain`

Searches past memories via vector similarity and knowledge graph. Call this BEFORE answering questions about the user's projects or history.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language search query |

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
| `MCP_PORT` | No | `8000` | Host port mapped to MCP server container port 8000 |
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API key for Mem0 |
| `OPENBRAIN_EMBEDDING_MODEL` | No | — | If set, stored in Mem0 memory metadata for re-embedding or A/B model tracking (e.g. `text-embedding-3-small`) |
| `MCP_TRANSPORT` | No | `streamable-http` | MCP transport: `streamable-http` (default) or `stdio` for local IDE |
| `MEMORY_RETENTION_MONTHS` | No | `12` | Retention window for the optional `memory_store.memories` admin table; `0` disables retention |
| `OPENBRAIN_CAPTURE_MODE` | No | `async` | `async` queues `/remember` for background indexing, `sync` blocks until ingestion finishes |
| `OPENBRAIN_INGEST_WORKERS` | No | `1` | Number of in-process capture workers per service |
| `OPENBRAIN_INGEST_POLL_MS` | No | `250` | Poll interval for queued capture jobs |
| `LOG_LEVEL` | No | `INFO` | Python log level |
| `TELEGRAM_BOT_TOKEN` | No | — | Telegram bot token from @BotFather |
| `TELEGRAM_AUTO_CAPTURE` | No | `false` | Auto-save plain text messages as thoughts |

---

## MCP transport

The MCP server uses **streamable-http** by default (e.g. in Docker it listens on port 8000 at `/mcp`). To use **stdio** for local IDE integration (Cursor, Claude Desktop, etc.), set `MCP_TRANSPORT=stdio` and run the server so the IDE can spawn it and talk over stdin/stdout. See [IDE Integration](#ide-integration) below for config examples.

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

For local IDE or Claude Desktop, run the server with stdio so the client spawns it. Example: Docker container with `MCP_TRANSPORT=stdio`:

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

## Telegram Bot

Chat with your brain from Telegram using `/remember` and `/search` commands.

### Setup

1. Message [@BotFather](https://t.me/BotFather) on Telegram → `/newbot` → copy the token
2. Add to your `.env`:
   ```env
   TELEGRAM_BOT_TOKEN=your-token-here
   ```
3. `docker compose up -d` — the bot container starts automatically

### Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message + usage |
| `/remember <text>` | Save anything; OpenBrain auto-detects enrichment |
| `/search <query>` | Search your memories |
| `/help` | Show available commands |

This repository runs one local brain. Telegram messages go into that same single-user store.
In the default `OPENBRAIN_CAPTURE_MODE=async`, `/remember` acknowledges immediately after durable queueing and finishes semantic indexing in the background.

Set `TELEGRAM_AUTO_CAPTURE=true` in `.env` to auto-save all plain text messages.

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
├── brain_core.py          # Shared memory layer (Mem0 + AGE init, capture/search)
├── age_provider.py       # Apache AGE graph provider (Mem0 BaseGraphProvider)
├── open_brain_mcp.py     # MCP server wrapping brain_core tools
├── telegram_bot.py       # Telegram bot client (/remember, /search)
├── init.sql              # Database schema, partitioning, indexes, cron
├── Dockerfile            # Custom Postgres 16 + AGE + pgvector + partman + cron
├── Dockerfile.mcp        # Python container for the MCP server
├── Dockerfile.telegram   # Python container for the Telegram bot
├── docker-compose.yml    # Three-service orchestration (DB + MCP + Telegram)
├── mcp.json              # MCP server config reference for IDE integration
├── requirements.txt      # Pinned Python dependencies
├── .env.example          # Environment variable template
├── connectors/           # Example: ingest_chat_export.py for JSON chat exports
└── tests/                # Unit and integration tests
```

---

## License

MIT
