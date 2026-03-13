# 🧠 OpenBrain

**Agentic memory framework with semantic recall + relational reasoning.**

This repository is intentionally **single-user**. The public API is one local brain by design: capture, ingest, and search all operate on the same memory store.

Give your AI agent a persistent memory that remembers *what* happened (vector search) and *why* it matters (knowledge graph) — all from a single Postgres instance, exposed as MCP tools.

The runtime is now **provider-agnostic**. OpenBrain owns its ingestion, embeddings, vector search, and graph extraction pipeline, and supports **OpenAI**, **OpenRouter**, and **Ollama** as first-class providers.

For **multi-source ingestion** (ChatGPT/Claude/Antigravity exports, PC/phone notes, etc.), see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Architecture Philosophy

OpenBrain keeps **semantic search and graph traversal in the same PostgreSQL instance** on purpose. The goal is to avoid the usual split-brain problems of a Postgres + Neo4j stack: duplicate identifiers, partial writes across two databases, stitched queries in application code, and double operational overhead.

The project originally leaned on Mem0 for extraction and graph logic. It now keeps that intelligence inside OpenBrain itself. The reason is control: retrieval quality, provider choice, storage semantics, and search behavior all turned out to be core product concerns rather than vendor details.

The design intent is:
- `raw_captures` preserves the original source text
- `memory_chunks` powers semantic recall over raw passages and extracted facts
- `managed_memories` stores active directives and preferences as coherent canonical records
- Apache AGE adds entity and relationship traversal
- the application owns the ingestion and ranking policy

That gives the project one database, one runtime truth, and one place to evolve retrieval behavior.

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
              ┌────────▼──────────────┐
              │  openbrain package    │
              │ providers │ repos │   │
              └────┬──────┴───────┬───┘
                   │              │
              ┌────▼──────────────▼────┐
              │  PostgreSQL 16  │
              │                 │
              │ pgvector → Chunks│
              │ AGE → Knowledge │
              └─────────────────┘
```

| Component | Role |
|-----------|------|
| **pgvector** | Runtime-owned HNSW vector search over `memory_store.memory_chunks` |
| **Managed memory** | Canonical active directives and preferences with semantic retrieval over `memory_store.managed_memories` |
| **Apache AGE** | Cypher-based knowledge graph for entity relationships |
| **OpenBrain runtime** | Chunking, embedding, fact extraction, ranking, and queue orchestration |
| **MCP** | Tool protocol for agent integration via Streamable HTTP or stdio |

Schema: main tables (`raw_captures`, `capture_jobs`) and key indexes are summarized in [ARCHITECTURE.md §7](ARCHITECTURE.md#7-schema-summary).

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- One provider setup for embeddings + extraction:
  - OpenAI
  - OpenRouter
  - Ollama

### 1. Configure

```bash
cp .env.example .env
# Edit .env — set provider env vars and a strong POSTGRES_PASSWORD
```

### 2. Start

```bash
docker compose up -d
```

This starts three containers:
- `open_brain_postgres` — Custom Postgres 16 with AGE + pgvector
- `open_brain_mcp` — Python MCP server exposing the memory tools
- `open_brain_telegram` — Telegram bot for chat-based memory access

### 3. Verify

```bash
# Check both services are running
docker compose ps

# Verify database extensions
docker compose exec open-brain-db psql -U brain_user -d open_brain \
  -c "SELECT extname FROM pg_extension;"
# Expected: vector, age

# Verify runtime tables
docker compose exec open-brain-db psql -U brain_user -d open_brain \
  -c "\dt memory_store.*"
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
| `managed_kind` | `str` \| `null` | `null` | Optional explicit managed-memory override: `directive` or `preference` |

### `search_brain`

Searches past memories via vector similarity and knowledge graph. Call this BEFORE answering questions about the user's projects or history.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language search query |

### `get_active_memories`

Returns active managed memories such as standing directives and preferences.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | `""` | Optional semantic filter over active managed memories |
| `kind` | `str` \| `null` | `null` | Optional kind filter: `directive` or `preference` |

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
| `MCP_PORT` | No | `8001` | Host port mapped to MCP server container port 8000. On Coolify, avoid `8000` because Coolify itself uses it. |
| `OPENBRAIN_LLM_PROVIDER` | No | `openai` | Structured-generation provider: `openai`, `openrouter`, or `ollama` |
| `OPENBRAIN_LLM_MODEL` | No | provider-specific | Model used for fact and graph extraction |
| `OPENBRAIN_LLM_BASE_URL` | No | provider-specific | Base URL for the configured LLM provider |
| `OPENBRAIN_LLM_API_KEY` | Sometimes | — | Required for `openai` and `openrouter`; not needed for local Ollama |
| `OPENBRAIN_EMBEDDING_PROVIDER` | No | same as LLM provider | Embedding provider: `openai`, `openrouter`, or `ollama` |
| `OPENBRAIN_EMBEDDING_MODEL` | No | provider-specific | Embedding model name |
| `OPENBRAIN_EMBEDDING_BASE_URL` | No | provider-specific | Base URL for embedding requests |
| `OPENBRAIN_EMBEDDING_API_KEY` | Sometimes | — | Required for `openai` and `openrouter`; not needed for local Ollama |
| `OPENBRAIN_EMBEDDING_DIMS` | **Yes** for `openrouter`/`ollama` | `1536` for `openai` | Embedding dimensionality used to create `memory_store.memory_chunks` |
| `OPENBRAIN_GRAPH_NAME` | No | `brain_graph_v2` | Apache AGE graph name used by the new runtime |
| `OPENBRAIN_OPENROUTER_SITE_URL` | No | — | Optional `HTTP-Referer` header for OpenRouter |
| `OPENBRAIN_OPENROUTER_APP_NAME` | No | — | Optional `X-Title` header for OpenRouter |
| `MCP_TRANSPORT` | No | `streamable-http` | MCP transport: `streamable-http` (default) or `stdio` for local IDE |
| `OPENBRAIN_CAPTURE_MODE` | No | `async` | `async` queues `/remember` for background indexing, `sync` blocks until ingestion finishes |
| `OPENBRAIN_INGEST_WORKERS` | No | `1` | Number of in-process capture workers per service |
| `OPENBRAIN_INGEST_POLL_MS` | No | `250` | Poll interval for queued capture jobs |
| `LOG_LEVEL` | No | `INFO` | Python log level |
| `TELEGRAM_BOT_TOKEN` | No | — | Telegram bot token from @BotFather |
| `TELEGRAM_AUTO_CAPTURE` | No | `false` | Auto-save plain text messages as thoughts |

---

## MCP transport

The MCP server uses **streamable-http** by default (e.g. in Docker it listens on container port 8000 at `/mcp`, mapped to host port `8001` by default). To use **stdio** for local IDE integration (Cursor, Claude Desktop, etc.), set `MCP_TRANSPORT=stdio` and run the server so the IDE can spawn it and talk over stdin/stdout. See [IDE Integration](#ide-integration) below for config examples.

---

## IDE Integration

### Network (Streamable HTTP — recommended)

When running via Docker, the MCP server exposes a Streamable HTTP endpoint. Any MCP-compatible IDE on the network can connect:

```json
{
  "mcpServers": {
    "open-brain": {
      "serverUrl": "http://localhost:8001/mcp"
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

Chat with your brain from Telegram using `/remember`, `/search`, and `/profile`.

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
| `/profile [query]` | Show active directives and preferences |
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
├── brain_core.py         # Stable public wrappers around the OpenBrain application
├── openbrain/           # Domain, application, and infrastructure packages
├── open_brain_mcp.py     # MCP server wrapping brain_core tools
├── telegram_bot.py       # Telegram bot client (/remember, /search)
├── init.sql              # Minimal database bootstrap for extensions and schema
├── Dockerfile            # Custom Postgres 16 + AGE + pgvector
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
