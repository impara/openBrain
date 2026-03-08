# OpenBrain — Multi-Source Memory Architecture

This doc describes how to keep one unified “brain” while ingesting from many channels: chat exports (ChatGPT, Claude, Antigravity), PC notes, phone notes, MCP, Telegram, etc.

This repository runs as **one local brain**. Public entrypoints are single-user by design; the only `user_id` values left are internal fixed columns required by Postgres storage.

---

## 1. Design principles

- **One memory store**: All sources write into the same OpenBrain-owned vector + graph + `raw_captures` pipeline. Search is global across sources; you can still filter or label by source when needed.
- **Source as first-class field**: Every ingested item has a `source` (e.g. `chatgpt`, `notes_phone`). It is stored in `raw_captures.source`, copied into `memory_chunks.source` and chunk metadata `origin`, and **shown in search results** so you can see where each memory came from.
- **Single ingest pipeline**: One code path writes to `raw_captures` and enqueues a job; workers run the same ingestion logic with an automatic default that can still be explicitly forced to enriched or raw-only modes.
- **Provider-agnostic runtime**: Extraction and embeddings are selected deployment-wide through adapters for OpenAI, OpenRouter, or Ollama.
- **Connectors stay outside core**: Format-specific logic (e.g. “parse ChatGPT export JSON”) lives in small adapters or scripts that call the core ingest API. Core stays format-agnostic.

---

## 2. Recommended architecture (high level)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SOURCES (many)                                                          │
│  ChatGPT export │ Claude export │ Antigravity │ PC notes │ Phone notes   │
│  MCP capture_thought │ Telegram /remember │ import                    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CORE INGEST API (brain_core)                                             │
│  ingest(content, source, external_id=None)                                │
│  → raw_captures + capture_jobs (same queue as today)                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  EXISTING WORKERS                                                         │
│  Poll capture_jobs → chunk → embed → store → extract facts/graph         │
│  → OpenBrain-owned runtime pipeline                                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  UNIFIED MEMORY                                                           │
│  memory_chunks + AGE graph + raw_captures                                │
│  search_brain(query) searches everything                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Sources**: Any app or script that has “something to remember” (chat logs, notes, CLI).
- **Core ingest API**: Single entry point: `ingest(..., source=..., external_id=...)`. `capture_thought` and imports both use the automatic classifier.
- **Workers**: Unchanged structurally. They consume `capture_jobs` and apply the stored ingest strategy (`auto`, forced enrich, or raw-only).
- **Unified memory**: One search surface; optional UI or filters by `source` later.

---

## 3. Source taxonomy

Use a small, stable set of source identifiers so search and analytics stay consistent:

| Source ID        | Typical use                           |
|------------------|---------------------------------------|
| `mcp`            | MCP `capture_thought`                 |
| `telegram`       | Telegram `/remember`                  |
| `chatgpt`        | ChatGPT conversation export           |
| `claude`         | Claude conversation export            |
| `antigravity`    | Antigravity chats                     |
| `notes_pc`       | Notes from PC (folder/file sync)      |
| `notes_phone`    | Notes from phone (sync or paste)      |
| `import`         | Generic one-off import / manual paste |

You can add more (e.g. `obsidian`, `notion`) without changing core; just pass the new `source` and optionally `external_id` for dedup.

---

## 3.1. Unified ingest: automatic enrichment selection

You don’t have to choose “will this be stored?” — **everything you send is stored verbatim**. The only choice is whether we **also** enrich it.

| Ingest mode | What happens |
|-------------|----------------|
| **Any** | Your content is **always** stored as one raw memory (exact recall). So commands, notes, and prose are all retrievable. |
| **auto** | Default. Heuristics decide whether the input looks like a command/code/reference block (`snippet`) or normal prose/notes (`personal`). |
| **personal** | Raw + **enrichment**: we also run LLM fact extraction and graph extraction. You get the verbatim memory plus extra semantic/graph memories for better search. |
| **snippet** | Raw only. No extra extraction. Use when you want **just** exact recall and no duplicate “fact” memories. |

So: **one rule** — we always keep what you sent. The public API always enters through the automatic classifier; `personal` and `snippet` are internal pipeline outcomes, not normal user-facing choices.

---

## 4. Retrieval

Search is **hybrid** and **single-user**:

- **Hybrid**: Results come from three paths — **vector** (`memory_store.memory_chunks` via pgvector), **keyword** (raw_captures `LIKE` on content), and **graph** (Apache AGE relations). Scores combine vector similarity, token overlap, source weight, and recency; candidates are deduplicated and ranked.
- **Single-brain scope**: All paths run against one local brain.
- **Score threshold**: Candidates below a fixed relevance threshold (configurable in code) are dropped so only confident evidence is returned.
- **Manual inspection**: Use `search_brain(query, debug=True)` (or the MCP `search_brain_debug` tool) to see candidates, scores, dropped reasons, and the final ranking. This supports tuning and "don't chunk without testing retrieval."

---

## 5. Chunking

Chunking is **semantic-ish** and differs by path:

- **Primary**: OpenBrain chunks by sentence boundaries with a max chunk size, stores raw chunks as vectors, and stores extracted fact bullets as additional vector memories.
- **Fallback**: If structured extraction fails, the system falls back to heuristic bullets rather than dropping enrichment entirely.
- **Retrieval**: When changing chunk sizes or strategies, **test retrieval** (e.g. golden queries or `search_brain_debug`) so you can verify that expected memories appear in results.

---

## 6. Schema (summary)

Main application tables:

| Table | Purpose |
|-------|--------|
| `memory_store.raw_captures` | Verbatim ingested content; columns include the internal fixed brain id (`user_id='default'`), `source`, `content`, `content_len`, `ingest_strategy`, `external_id`, `created_at`. |
| `memory_store.capture_jobs` | Queue for async ingestion; links to `raw_captures`, has `status`, `attempt_count`, `available_at`, `last_error`. |
| `memory_store.memory_chunks` | Runtime vector storage for raw chunks and extracted facts; owned by OpenBrain and created at bootstrap using the configured embedding dimensions. |

Key indexes: `idx_raw_captures_user_created`, `idx_raw_captures_user_source_external_id` (partial, for dedup), `idx_capture_jobs_status_available_created`, and the HNSW index on `memory_store.memory_chunks.embedding`. `memory_store.memories` remains an optional parallel/admin table.

**At scale (vector index):** For large volumes, latency and recall depend primarily on the `memory_store.memory_chunks` pgvector path owned by the runtime. The `memory_store.memories_template` HNSW index in `init.sql` applies only to the optional parallel/admin table, not the main runtime memory path.

---


## 7. How to get data in (export → ingest)

ChatGPT, Claude, Antigravity, and most chat apps **do not give OpenBrain (or MCP) direct API access** to pull your conversations. So the flow is always: **get the data out of the source** → **send it into OpenBrain**. The “get out” step is manual or tool-assisted; the “send in” step can be MCP or a script.

| Question | Answer |
|----------|--------|
| **Can MCP “fetch” from ChatGPT?** | No. MCP only exposes tools that *receive* content (e.g. `ingest`). Something else must get the data from ChatGPT first. |
| **Do I have to manually export from ChatGPT?** | Yes, in practice. You use ChatGPT’s export (e.g. data export, copy-paste) or a browser extension. Then you *import* that into OpenBrain (see below). |
| **Where does “import” happen?** | Either in Cursor via MCP, or in a connector script that calls the ingest API. |

### Option A: Manual export → import in Cursor (MCP)

1. **Export** from ChatGPT (or Claude, etc.): use their export feature or copy the conversation.
2. **In Cursor**: paste the content (or attach the export file) and ask the agent to send it to your brain, e.g. *“Ingest this ChatGPT conversation into my brain with source=chatgpt.”*
3. The agent uses the **`ingest`** MCP tool with `content=...`, `source="chatgpt"`, and optionally `external_id=` (e.g. conversation id) so re-imports don’t duplicate.

So: **manual export**, **MCP-based import** (you don’t leave Cursor to run a script).

### Option B: Export file + connector script (semi-automated)

1. **Export** from ChatGPT/Claude (e.g. download JSON or markdown).
2. Run a **connector script** (e.g. in `connectors/`) that:
   - Reads the export file,
   - Parses conversations/messages,
   - Calls `ingest(content=..., source="chatgpt", external_id=...)` for each item (via HTTP to the MCP server or by importing `brain_core.ingest`).

You still trigger the export (or use a scheduled export if the product supports it); the script does the repetitive “import into OpenBrain” work.

### Option C: Fully manual

Copy-paste into Telegram (`/remember` or a future “paste to ingest” flow) or into Cursor and use `capture_thought` / `ingest` by hand. No connector, no automation.

### Summary

- **Getting data *out* of ChatGPT/Claude/notes**: manual (export, copy-paste) or your own automation (e.g. browser extension, sync tool). OpenBrain doesn’t fetch from those services.
- **Getting data *into* OpenBrain**: use the **MCP `ingest` tool** from Cursor (or Telegram, or any MCP client), or a **connector script** that calls the same ingest API. So you’re not doing “manual DB work” — you’re either asking the agent to ingest (Option A) or running a small script (Option B).

---

## 8. Connectors (outside core)

Each channel is a small adapter that:

1. **Reads** content in its native format (JSON export, markdown file, API response).
2. **Normalizes** to plain text (or a list of messages → one blob per message/turn).
3. **Calls** the core ingest API (or MCP tool) once per logical “item” (e.g. one message, one note, one merged conversation).

Examples:

- **ChatGPT/Claude/Antigravity**: Script that parses export file → for each conversation or turn, calls `ingest(text, source="chatgpt"|"claude"|"antigravity", external_id=conversation_id)` so the same export re-run doesn’t duplicate.
- **PC notes**: Folder watcher or cron job that reads new/changed files → `ingest(file_content, source="notes_pc", external_id=file_path_or_hash)`.
- **Phone notes**: Sync app or manual “paste from phone” → `ingest(content, source="notes_phone", external_id=note_id)`.

Connectors can live in the same repo under `connectors/` or in separate repos; they only need network/stdio access to the MCP server or a small HTTP wrapper around `ingest`.

---

## 9. Deduplication (optional)

To avoid re-ingesting the same conversation or note:

- Store an **external_id** per item (e.g. conversation ID, file path hash, note ID).
- The core API uses one atomic insert-or-reuse path keyed by `(user_id, source, external_id)` so concurrent re-imports skip cleanly instead of racing.
- The unique partial index on `(user_id, source, external_id)` where `external_id IS NOT NULL` keeps that dedup safe and fast.

---

## 10. Current implementation details

- **Core**: `ingest(content, source, external_id=None)` writes to `raw_captures`, dedups atomically on `(user_id, source, external_id)`, and enqueues the background job.
- **Wrappers**: `capture_thought` is the normal interactive path; the classifier decides whether each capture stays raw-only or gets extra enrichment.
- **Schema**: `external_id TEXT` plus the unique partial index on `(user_id, source, external_id)` keeps re-imports safe and idempotent.
- **Connectors**: An example lives in **`connectors/`**: `ingest_chat_export.py` reads a JSON file (array of `{content, external_id?}`) and calls `ingest()` for each item. See `connectors/README.md`. You can run it after exporting from ChatGPT/Claude and reshaping to that JSON format.

This keeps the project flexible for chat conversations and notes across ChatGPT, Claude, Antigravity, PC, and phone, without turning the core into a format-specific monolith.
