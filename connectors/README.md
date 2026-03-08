# Connectors

Small scripts that pull content from export files (or other sources) and send it into OpenBrain via `brain_core.ingest()`.

**Run any connector from the repo root** so `brain_core` and `.env` are available. Use the same Python environment as the rest of OpenBrain (e.g. activate the venv first):

```bash
cd /path/to/openBrain
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
python connectors/ingest_chat_export.py path/to/export.json --source chatgpt
```

---

## ingest_chat_export.py

Imports a JSON export of chat conversations (or any list of text items) into your brain.

### JSON format

Array of items. Each item can be:

- `{"content": "text to remember"}` — required
- `{"content": "...", "external_id": "unique-id"}` — optional; use for dedup (same id = skip on re-run)

Example `my_export.json`:

```json
[
  {"content": "User: What's the best way to deploy with Docker?\nAssistant: Use docker compose for multi-container apps...", "external_id": "chatgpt-conv-1"},
  {"content": "User: Remember I prefer TypeScript for new projects.\nAssistant: I'll remember that.", "external_id": "chatgpt-conv-2"}
]
```

If your export has a different shape (e.g. nested `messages`), either reshape it to this format or edit the script to parse your format.

### Usage

```bash
# From repo root; default source=import.
python connectors/ingest_chat_export.py my_export.json

# Set source for provenance (shown in search)
python connectors/ingest_chat_export.py my_export.json --source chatgpt

# Dry-run (print only, no ingest)
python connectors/ingest_chat_export.py my_export.json --source claude --dry-run
```

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | `import` | Origin label (e.g. `chatgpt`, `claude`, `notes_pc`) |
| `--dry-run` | off | Print what would be ingested, do not call ingest |

### Getting exports

- **ChatGPT**: Settings → Data controls → Export data (or copy-paste a conversation).
- **Claude**: Copy the conversation or use any export they provide.
- Reshape the export into the JSON format above (one object per conversation or per message), then run the script.
