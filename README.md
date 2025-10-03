# Local-only TBH Canvas (Ollama) — Chat, Files, and Memory

A local-first chat assistant that runs on your machine using **Gradio** for UI, **Ollama** for LLMs (default: `gpt-oss:20b`), and **PostgreSQL** for durable chat history, memories, and retrieval (FTS + pgvector). Optional tools include file browsing/reading and image generation via local Fooocus.

## Features

- **Local chat UI** (Gradio Blocks/Chatbot)
- **Engines**: Ollama (`/api/chat`, fallback `/api/generate`), model tag configurable
- **Tools**: `browse` (list directory), `read` (preview text), `locate` (recursive file search), `summarize` (LLM summary of a file)
- **Logging**: per-turn LLM/tool meta (endpoint, status, latency, request/response)
- **Memory** (Postgres): messages and long-term memories in JSONB, with
  - **Full-text search** (GIN on `tsvector`)
  - **Semantic search** (pgvector HNSW)
- **Agentic-ready**: structured outputs/tool calling loop (plan → tool → observe → continue)

---

## Requirements

- **OS**: Ubuntu 24.04 (or similar Linux/macOS)
- **Python**: 3.10+
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM recommended for `gpt-oss:20b`)
- **Ollama**: latest stable
- **PostgreSQL**: 16+ with `pgvector` and `pg_trgm`
- **Fooocus** (optional): running locally via its UI or a small REST shim

---

## Quick Start

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install gradio requests psycopg[binary] psycopg_pool
```

### 2) Ollama

```bash
# Install Ollama (see ollama.com)
ollama pull gpt-oss:20b
# Optional embedding model for memory
ollama pull nomic-embed-text
```

Environment knobs for the app:

```bash
export TBH_OLLAMA_HOST=http://127.0.0.1:11434
export TBH_OLLAMA_MODEL=gpt-oss:20b
# Allowlist base directory (sandbox for tools)
export TBH_ALLOWLIST_BASE="$HOME"
```

### 3) PostgreSQL + extensions

See the project canvas **“Postgres Memory Schema & Ubuntu 24.04 Setup”** for full DDL and OS install notes. Minimal outline:

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib postgresql-16-pgvector
sudo -u postgres psql
```

In `psql`:

```sql
CREATE ROLE tbh_user LOGIN PASSWORD 'change-me';
CREATE DATABASE tbh_db OWNER tbh_user;
\c tbh_db
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

Apply the DDL from the canvas to create:

- `conversations`, `messages`, `memories`
- GIN FTS indexes and HNSW vector indexes

DB connection env:

```bash
export TBH_DB_DSN=postgresql://tbh_user:change-me@127.0.0.1:5432/tbh_db
# or via Unix socket:
# export TBH_DB_DSN=postgresql://tbh_user:change-me@/tbh_db?host=/var/run/postgresql
```

### 4) Run the app

```bash
python gradio_tbh_canvas.py
# UI opens at http://127.0.0.1:7860
```

---

## Usage

Type into the chat input:

- **Chat**: `what can you do?`
- **Browse**: `browse ~` or `browse Documents`
- **Read**: `read ~/notes/todo.txt`
- **Summarize**: `summarize ~/projects/README.md`
- **Locate**: `locate invoice` or `find photo_2024`

The right-hand **LLM / Tool Log** shows raw request/response meta for each turn.

---

## Memory & Retrieval (Postgres)

- Messages and memories are stored as **JSONB**, searchable via:
  - **FTS** (`tsvector` + GIN) for keywords/paths
  - **pgvector HNSW** for semantic recall
- Embeddings are generated locally via Ollama’s `/api/embeddings` (e.g., `nomic-embed-text`) and updated asynchronously.

**Hybrid retrieval** (included in the canvas) builds a compact context each turn:

1. Recent N messages (conversation-scoped)
2. Top-K FTS matches
3. Top-K vector matches from messages + memories
4. Deduplicate, rank, and trim to token budget
5. Send to `/api/chat` with your model tag

---

## Configuration

Environment variables (common):

```
TBH_OLLAMA_HOST=http://127.0.0.1:11434
TBH_OLLAMA_MODEL=gpt-oss:20b
TBH_ALLOWLIST_BASE=/home/youruser
TBH_DB_DSN=postgresql://tbh_user:change-me@127.0.0.1:5432/tbh_db
```

Optional:

```
# Persona seed (first system message)
TBH_PERSONA="You are a single consistent assistant persona named 'Dax'..."
```

---

## Architecture

- **Core** (no UI assumptions): tools (files), allowlist/path policy, engine adapter (Ollama), orchestrator, retrieval/storage layer.
- **Engines**: `OllamaEngine` with `/api/chat` (fallback `/api/generate`), optional streaming later.
- **UI**: Gradio Blocks (chat widget, preview pane, log box).
- **Tools bus**: local-only helpers (file ops now; Fooocus, scheduler, etc. later) behind simple JSON schemas.

---

## Fooocus Integration (optional)

Treat Fooocus as a local tool:

- A small REST wrapper that accepts `{prompt, params}` and returns `{image_path, meta}`.
- The assistant calls the tool; the UI previews the returned image path.
- Store prompt/meta in `messages.content` (JSONB); optionally add a JSONB GIN index for common keys.

---

## Troubleshooting

- **“Host not allowed”**: paths or URLs outside the allowlist/whitelist are blocked by design.
- **Ollama 404 on **``: the app falls back to `/api/generate`. Verify the model tag exists (`ollama list`) and service is running.
- **Slow responses**: use 4-bit quant models, reuse a single HTTP session in the engine (already implemented), lower context window, or disable large file previews.
- **DB errors**: confirm extensions are installed in your **database** (not just server), and that HNSW indexes were created for `embedding`.

---

## Development

- **Formatting/Linting**: consider `pre-commit` with Ruff.
- **Git ignore**: add a standard Python `.gitignore` (`.venv/`, `__pycache__/`, `.env`, etc.).
- **Event bindings**: all Gradio `.click/.submit/.change` calls must be **inside** the `with gr.Blocks(...):` context.

---

## Roadmap

- Add streaming responses (SSE) to the chat UI
- Tooling: scheduler/alarms, Fooocus wrapper, HTTP webhook with HMAC
- Multi-project support and cross-conversation retrieval policies
- Summarization jobs (rolling conversation summaries → `memories(kind='summary')`)

---

## License

Choose a license (MIT/Apache-2.0 recommended) before publishing the repo.

