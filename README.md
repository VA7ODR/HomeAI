# HomeAI Canvas — Local Chat, Files, and Memory

HomeAI is a local-first chat assistant that runs on your machine using **Gradio** for UI, a configurable **local model host** (default model tag: `gpt-oss:20b`), and an optional **PostgreSQL** backend for durable chat history, memories, and retrieval (FTS + pgvector). Optional tools include file browsing/reading and image generation via local Fooocus. See [`docs/postgresql_setup.md`](docs/postgresql_setup.md) for detailed PostgreSQL setup instructions.

## Features

- **Local chat UI** (Gradio Blocks/Chatbot)
- **Engines**: Local model host with `/api/chat` (fallback `/api/generate`), model tag configurable
- **Tools**: `browse` (list directory), `read` (preview text), `locate` (recursive file search), `summarize` (LLM summary of a file)
- **Logging**: per-turn LLM/tool meta (endpoint, status, latency, request/response)
- **Memory** (JSON/optional Postgres): disk-backed message store with retrieval-ready context builder
  - Defaults to a local JSON backend (`~/.homeai/memory`) for quick-start setups
  - Optional Postgres backend for shared persistence (install via `pip install -e .[postgres]`); falls back to the JSON store if dependencies are missing
  - Uses standard-library `timezone.utc` timestamps so the fallback backend works on Python 3.10+
- **Agentic-ready**: structured outputs/tool calling loop (plan → tool → observe → continue)

---

## Requirements

- **OS**: Ubuntu 24.04 (or similar Linux/macOS)
- **Python**: 3.10+
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM recommended for `gpt-oss:20b`)
- **Local model host**: expose `/api/chat` (and optionally `/api/generate`)
- **PostgreSQL** (optional): 16+ with `pgvector` and `pg_trgm`
- **psql client tools**: e.g., `postgresql-client` so the bootstrap script can run
- **Fooocus** (optional): running locally via its UI or a small REST shim

---

## Quick Start

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# If you plan to use the PostgreSQL backend, install its optional extras instead:
# pip install -e .[postgres]
```

### 2) Model host

```bash
# Install or configure your preferred local model host
# Ensure the host serves POST /api/chat and optionally /api/generate
# Example: pull or load the `gpt-oss:20b` model tag
```

Environment knobs for the app:

```bash
export HOMEAI_MODEL_HOST=http://127.0.0.1:11434
export HOMEAI_MODEL_NAME=gpt-oss:20b
# Allowlist base directory (sandbox for tools)
export HOMEAI_ALLOWLIST_BASE="$HOME"
```

### 3) PostgreSQL + extensions (optional)

See [`docs/postgresql_setup.md`](docs/postgresql_setup.md) for a detailed walkthrough of preparing PostgreSQL on Ubuntu, including a reusable bootstrap script. Minimal outline:

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib postgresql-16-pgvector
sudo -u postgres psql
# If your distro separates the CLI tools, also install the `postgresql-client` package for `psql`.
```

In `psql`:

```sql
CREATE ROLE homeai_user LOGIN PASSWORD 'change-me';
CREATE DATABASE homeai_db OWNER homeai_user;
\c homeai_db
-- Optional: install extensions that improve FTS matching for other workloads
-- (HomeAI currently relies on lexical search only.)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

The app stores messages in a single `homeai_memory` table. The bootstrap script will create it automatically with sensible indexes so you do not need to run any additional DDL.

To run the bootstrapper with explicit credentials in one line (non-interactive shells):

```bash
POSTGRES_SUPERUSER=postgres \
POSTGRES_SUPERUSER_PASSWORD='your-postgres-password' \
HOMEAI_DB_PASSWORD='your-app-password' \
python scripts/bootstrap_postgres.py
```

DB connection env:

```bash
export HOMEAI_PG_DSN=postgresql://homeai_user:change-me@127.0.0.1:5432/homeai_db
# or via Unix socket:
# export HOMEAI_PG_DSN=postgresql://homeai_user:change-me@/homeai_db?host=/var/run/postgresql
```

### Semantic retrieval with pgvector (optional)

Stage 1 introduces a dedicated vector store that captures repository files and
chat history.  Apply the migration in [`migrations/`](migrations/) after
provisioning the database:

```bash
psql "$HOMEAI_PG_DSN" -f migrations/001_create_vector_store.sql
```

Key environment variables:

```bash
# Select the embedding model and dimension used by the ingestion pipeline
export HOMEAI_EMBEDDING_MODEL="mini-lm-embedding"
export HOMEAI_EMBEDDING_DIMENSION=384
```

Use the same dimension when building ANN indexes.  The defaults assume a
384-dimensional sentence embedding model and configure IVFFlat indexes with a
balanced `lists=100` parameter (tweak in the migration if your hardware suits a
different trade-off).

Run a dedicated embedding model alongside your chat model so that the vector
store can index repository files and chat history.  The service uses the
`HOMEAI_MODEL_HOST` endpoint for both chat completions and embeddings, so make
sure Ollama has the configured embedding model available (defaults to
`mini-lm-embedding`).

```bash
ollama pull mini-lm-embedding
ollama run mini-lm-embedding --keep-alive 30m  # keep the embedder warm
```

If you change `HOMEAI_EMBEDDING_MODEL`, download and run the corresponding
Ollama model instead.  Chat completions continue to use
`HOMEAI_MODEL_NAME` (default `gpt-oss:20b`).

### 4) Run the app

```bash
# JSON (filesystem) memory backend
python homeai_app.py

# PostgreSQL memory backend (requires optional extras: `pip install -e .[postgres]`)
HOMEAI_PG_DSN=postgresql://homeai_user:change-me@127.0.0.1:5432/homeai_db \
HOMEAI_STORAGE=pg \
  python homeai_app.py

# UI opens at http://127.0.0.1:7860
```

---

## Architecture Overview

See [`docs/architecture_overview.md`](docs/architecture_overview.md) for a component-level guide
to the UI, tool adapters, model engine, and memory backends.

---

## Testing

Install the development extras and run the test suite from the project root:

```bash
pip install -e .[dev]
pytest
```

---

## Code Quality & Tooling

- Install [`pre-commit`](https://pre-commit.com) hooks to run Ruff and Black automatically:

  ```bash
  pip install pre-commit
  pre-commit install
  ```

- Run formatting and linting manually with:

  ```bash
  pre-commit run --all-files
  ```

---

## Usage

Type into the chat input:

- **Chat**: `what can you do?`
- **Browse**: `/browse ~` or `/browse Documents`
- **Read**: `/read ~/notes/todo.txt`
- **Summarize**: `/summarize ~/projects/README.md`
- **Locate**: `/locate invoice` or `/find photo_2024`

The right-hand **LLM / Tool Log** shows raw request/response meta for each turn.

---

## Memory & Retrieval

- Messages are stored in a **local JSON backend** by default (`~/.homeai/memory`).
- When `HOMEAI_PG_DSN` is configured (and `HOMEAI_STORAGE=pg`) you can swap in a Postgres-backed backend that mirrors the JSON store’s behaviour while keeping data in Postgres.
- Both backends currently perform lexical ranking. Semantic recall is stubbed in the code and will surface once embeddings are integrated.

**Hybrid retrieval** (included in the canvas) builds a compact context each turn:

1. Recent N messages (conversation-scoped)
2. Top-K FTS matches (lexical fallback when using the JSON backend)
3. Top-K vector matches from messages + memories (currently reuses lexical scores as a placeholder)
4. Deduplicate, rank, and trim to token budget
5. Send to `/api/chat` with your model tag

---

## Configuration

Environment variables (common):

```
HOMEAI_MODEL_HOST=http://127.0.0.1:11434
HOMEAI_MODEL_NAME=gpt-oss:20b
HOMEAI_ALLOWLIST_BASE=/home/youruser
HOMEAI_PG_DSN=postgresql://homeai_user:change-me@127.0.0.1:5432/homeai_db
```

Optional:

```
# Persona seed (first system message)
HOMEAI_PERSONA="You are a single consistent assistant persona named 'Dax'..."

# Conversation context shaping
# (all values are integers; leave unset to keep defaults)
HOMEAI_CONTEXT_RECENT_LIMIT=128          # 0 means "include entire stored chat"
HOMEAI_CONTEXT_TOKEN_BUDGET=20000        # approximate total tokens allowed in prompt
HOMEAI_CONTEXT_RESERVE_FOR_RESPONSE=1800 # tokens to save for the model reply
HOMEAI_CONTEXT_FTS_LIMIT=10              # retrieved keyword matches per turn
HOMEAI_CONTEXT_VECTOR_LIMIT=10           # retrieved semantic matches per turn
HOMEAI_CONTEXT_MEMORY_LIMIT=8            # durable memory snippets per turn

# Vector store ingestion (pgvector)
HOMEAI_VECTOR_AUTO_INGEST=1              # set to 0/false to skip auto-scan on startup
HOMEAI_VECTOR_INGEST_PATHS=/path/one:/path/two  # optional override for allowlisted roots

```

The context-related variables control how much text we feed to the model every
turn. They tune **counts of messages/snippets** and the **token budget** used by
`ContextBuilder`. Higher values mean more history, but they are ultimately
limited by the model host and hardware.

| Variable | What it measures | Default | Notes |
| --- | --- | --- | --- |
| `HOMEAI_CONTEXT_RECENT_LIMIT` | Maximum number of stored chat messages to fetch from disk each turn. Set to `0` to replay the entire conversation before trimming by token budget. | 128 | Raising this only helps if the token budget is large enough to keep the extra turns once trimming runs. |
| `HOMEAI_CONTEXT_TOKEN_BUDGET` | Approximate **total** tokens we allow in the prompt (system + history + retrieved snippets + current user message). | 20000 | We estimate tokens with a word-count heuristic (`context_memory._rough_tokens`). Large markdown/code blocks quickly consume this budget, so long answers push older turns out even if `recent_limit` is high. |
| `HOMEAI_CONTEXT_RESERVE_FOR_RESPONSE` | Portion of the token budget we hold back so the model has space to reply. | 1800 | The effective context window is `token_budget - reserve`. If responses are truncated, lower this number; if history is trimmed too aggressively, increase `token_budget`. |
| `HOMEAI_CONTEXT_FTS_LIMIT` | Number of keyword (full-text search) matches to pull in as additional references. | 10 | Applies to both storage backends; the JSON backend performs a simple lexical search. |
| `HOMEAI_CONTEXT_VECTOR_LIMIT` | Number of semantic (vector) matches to include when embeddings are enabled. | 10 | Currently reuses lexical scores until embeddings land. |
| `HOMEAI_CONTEXT_MEMORY_LIMIT` | Durable memory snippets to surface each turn (notes, summaries, etc.). | 8 | These appear as an extra system message with bullet points once memories are implemented. |

> **Why deleting long messages helps:** the budget trimming step removes the
> oldest non-system messages until the heuristic token estimate fits within
> `token_budget - reserve_for_response`. Large answers with code fences or
> markdown are “expensive” in tokens, so removing them frees enough room for the
> builder to keep earlier turns. Increasing `HOMEAI_CONTEXT_TOKEN_BUDGET` (and
> verifying the model host can accept that many tokens) is the lever for keeping
> those long responses *and* more chat history.

When the pgvector store is enabled (install extras + set `HOMEAI_PG_DSN`), the
app writes every chat turn to the `messages` table and, by default, pre-ingests
files under the allowlisted base into `doc_chunks` on startup. Disable the scan
with `HOMEAI_VECTOR_AUTO_INGEST=0` or supply specific roots via
`HOMEAI_VECTOR_INGEST_PATHS`.

---

## Architecture

- **Core** (no UI assumptions): tools (files), allowlist/path policy, engine adapter for the local model host, orchestrator, retrieval/storage layer.
- **Engines**: `LocalModelEngine` with `/api/chat` (fallback `/api/generate`), optional streaming later.
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
- **404 on chat endpoint**: the app falls back to `/api/generate`. Verify the model tag exists and the service is running.
- **Slow responses**: use 4-bit quant models, rely on the engine's shared HTTP session, lower the context window, or disable large file previews.
- **DB errors**: confirm extensions are installed in your **database** (not just server), and that the `homeai_memory` table exists.

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

### Copyright (c) 2025 James Baker VA7ODR

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

---
