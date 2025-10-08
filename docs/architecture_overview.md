# HomeAI Architecture Overview

This document summarises the runtime building blocks that make up HomeAI and how they
collaborate during a typical chat session.

## High-Level Flow

1. **Gradio UI (`homeai_app.py`)** bootstraps dependencies through
   `build_dependencies()` and wires the chat interface, preview panel, persona controls, and
   event log.
2. **Tool Registry** exposes browse/read/summarise/locate helpers backed by the hardened
   filesystem APIs.
3. **Model Engine** (`LocalModelEngine`) translates chat transcripts into requests against the
   local model host and captures detailed telemetry for success and failure paths.
4. **Memory Backends** persist each conversation turn and provide retrieval utilities used by the
   context builder to compose prompts on subsequent interactions.

The diagram below highlights the primary relationships:

```
User ↔ Gradio UI ↔ App Handlers
                     │
                     ├── Tool Registry ──▶ Filesystem utilities
                     │
                     ├── Model Engine ───▶ Local model HTTP API
                     │
                     └── Context Builder ─▶ Memory Backend (JSON or Postgres)
```

## Module Responsibilities

### Gradio Application (`homeai_app.py`)

* Declares dependency factories so tests and alternate deployments can inject custom engines,
  memory stores, or tool registries.
* Normalises tool outputs and handles persona metadata so every conversation begins with the
  allowlist and protocol hints.
* Maintains an event log summarising UI actions and model/tool results.

### Filesystem Utilities (`homeai/filesystem.py`)

* Enforce sandboxing via the configured project root and provide safe read/list/locate helpers.
* Surface errors with descriptive messages that bubble back to the UI for display.

### Model Engine (`homeai/model_engine.py`)

* Wraps the local HTTP API, retrying against `/api/generate` when `/api/chat` is unavailable.
* Captures structured metadata (endpoint, status code, request payload, elapsed time) to support
  debugging and observability.

### Memory and Context (`context_memory.py`)

* `LocalJSONMemoryBackend` stores histories on disk with atomic writes and quarantine for
  corrupted files.
* `PgMemoryRepo` provides an optional Postgres implementation used when `HOMEAI_STORAGE=pg`.
* `ContextBuilder` assembles persona + history + retrieval snippets, handling token budgeting and
  duplicate suppression for the most recent user prompt.

## Operational Considerations

* Configure the storage backend via `HOMEAI_STORAGE` (`json` by default, `pg` for Postgres).
* When running with Postgres, set `HOMEAI_PG_DSN` and optionally `HOMEAI_PG_SCHEMA`.
* Code quality checks are automated via Ruff and Black (see the README for usage) and can be run
  locally with `pre-commit run --all-files`.

## Future Enhancements

* Introduce shared logging configuration so UI, engine, and storage components emit structured
  logs to the same sink.
* Extend the architecture diagram with sequence diagrams for tool execution and context building
  once additional integrations are added.
