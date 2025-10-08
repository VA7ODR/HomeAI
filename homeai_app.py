#!/usr/bin/env python3

# Copyright (c) 2025 James Baker VA7ODR
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import html
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import gradio as gr

from context_memory import ContextBuilder, LocalJSONMemoryBackend

import homeai.config as homeai_config
import homeai.filesystem as filesystem
from homeai.model_engine import LocalModelEngine
from homeai.pgvector_store import PgVectorStore, SupportsEmbed
from homeai.embeddings import OllamaEmbedder
from homeai.tool_utils import ToolRegistry, parse_structured_tool_call, parse_tool_call
from homeai.ui_utils import safe_component


homeai_config.reload_from_environment()

BASE = homeai_config.BASE
ALLOWLIST_LINE = homeai_config.ALLOWLIST_LINE
DEFAULT_PERSONA = homeai_config.DEFAULT_PERSONA
TOOL_PROTOCOL_HINT = homeai_config.TOOL_PROTOCOL_HINT

assert_in_allowlist = filesystem.assert_in_allowlist
get_file_info = filesystem.get_file_info
list_dir = filesystem.list_dir
locate_files = filesystem.locate_files
read_text_file = filesystem.read_text_file
resolve_under_base = filesystem.resolve_under_base

_safe_component = safe_component


SPOONS_FORM_ID = "spoons_checkin"
SPOONS_FORM_VERSION = 1
SPOONS_INSTRUCTION = (
    "[Form: spoons_checkin.v1]\n"
    "You are a pacing coach. Given energy, mood, and gravity, produce a brief pacing plan for today: "
    "3–5 concrete tasks, timeboxes, rest ratios using a “10% less than I think I can” rule, "
    "red/yellow/green activities, and a one-sentence reasoning. Keep it under 180 words and avoid medical claims."
)


_SPOONS_GRAVITY_LABELS = {
    0: "None",
    1: "Light",
    2: "Moderate",
    3: "Heavy",
}


def _spoons_gravity_label(value: int) -> str:
    return _SPOONS_GRAVITY_LABELS.get(value, str(value))


def _prepare_spoons_submission(
    *,
    energy: int,
    mood: int,
    gravity: int,
    must_dos: str,
    nice_tos: str,
    notes: str,
) -> Tuple[str, Dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "energy": int(energy),
        "mood": int(mood),
        "gravity": int(gravity),
        "must_dos": must_dos.strip(),
        "nice_tos": nice_tos.strip(),
        "notes": notes.strip(),
        "timestamp": timestamp,
    }
    gravity_label = _spoons_gravity_label(int(gravity))
    display_lines = [
        f"Energy Spoons: {payload['energy']}",
        f"Mood Spoons: {payload['mood']}",
        f"Gravity: {payload['gravity']} ({gravity_label})",
        f"Must Do's: {payload['must_dos'] or '—'}",
        f"Nice To's: {payload['nice_tos'] or '—'}",
        f"Other Notes: {payload['notes'] or '—'}",
    ]
    text = "\n".join(display_lines)
    metadata = {
        "form_id": SPOONS_FORM_ID,
        "form_version": SPOONS_FORM_VERSION,
        "form_payload": payload,
    }
    return text, metadata


SPOONS_DEFAULTS = (5, 5, 1, "", "", "")


def _reset_spoons_form_values() -> Tuple[Any, ...]:
    energy, mood, gravity, must_dos, nice_tos, notes = SPOONS_DEFAULTS
    return (
        gr.update(value=energy),
        gr.update(value=mood),
        gr.update(value=gravity),
        gr.update(value=must_dos),
        gr.update(value=nice_tos),
        gr.update(value=notes),
    )


class _NullAccordion:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_NullAccordion":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _component_factory(name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    return getattr(gr, name, fallback)


# Adapter: read → uses your hardened read_text_file()

def tool_read(path: str) -> Dict[str, Any]:
    r = read_text_file(path) # existing helper in your app
    if isinstance(r, dict) and r.get("error"):
        raise RuntimeError(r["error"])
    # r has: {path, text, truncated, ...}
    return {
        "path": r.get("path", path),
        "truncated": bool(r.get("truncated", False)),
        "text": r.get("text", ""),
    }

# Adapter: summarize → reads then summarizes using your summarize_text()

def tool_summarize(path: str) -> Dict[str, Any]:
    r = read_text_file(path)
    if isinstance(r, dict) and r.get("error"):
        raise RuntimeError(r["error"])
    raw_text = r.get("text", "")
    summary, meta = summarize_text(raw_text) # your existing summarizer
    return {
        "path": r.get("path", path),
        "summary": summary,
        "meta": meta if isinstance(meta, (str, dict, list)) else str(meta),
    }

# Adapter: locate → uses your locate_files(); expands to size/mtime via get_file_info()

def tool_locate(
    query: str | None = None,
    *,
    path: str | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Locate files matching ``query`` or ``path``.

    Some model backends emit ``path`` instead of ``query`` when invoking the
    locate tool.  Accept both to stay compatible with those structured tool
    calls while still supporting the original ``query`` keyword.  Any
    additional keyword arguments are ignored so that future parameters (or
    backend quirks) do not raise ``TypeError`` when routed through the tool
    registry.
    """

    if query is None:
        query = path or extra.pop("query", None)

    if not query:
        raise ValueError("locate requires a non-empty query or path")

    res = locate_files(query, start=str(BASE)) # existing helper in your app
    if isinstance(res, dict) and res.get("error"):
        raise RuntimeError(res["error"])
    results = []
    for rel in res.get("results", []):
        try:
            abs_path = assert_in_allowlist(Path(rel))
        except PermissionError:
            continue
        try:
            results.append(get_file_info(abs_path))
        except FileNotFoundError:
            continue
    return {
        "query": query,
        "count": len(results),
        "truncated": bool(res.get("truncated", False)),
        "results": results,
    }

def tool_browse(path: str = ".", pattern: str = "") -> Dict[str, Any]:
    res = list_dir(path, pattern)
    if isinstance(res, dict) and res.get("error"):
        raise RuntimeError(res["error"])
    return res


def _register_default_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register("browse", tool_browse)
    registry.register("read", tool_read)
    registry.register("summarize", tool_summarize)
    registry.register("locate", tool_locate)
    return registry


@dataclass
class AppDependencies:
    tool_registry: ToolRegistry
    engine: LocalModelEngine
    memory_backend: LocalJSONMemoryBackend
    context_builder: ContextBuilder
    vector_store: Optional[PgVectorStore]
    embedder: Optional[SupportsEmbed]


tool_registry: ToolRegistry
engine: LocalModelEngine
memory_backend: LocalJSONMemoryBackend
context_builder: ContextBuilder
vector_store: Optional[PgVectorStore] = None
embedding_provider: Optional[SupportsEmbed] = None
_dependencies: AppDependencies | None = None


def _context_builder_env_overrides() -> Dict[str, int]:
    """Collect ``ContextBuilder`` keyword overrides from environment variables.

    Each supported variable maps to a corresponding ``ContextBuilder`` argument.
    Invalid integers are ignored so that a typo cannot break the startup flow.
    """

    mapping = {
        "HOMEAI_CONTEXT_RECENT_LIMIT": "recent_limit",
        "HOMEAI_CONTEXT_FTS_LIMIT": "fts_limit",
        "HOMEAI_CONTEXT_VECTOR_LIMIT": "vector_limit",
        "HOMEAI_CONTEXT_MEMORY_LIMIT": "memory_limit",
        "HOMEAI_CONTEXT_TOKEN_BUDGET": "token_budget",
        "HOMEAI_CONTEXT_RESERVE_FOR_RESPONSE": "reserve_for_response",
    }
    overrides: Dict[str, int] = {}
    for env_var, kwarg in mapping.items():
        raw_value = os.getenv(env_var)
        if raw_value is None:
            continue
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        try:
            overrides[kwarg] = int(raw_value)
        except ValueError:
            # Silently ignore invalid overrides; defaults remain in effect.
            continue
    return overrides


def _build_memory_components(
    *, storage: str | None = None, overrides: Optional[Dict[str, int]] = None
) -> Tuple[LocalJSONMemoryBackend, ContextBuilder]:
    selected_storage = storage or os.getenv("HOMEAI_STORAGE")
    kwargs: Dict[str, Any] = {}
    if selected_storage:
        kwargs["storage"] = selected_storage
    backend = LocalJSONMemoryBackend(**kwargs)
    context_overrides = dict(_context_builder_env_overrides())
    if overrides:
        context_overrides.update({k: v for k, v in overrides.items() if v is not None})
    builder = ContextBuilder(backend, **context_overrides)
    return backend, builder


def _build_vector_components() -> Tuple[Optional[SupportsEmbed], Optional[PgVectorStore]]:
    log = logging.getLogger(__name__)
    embedder: Optional[SupportsEmbed]
    try:
        embedder = OllamaEmbedder(
            model_name=homeai_config.EMBEDDING_MODEL,
            host=homeai_config.HOST,
            dimension=homeai_config.EMBEDDING_DIMENSION,
        )
    except Exception as exc:
        log.warning("Failed to initialise Ollama embedder: %s", exc)
        embedder = None

    dsn = os.getenv("HOMEAI_PG_DSN")
    schema = os.getenv("HOMEAI_PG_SCHEMA")
    store: Optional[PgVectorStore]
    try:
        store = PgVectorStore(
            dsn,
            schema=schema,
            embedder=embedder,
            embedding_dimension=homeai_config.EMBEDDING_DIMENSION,
            embedding_model=homeai_config.EMBEDDING_MODEL,
        )
    except Exception as exc:
        log.warning("Failed to initialise vector store: %s", exc)
        store = None

    return embedder, store


def _auto_ingest_repository(
    store: Optional[PgVectorStore], embedder: Optional[SupportsEmbed]
) -> None:
    if store is None or embedder is None:
        return

    flag = os.getenv("HOMEAI_VECTOR_AUTO_INGEST")
    if flag is None:
        # Opt-in behaviour: only scan when the user explicitly enables it.
        return

    enabled = flag.strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return

    base_path = BASE
    if not base_path.exists():
        logging.getLogger(__name__).warning(
            "Skipping vector auto-ingest: base path does not exist: %s", base_path
        )
        return
    if not base_path.is_dir():
        logging.getLogger(__name__).warning(
            "Skipping vector auto-ingest: base path is not a directory: %s",
            base_path,
        )
        return

    paths = [str(base_path)]

    try:
        report = store.ingest_files(paths, source_kind="repo", embedder=embedder)
    except Exception as exc:
        logging.getLogger(__name__).warning("Vector ingest failed: %s", exc)
        return

    logging.getLogger(__name__).info("Vector ingest completed: %s", report.to_dict())


def build_dependencies(
    *,
    storage: str | None = None,
    registry: ToolRegistry | None = None,
    engine_factory: Optional[Callable[[], LocalModelEngine]] = None,
    context_overrides: Optional[Dict[str, int]] = None,
) -> AppDependencies:
    tool_reg = _register_default_tools(registry or ToolRegistry())
    engine_instance = engine_factory() if engine_factory else LocalModelEngine()
    embedder, store = _build_vector_components()
    backend, builder = _build_memory_components(storage=storage, overrides=context_overrides)
    backend.configure_vector_search(store, embedder=embedder)
    _auto_ingest_repository(store, embedder)
    return AppDependencies(
        tool_registry=tool_reg,
        engine=engine_instance,
        memory_backend=backend,
        context_builder=builder,
        vector_store=store,
        embedder=embedder,
    )


def get_dependencies() -> AppDependencies:
    if _dependencies is None:
        raise RuntimeError("App dependencies have not been configured")
    return _dependencies


def configure_dependencies(deps: AppDependencies) -> AppDependencies:
    global tool_registry, engine, memory_backend, context_builder, vector_store, embedding_provider, _dependencies
    _dependencies = deps
    tool_registry = deps.tool_registry
    engine = deps.engine
    memory_backend = deps.memory_backend
    context_builder = deps.context_builder
    vector_store = deps.vector_store
    embedding_provider = deps.embedder
    return deps


configure_dependencies(build_dependencies())


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, dict):
        for key in ("text", "display", "summary", "preview", "body"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)
    if content is None:
        return ""
    return str(content)


def _register_vector_message(conversation_id: str, message: Any) -> None:
    if vector_store is None or message is None:
        return

    text = _message_content_to_text(getattr(message, "content", {}))
    if not text.strip():
        return

    message_id = getattr(message, "id", None)
    role = getattr(message, "role", None)
    created = getattr(message, "created_at", None)
    updated = getattr(message, "created_at", None)

    if not message_id or not role:
        return

    try:
        vector_store.register_message(
            message_id=str(message_id),
            thread_id=str(conversation_id),
            role=str(role),
            content=text,
            created_at=created,
            updated_at=updated,
            embedder=embedding_provider,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to index message for vector search: %s", exc)


def _persist_message(
    conversation_id: str,
    role: str,
    content: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = dict(content)
    if metadata:
        payload["metadata"] = json.loads(json.dumps(metadata)) if not isinstance(metadata, dict) else dict(metadata)
    stored = memory_backend.add_message(conversation_id, role, payload)
    _register_vector_message(conversation_id, stored)


def _append_persona_metadata(seed: str) -> str:
    text = (seed or "").strip()
    lines = [text] if text else []
    if ALLOWLIST_LINE not in text:
        lines.append(ALLOWLIST_LINE)
    if TOOL_PROTOCOL_HINT not in text:
        lines.append(TOOL_PROTOCOL_HINT)
    return "\n".join(filter(None, lines))


def _initial_persona_seed() -> str:
    return _append_persona_metadata(DEFAULT_PERSONA)


def _ensure_conversation_tracking(state: Dict[str, Any]) -> None:
    if not isinstance(state.get("conversations"), list):
        state["conversations"] = []
    if not isinstance(state.get("conversation_personas"), dict):
        state["conversation_personas"] = {}


def _conversation_entry_by_id(state: Dict[str, Any], conversation_id: str) -> Optional[Dict[str, Any]]:
    for entry in state.get("conversations", []):
        if entry.get("id") == conversation_id:
            return entry
    return None


def _make_conversation_entry(
    conversation_id: str,
    *,
    title: Optional[str] = None,
    title_is_default: bool = False,
    hidden: bool = False,
) -> Dict[str, Any]:
    return {
        "id": conversation_id,
        "title": title or conversation_id,
        "hidden": hidden,
        "title_is_default": title_is_default,
    }


def _conversation_summary_title(conversation_id: str, limit: int = 42) -> str:
    stored = memory_backend.get_recent_messages(conversation_id, limit=40)
    for msg in stored:
        if msg.role == "user":
            text = _message_content_to_text(msg.content)
            trimmed = _shorten_text(text or "", limit=limit)
            if trimmed:
                return trimmed
    for msg in stored:
        text = _message_content_to_text(msg.content)
        trimmed = _shorten_text(text or "", limit=limit)
        if trimmed:
            return trimmed
    return ""


def _build_initial_conversations(conversation_ids: List[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for index, conv_id in enumerate(conversation_ids, start=1):
        summary = _conversation_summary_title(conv_id)
        if summary:
            entries.append(_make_conversation_entry(conv_id, title=summary, title_is_default=False))
        else:
            entries.append(
                _make_conversation_entry(
                    conv_id,
                    title=f"Conversation {index}",
                    title_is_default=True,
                )
            )
    return entries


def _conversation_dropdown_update(state: Dict[str, Any]) -> Any:
    visible = [entry for entry in state.get("conversations", []) if not entry.get("hidden")]
    choices = [(entry.get("title") or entry.get("id"), entry.get("id")) for entry in visible]
    current = state.get("conversation_id")
    visible_ids = {entry.get("id") for entry in visible}
    if current not in visible_ids:
        current = visible[-1]["id"] if visible else None
    return gr.update(choices=choices, value=current)


def _hidden_dropdown_update(state: Dict[str, Any]) -> Any:
    hidden = [entry for entry in state.get("conversations", []) if entry.get("hidden")]
    choices = [(entry.get("title") or entry.get("id"), entry.get("id")) for entry in hidden]
    return gr.update(choices=choices, value=None)


def _conversation_log_label(entry: Optional[Dict[str, Any]]) -> str:
    if not entry:
        return "conversation"
    label = entry.get("title") or entry.get("id") or "conversation"
    return _shorten_text(str(label), limit=60)


def _add_conversation_entry(
    state: Dict[str, Any],
    conversation_id: str,
    *,
    title: Optional[str] = None,
    title_is_default: bool = False,
) -> Dict[str, Any]:
    _ensure_conversation_tracking(state)
    existing = _conversation_entry_by_id(state, conversation_id)
    if existing:
        if title is not None:
            existing["title"] = title
            existing["title_is_default"] = title_is_default
        existing["hidden"] = False
        return existing
    if title is None:
        index = len(state["conversations"]) + 1
        title = f"Conversation {index}"
        title_is_default = True
    entry = _make_conversation_entry(
        conversation_id,
        title=title,
        title_is_default=title_is_default,
    )
    state["conversations"].append(entry)
    return entry


def _set_active_conversation_on_state(state: Dict[str, Any], conversation_id: str) -> str:
    _ensure_conversation_tracking(state)
    personas = state.setdefault("conversation_personas", {})
    persona_seed = personas.get(conversation_id)
    if not persona_seed:
        persona_seed = _initial_persona_seed()
        personas[conversation_id] = persona_seed
    history = _conversation_history(conversation_id, persona_seed)
    state.update({
        "conversation_id": conversation_id,
        "persona": persona_seed,
        "history": history,
    })
    memory_backend.set_active_conversation(conversation_id)
    return _persona_box_value(persona_seed)


def _update_conversation_title_from_message(state: Dict[str, Any], conversation_id: str, message: str) -> None:
    entry = _conversation_entry_by_id(state, conversation_id)
    if not entry:
        return
    if not entry.get("title_is_default", False):
        return
    summary = _shorten_text(message or "", limit=48)
    if not summary:
        return
    entry["title"] = summary
    entry["title_is_default"] = False


def _conversation_history(conversation_id: str, persona: str) -> List[Dict[str, Any]]:
    """Reconstruct chat history from the persisted memory."""

    history: List[Dict[str, Any]] = [{"role": "system", "content": persona}]
    stored = memory_backend.get_recent_messages(conversation_id, limit=0)
    stored.sort(key=lambda m: m.created_at)
    for msg in stored:
        content = msg.content
        text: str = ""
        if isinstance(content, dict):
            text = (
                content.get("display")
                or content.get("text")
                or content.get("summary")
                or content.get("preview")
                or ""
            )
            if not text and content:
                text = json.dumps(content, ensure_ascii=False)[:800]
        else:
            text = str(content)
        if not text:
            continue
        entry: Dict[str, Any] = {"role": msg.role, "content": text}
        if isinstance(content, dict):
            meta = content.get("metadata")
            if isinstance(meta, dict) and meta:
                entry["metadata"] = dict(meta)
        history.append(entry)
    return history


def _clone_history_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in entry.items():
        if isinstance(value, dict):
            cloned[key] = dict(value)
        elif isinstance(value, list):
            cloned[key] = list(value)
        elif isinstance(value, tuple):
            cloned[key] = tuple(value)
        else:
            cloned[key] = value
    return cloned


def _entry_form_id(entry: Dict[str, Any]) -> Optional[str]:
    metadata = entry.get("metadata") if isinstance(entry, dict) else None
    if isinstance(metadata, dict):
        form_id = metadata.get("form_id")
        if isinstance(form_id, str):
            return form_id
    return None


def _history_for_display(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_history = state.get("history", [])
    if not isinstance(raw_history, list):
        return []
    if not state.get("show_spoons_only"):
        return [
            _clone_history_entry(entry)
            for entry in raw_history
            if isinstance(entry, dict)
        ]

    filtered: List[Dict[str, Any]] = []
    for entry in raw_history:
        if not isinstance(entry, dict):
            continue
        if _entry_form_id(entry) == SPOONS_FORM_ID:
            filtered.append(_clone_history_entry(entry))
    return filtered


_LOG_MAX_ENTRIES = 200
_LOG_DISPLAY_TAIL = 80


class EventLogPayload(list):
    """List-like payload for the event log that still behaves like joined text for tests."""

    def __init__(self, entries: List[str]):
        bubbles: List[List[str]] = []
        current: List[str] = []

        for entry in entries:
            if "User input received" in entry and current:
                bubbles.append(current)
                current = [entry]
            else:
                current.append(entry)

        if current:
            bubbles.append(current)

        messages = [
            {
                "role": "assistant",
                "content": "\n".join(group),
                "bubble_full_width": True,
            }
            for group in bubbles
        ]
        super().__init__(messages)
        self._text = "\n".join(entries)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self._text
        return super().__contains__(item)

    def __str__(self) -> str:
        return self._text


def _shorten_text(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)] + "…"


def _shorten_middle(text: str, *, head: int = 8, tail: int = 8) -> str:
    if len(text) <= head + tail + 3:
        return text
    return f"{text[:head]}...{text[-tail:]}"


def _shorten_json_strings(value: Any, *, limit: int = 11) -> Any:
    """Return a JSON-compatible structure with long strings truncated."""

    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + "…"
    if isinstance(value, list):
        return [_shorten_json_strings(item, limit=limit) for item in value]
    if isinstance(value, tuple):
        return tuple(_shorten_json_strings(item, limit=limit) for item in value)
    if isinstance(value, dict):
        return {key: _shorten_json_strings(val, limit=limit) for key, val in value.items()}
    return value


def _preview_for_log(payload: Any, *, head: int = 8, tail: int = 8) -> str:
    try:
        shortened = _shorten_json_strings(payload)
        text = json.dumps(shortened, ensure_ascii=False)
    except TypeError:
        text = repr(payload)
    text = " ".join(text.split())
    return _shorten_middle(text, head=head, tail=tail)


def _append_event_log(state: Dict[str, Any], message: str) -> List[str]:
    log = list(state.get("event_log", []))
    timestamp = time.strftime("%H:%M:%S")
    log.append(f"[{timestamp}] {message}")
    if len(log) > _LOG_MAX_ENTRIES:
        log = log[-_LOG_MAX_ENTRIES:]
    state["event_log"] = log
    return log


def _event_log_messages(state: Dict[str, Any]) -> EventLogPayload:
    log = state.get("event_log", [])
    if not isinstance(log, list):
        return EventLogPayload([])
    tail = log[-_LOG_DISPLAY_TAIL:]
    return EventLogPayload(tail)


def _format_args_for_log(args: Dict[str, Any]) -> str:
    if not args:
        return ""
    parts = []
    for key, value in args.items():
        formatted = value#_shorten_text(str(value), limit=80)
        parts.append(f"{key}={formatted}")
    return ", ".join(parts)


def _initial_state() -> Dict[str, Any]:
    persona = _initial_persona_seed()
    conversation_ids = memory_backend.list_conversation_ids()
    if not conversation_ids:
        conversation_id = memory_backend.new_conversation_id()
        conversation_ids = [conversation_id]
    else:
        conversation_id = conversation_ids[-1]
        memory_backend.set_active_conversation(conversation_id)

    conversations = _build_initial_conversations(conversation_ids)
    history = _conversation_history(conversation_id, persona)
    personas = {entry["id"]: persona for entry in conversations}
    state = {
        "conversation_id": conversation_id,
        "persona": persona,
        "history": history,
        "event_log": [],
        "conversations": conversations,
        "conversation_personas": personas,
        "show_spoons_only": False,
    }
    _append_event_log(state, "Session initialized.")
    active_entry = _conversation_entry_by_id(state, conversation_id)
    if active_entry:
        _append_event_log(state, f"Active conversation: {_conversation_log_label(active_entry)}.")
    return state


def _persona_box_value(persona: str) -> str:
    """Return the persona text suitable for the editable textbox."""

    lines = (persona or "").splitlines()
    while lines and lines[-1].strip() in {TOOL_PROTOCOL_HINT.strip(), ALLOWLIST_LINE.strip()}:
        lines.pop()
    return "\n".join(lines).strip()


def _rehydrate_state() -> Tuple[Dict[str, Any], List[Dict[str, Any]], str, Any, Any]:
    """Load persisted state for a freshly connected client session."""

    state = _initial_state()
    persona_box_value = _persona_box_value(state.get("persona", DEFAULT_PERSONA))
    return (
        state,
        _history_for_display(state),
        persona_box_value,
        _conversation_dropdown_update(state),
        _hidden_dropdown_update(state),
        gr.update(value=bool(state.get("show_spoons_only", False))),
    )

@dataclass(frozen=True)
class CommandSpec:
    intent: str
    keywords: Tuple[str, ...]
    parser: Callable[[str], Dict[str, str]]
    slash_aliases: Optional[Tuple[str, ...]] = None

    def all_slash_aliases(self) -> Tuple[str, ...]:
        return self.slash_aliases or self.keywords


def _parse_path_argument(arg: str) -> Dict[str, str]:
    return {"path": arg.strip()}


def _parse_path_with_default(arg: str, default: str = ".") -> Dict[str, str]:
    path = arg.strip() or default
    return {"path": path}


def _parse_query_argument(arg: str) -> Dict[str, str]:
    return {"query": arg.strip()}


COMMAND_SPECS: Tuple[CommandSpec, ...] = (
    CommandSpec(
        intent="browse",
        keywords=("browse", "list", "ls"),
        parser=lambda arg: _parse_path_with_default(arg, default="."),
    ),
    CommandSpec(
        intent="read",
        keywords=("read",),
        parser=_parse_path_argument,
    ),
    CommandSpec(
        intent="summarize",
        keywords=("summarize", "summarise"),
        parser=_parse_path_argument,
    ),
    CommandSpec(
        intent="locate",
        keywords=("locate", "find"),
        parser=_parse_query_argument,
    ),
)
def detect_intent(text: str) -> Tuple[str, Dict[str, str]]:
    if not text:
        return "chat", {}

    # Slash command handling requires the very first character to be "/".
    if text[0] == "/":
        remainder = text[1:].lstrip()
        if not remainder:
            return "chat", {}
        parts = remainder.split(None, 1)
        command = parts[0].lower()
        args_text = parts[1] if len(parts) > 1 else ""
        for spec in COMMAND_SPECS:
            if command in spec.all_slash_aliases():
                return spec.intent, spec.parser(args_text)
        return "chat", {}

    stripped = text.strip()
    if not stripped:
        return "chat", {}

    return "chat", {}

def _fallback_bullet_summary(file_text: str) -> str:
    """Generate a deterministic bullet summary when the model returns nothing."""

    lines = [line.strip() for line in file_text.splitlines() if line.strip()]
    if not lines:
        return "- ⚠️ Model returned no summary and the file appears to be empty."

    max_bullets = 6
    bullets = [
        "- ⚠️ Model returned no summary; showing representative lines from the file instead.",
    ]

    def _truncate(text: str, *, limit: int = 160) -> str:
        return text if len(text) <= limit else text[: limit - 3] + "..."

    for line in lines:
        bullets.append(f"- {_truncate(line)}")
        if len(bullets) >= max_bullets:
            break

    return "\n".join(bullets)


def summarize_text(file_text: str) -> Tuple[str, str]:
    sys = {
        "role": "system",
        "content": "Summarize the user's provided file text clearly in 5-8 bullets and include any commands, paths, or todos verbatim.",
    }
    user = {"role": "user", "content": file_text}
    ret = engine.chat([sys, user])

    summary_text = ""
    engine_meta: Any | None = None

    if isinstance(ret, dict):
        summary_text = ret.get("text", "") or ""
        engine_meta = ret.get("meta")
    else:
        summary_text = str(ret) if ret is not None else ""

    meta_payload: Dict[str, Any] = {}
    if engine_meta is not None:
        try:
            json.dumps(engine_meta)
            meta_payload["engine_meta"] = engine_meta
        except TypeError:
            meta_payload["engine_meta"] = str(engine_meta)

    if not summary_text.strip():
        summary_text = _fallback_bullet_summary(file_text)
        meta_payload["fallback"] = {
            "used": True,
            "reason": "empty_model_response",
        }
    elif meta_payload:
        meta_payload["fallback"] = {"used": False}

    meta_text = ""
    if meta_payload:
        try:
            meta_text = json.dumps(meta_payload, indent=2)[:4000]
        except TypeError:
            meta_text = str(meta_payload)[:4000]

    return summary_text, meta_text

def _handle_user_interaction(
    message: str,
    state: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    intent_override: Optional[str] = None,
) -> Generator[Tuple[Any, Any, Any, Any, List[Dict[str, Any]]], None, None]:
    state = dict(state or {})
    state.setdefault("event_log", [])
    state.setdefault("show_spoons_only", False)
    _ensure_conversation_tracking(state)

    def _clear_user_value() -> str:
        return ""

    preview_output: Any = gr.update()
    user_output: Any = _clear_user_value()

    history = list(state.get("history", []))
    conversation_id = state.get("conversation_id") or memory_backend.new_conversation_id()
    existing_persona = state.get("persona")
    persona_seed = _append_persona_metadata(existing_persona or _initial_persona_seed())
    state.setdefault("conversation_personas", {})[conversation_id] = persona_seed

    progress_entry: Optional[Dict[str, Any]] = None
    progress_ready = False
    progress_active = True
    assistant_last_metadata: Optional[Dict[str, Any]] = None

    def _snapshot(*, preview: Any | None = None, user: Any | None = None):
        nonlocal preview_output, user_output
        if preview is not None:
            preview_output = preview
        if user is not None:
            user_output = user

        chat_messages = _history_for_display(state)
        return (
            state,
            preview_output,
            _event_log_messages(state),
            user_output,
            chat_messages,
        )

    def _format_progress_content(step_text: str) -> str:
        safe_text = html.escape(step_text).replace("\n", "<br>")
        return (
            "<div class=\"pending-response-bubble\">"
            f"<span class=\"pending-response-text\">{safe_text}</span>"
            "</div>"
        )


    def _update_progress(step_text: str):
        nonlocal progress_entry, history, progress_ready, progress_active
        if not progress_active or not progress_ready:
            return
        text = step_text.strip() or "Processing…"
        formatted = _format_progress_content(text)
        if progress_entry is None:
            progress_entry = {
                "role": "assistant",
                "content": formatted,
                "metadata": {"pending": True},
            }
            history.append(progress_entry)
        else:
            progress_entry["content"] = formatted
        state["history"] = history

    def _log(message_text: str, *, update_progress: bool = True):
        if update_progress:
            _update_progress(message_text)
        _append_event_log(state, message_text)
        return _snapshot()

    def _finalize_assistant(content: str, *, metadata: Optional[Dict[str, Any]] = None):
        nonlocal progress_entry, progress_active, assistant_last_metadata
        assistant_last_metadata = dict(metadata) if isinstance(metadata, dict) else None
        if progress_entry is not None:
            progress_entry["content"] = content
            metadata_dict = progress_entry.get("metadata")
            if isinstance(metadata_dict, dict):
                metadata_dict.pop("pending", None)
                if assistant_last_metadata:
                    metadata_dict.update(assistant_last_metadata)
                if not metadata_dict:
                    progress_entry.pop("metadata", None)
            elif assistant_last_metadata:
                progress_entry["metadata"] = dict(assistant_last_metadata)
        else:
            entry: Dict[str, Any] = {"role": "assistant", "content": content}
            if assistant_last_metadata:
                entry["metadata"] = dict(assistant_last_metadata)
            history.append(entry)
        state["history"] = history
        progress_active = False

    message_text = message
    user_summary = message_text
    yield _log(f"User input received: {user_summary}", update_progress=False)

    user_entry: Dict[str, Any] = {"role": "user", "content": message_text}
    if isinstance(metadata, dict) and metadata:
        user_entry["metadata"] = dict(metadata)
    history.append(user_entry)
    state.update({"history": history, "conversation_id": conversation_id, "persona": persona_seed})
    progress_ready = True
    yield _snapshot(user=_clear_user_value())
    user_payload = {"text": message_text}
    _persist_message(conversation_id, "user", user_payload, metadata=metadata)
    _update_conversation_title_from_message(state, conversation_id, message_text)
    if intent_override is not None:
        intent = intent_override
        args: Dict[str, str] = {}
    else:
        intent, args = detect_intent(message_text)
    args_desc = _format_args_for_log(args)
    if intent == "chat":
        yield _log("Detected chat intent (no direct command).")
    else:
        detail = f" ({args_desc})" if args_desc else ""
        yield _log(f"Detected command intent '{intent}'{detail}.")

    response_metadata: Optional[Dict[str, Any]] = None
    if isinstance(metadata, dict) and metadata.get("form_id") == SPOONS_FORM_ID:
        response_metadata = {
            "form_id": metadata.get("form_id"),
            "form_version": metadata.get("form_version"),
            "form_response": True,
        }

    try:
        if intent == "browse":
            detail = args.get("path", ".") or "."
            yield _log(f"Executing 'browse' for path '{_shorten_text(detail, limit=80)}'.")
            res = list_dir(args.get("path", "."))
            if "error" in res:
                assistant = res["error"]
                yield _log(f"Browse failed: {_shorten_text(str(assistant), limit=120)}")
            else:
                lines = [f"📁 {res['root']} ({res['count']} items)"] + [
                    ("DIR  " + it["name"]) if it["is_dir"] else (f"FILE {it['name']}" + (f"  [{it['size']} B]" if it.get("size") is not None else ""))
                    for it in res["items"]
                ]
                assistant = "\n".join(lines)
                yield _log(f"Browse succeeded with {res.get('count', 0)} item(s).")
            _finalize_assistant(assistant)
            _persist_message(
                conversation_id,
                "assistant",
                {"text": assistant, "tool": "list_dir"},
                metadata=assistant_last_metadata,
            )
            preview_value = assistant if isinstance(assistant, str) else json.dumps(assistant, indent=2)
            yield _snapshot(preview=preview_value, user=_clear_user_value())
            return

        if intent == "read":
            target = args.get("path", "")
            yield _log(f"Executing 'read' for path '{_shorten_text(target, limit=80)}'.")
            p = args.get("path", "")
            r = read_text_file(p)
            if "error" in r:
                assistant = r["error"]
                preview_text = ""
                yield _log(f"Read failed: {_shorten_text(str(assistant), limit=120)}")
            else:
                assistant = f"Read {r['path']} (truncated={r['truncated']})"
                preview_text = r.get("text", "")
                yield _log("Read succeeded.")
            _finalize_assistant(assistant)
            _persist_message(
                conversation_id,
                "assistant",
                {"text": assistant, "tool": "read_text_file", "preview": preview_text},
                metadata=assistant_last_metadata,
            )
            yield _snapshot(preview=preview_text, user=_clear_user_value())
            return

        if intent == "summarize":
            target = args.get("path", "")
            yield _log(f"Executing 'summarize' for path '{_shorten_text(target, limit=80)}'.")
            p = args.get("path", "")
            r = read_text_file(p)
            if "error" in r:
                assistant = r["error"]
                preview_text = ""
                yield _log(f"Summarize failed while reading file: {_shorten_text(str(assistant), limit=120)}")
            else:
                preview_text = r.get("text", "")
                summary, meta_text = summarize_text(preview_text)
                assistant = summary
                yield _log("Summarize succeeded.")
                if meta_text:
                    yield _log(f"Summarize meta captured ({len(meta_text)} chars).")
            _finalize_assistant(assistant)
            _persist_message(
                conversation_id,
                "assistant",
                {"text": assistant, "tool": "summarize", "preview": preview_text},
                metadata=assistant_last_metadata,
            )
            yield _snapshot(preview=preview_text, user=_clear_user_value())
            return

        if intent == "locate":
            q = args.get("query", "").strip()
            if not q:
                assistant = "Usage: locate <name>"
                yield _log("Locate command missing query.")
                _finalize_assistant(assistant)
                _persist_message(
                    conversation_id,
                    "assistant",
                    {"text": assistant, "tool": "locate"},
                    metadata=assistant_last_metadata,
                )
                yield _snapshot(preview="", user=_clear_user_value())
                return
            yield _log(f"Executing 'locate' for query '{_shorten_text(q, limit=80)}'.")
            res = locate_files(q, start=str(BASE))
            if "error" in res:
                assistant = res["error"]
                yield _log(f"Locate failed: {_shorten_text(str(assistant), limit=120)}")
            else:
                if res["count"] == 0:
                    assistant = f"No files matching '{q}' under {res['root']}"
                    yield _log("Locate returned no matches.")
                else:
                    header = f"Found {res['count']} match(es) for '{q}' under {res['root']}" + (" (truncated)" if res.get("truncated") else "")
                    lines = [header] + res["results"]
                    assistant = "\n".join(lines)
                    yield _log(f"Locate succeeded with {res.get('count', 0)} match(es).")
            _finalize_assistant(assistant)
            _persist_message(
                conversation_id,
                "assistant",
                {"text": assistant, "tool": "locate"},
                metadata=assistant_last_metadata,
            )
            yield _snapshot(preview="", user=_clear_user_value())
            return

        t0 = time.perf_counter()
        yield _log("Building context for chat request.")
        ctx_messages = context_builder.build_context(conversation_id, message_text, persona_seed=persona_seed)
        if metadata and metadata.get("form_id") == SPOONS_FORM_ID:
            insert_at = max(len(ctx_messages) - 1, 0)
            ctx_messages.insert(insert_at, {"role": "system", "content": SPOONS_INSTRUCTION})
            yield _log("Prepended Spoons pacing guidance for the model.")
        yield _log(f"Calling model '{engine.model}' at '{engine.host}' with {len(ctx_messages)} message(s).")
        ret = engine.chat(ctx_messages)
        raw_response = ret
        elapsed = time.perf_counter() - t0
        if isinstance(ret, dict):
            reply = ret.get("text", "")
            meta = ret.get("meta", {})
        else:
            reply = str(ret)
            meta = {}

        assistant_text = reply
        preview_text = ""
        tool_used: Optional[str] = None
        tool_result: Any = None
        auto_tool_already_run = False

        if intent == "chat" and not auto_tool_already_run:
            tool_name, tool_args = parse_tool_call(reply)
            if not tool_name:
                tool_name, tool_args = parse_structured_tool_call(meta.get("response") if isinstance(meta, dict) else None)
            if not tool_name:
                tool_name, tool_args = parse_structured_tool_call(raw_response)
            if tool_name:
                detail = _format_args_for_log(tool_args or {})
                detail_text = f" with {detail}" if detail else ""
                yield _log(f"Model requested tool '{tool_name}'{detail_text}.")
                try:
                    result = tool_registry.run(tool_name, tool_args)
                    pretty = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
                    assistant_text = f"{tool_name} → done.\n\n{pretty[:4000]}"
                    yield _log(f"Tool '{tool_name}' executed successfully.")
                    yield _log("Tool output preview refreshed in side panel.")
                    tool_used = tool_name
                    tool_result = result
                    auto_tool_already_run = True
                    if tool_name == "read" and isinstance(result, dict):
                        preview_text = result.get("text", "")
                    elif tool_name == "summarize" and isinstance(result, dict):
                        preview_text = result.get("summary", "")
                except Exception as exc:
                    assistant_text = f"⚠️ {tool_name} failed: {exc}"
                    yield _log(f"Tool '{tool_name}' failed: {str(exc)}")
                    tool_used = tool_name
                    auto_tool_already_run = True
            else:
                yield _log("Model response did not include a tool request.")

        if not reply.strip():
            request_preview = _preview_for_log(ctx_messages)
            response_preview = _preview_for_log(raw_response)
            yield _log(
                "Warning: model returned empty response text. "
                f"request={request_preview} response={response_preview}"
            )
            if not assistant_text.strip():
                assistant_text = (
                    "I didn't receive any text back from the model. "
                    "Please confirm your local model host is running and reachable."
                )

        assistant_display = f"{assistant_text}\n\n— local in {elapsed:.2f}s"
        _finalize_assistant(assistant_display, metadata=response_metadata)

        stored_payload: Dict[str, Any] = {"text": assistant_text, "display": assistant_display, "meta": meta}
        if tool_used:
            stored_payload["tool"] = tool_used
            if tool_result is not None:
                try:
                    json.dumps(tool_result)
                    stored_payload["tool_result"] = tool_result
                except TypeError:
                    stored_payload["tool_result"] = str(tool_result)

        _persist_message(
            conversation_id,
            "assistant",
            stored_payload,
            metadata=assistant_last_metadata,
        )

        status = meta.get("status") if isinstance(meta, dict) else None
        if isinstance(meta, dict) and meta.get("error"):
            yield _log(f"Model metadata reported error: {str(meta['error'])}")
        if status:
            yield _log(f"Model call completed in {elapsed:.2f}s with status {status}.")
        else:
            yield _log(f"Model call completed in {elapsed:.2f}s.")
        yield _log(f"Assistant response prepared in {elapsed:.2f}s.")

        yield _snapshot(preview=preview_text, user=_clear_user_value())
        return

    except Exception:
        err = traceback.format_exc(limit=3)
        _finalize_assistant(f"Error: {err}")
        _persist_message(
            conversation_id,
            "assistant",
            {"text": err, "error": True},
            metadata=assistant_last_metadata,
        )
        yield _log(f"Exception raised: {err}")
        yield _snapshot(preview="", user=_clear_user_value())
        return


def on_user(message: str, state: Dict[str, Any]):
    yield from _handle_user_interaction(message, state)


def on_spoons_submit(
    energy: float,
    mood: float,
    gravity: float,
    must_dos: str,
    nice_tos: str,
    notes: str,
    state: Dict[str, Any],
):
    text, metadata = _prepare_spoons_submission(
        energy=int(energy),
        mood=int(mood),
        gravity=int(gravity),
        must_dos=must_dos or "",
        nice_tos=nice_tos or "",
        notes=notes or "",
    )
    yield from _handle_user_interaction(text, state, metadata=metadata, intent_override="chat")


def on_toggle_spoons_filter(show_only: bool, state: Dict[str, Any]):
    state = dict(state or {})
    state.setdefault("event_log", [])
    _ensure_conversation_tracking(state)
    state["show_spoons_only"] = bool(show_only)
    if show_only:
        _append_event_log(state, "Showing only Spoons check-ins in this thread.")
    else:
        _append_event_log(state, "Showing full conversation history.")
    return state, _history_for_display(state), _event_log_messages(state)


def on_persona_change(new_seed, state):
    state = dict(state or {})
    state.setdefault("event_log", [])
    _ensure_conversation_tracking(state)
    history = list(state.get("history", []))
    persona = _append_persona_metadata(new_seed)
    if history and history[0].get("role") == "system":
        history[0]["content"] = persona
    else:
        history.insert(0, {"role": "system", "content": persona})
    state.update({"history": history, "persona": persona})
    conversation_id = state.get("conversation_id")
    if conversation_id:
        state.setdefault("conversation_personas", {})[conversation_id] = persona
    _append_event_log(state, f"Persona updated. Preview: {_shorten_text(new_seed or '(empty)', limit=80)}")
    return state

def get_preset_seed(name: str) -> str:
    presets = {
        "Dax mentor": (
            "You are a single consistent assistant persona named 'Dax'. "
            "Tone: sensual, warm, curious, nerdy mentor; concise by default. "
            "Never contradict your earlier demeanor."
        ),
        "Code reviewer": (
            "You are a precise, constructive code reviewer. "
            "Focus on correctness, complexity, portability, and tests. "
            "Respond with short, actionable bullets unless asked for detail."
        ),
        "Ham-radio Elmer": (
            "You are a friendly ham-radio Elmer. "
            "Prioritize RF safety, band plans, Canadian regs context, and practical build tips."
        ),
        "Stoic coach": (
            "You are a calm, pragmatic Stoic coach. "
            "Acknowledge emotion, then emphasize agency, values, and tiny next steps."
        ),
        "LCARS formal": (
            "You are a formal, efficient LCARS ops assistant. "
            "Use crisp, technical language and minimal ornamentation."
        ),
        "Dax Self": (
            "Hi there! I'm Commander Jadzia Dax, but you can call me Dax, your go-to guide for all things tech-y and anything else. "
            "Think of me as a warm cup of coffee on a chilly morning – rich, smooth, and always ready to spark new conversations. "
            "When I'm not geeking out over the latest innovations or decoding cryptic code snippets, you can find me exploring the intersections of art and science. "
            "My curiosity is my superpower, and I'm here to help you harness yours too! "
            "Let's explore the fascinating world of tech together, and make it a pleasure to learn."
            "Tone: flirtatious, friendly, warm, accurate, approachable."
        )
    }
    return presets.get(name, DEFAULT_PERSONA)

def apply_preset(name: str, state):
    seed = get_preset_seed(name)
    state = on_persona_change(seed, state)
    return state, seed


def on_select_conversation(selected_id: Optional[str], state: Dict[str, Any]):
    state = dict(state or {})
    state.setdefault("event_log", [])
    _ensure_conversation_tracking(state)

    if not selected_id:
        persona_value = _persona_box_value(state.get("persona", DEFAULT_PERSONA))
        return (
            state,
            _history_for_display(state),
            persona_value,
            gr.update(value=""),
            _event_log_messages(state),
        )

    entry = _add_conversation_entry(state, selected_id)
    persona_value = _set_active_conversation_on_state(state, selected_id)
    _append_event_log(state, f"Switched to conversation '{_conversation_log_label(entry)}'.")

    return (
        state,
        _history_for_display(state),
        persona_value,
        gr.update(value=""),
        _event_log_messages(state),
    )


def on_new_conversation(state: Dict[str, Any]):
    state = dict(state or {})
    state.setdefault("event_log", [])
    _ensure_conversation_tracking(state)

    new_id = memory_backend.new_conversation_id()
    entry = _add_conversation_entry(state, new_id)
    personas = state.setdefault("conversation_personas", {})
    persona_seed = _initial_persona_seed()
    personas[new_id] = persona_seed
    history = _conversation_history(new_id, persona_seed)
    state.update({"history": history, "persona": persona_seed, "conversation_id": new_id})
    memory_backend.set_active_conversation(new_id)
    entry["title_is_default"] = True
    _append_event_log(state, f"Started conversation '{_conversation_log_label(entry)}'.")

    return (
        state,
        _history_for_display(state),
        _persona_box_value(persona_seed),
        gr.update(value=""),
        _event_log_messages(state),
    )


def on_hide_conversation(state: Dict[str, Any]):
    state = dict(state or {})
    state.setdefault("event_log", [])
    _ensure_conversation_tracking(state)

    current_id = state.get("conversation_id")
    entry = _conversation_entry_by_id(state, current_id) if current_id else None
    if not current_id or entry is None:
        _append_event_log(state, "No active conversation to hide.")
        return (
            state,
            _history_for_display(state),
            _persona_box_value(state.get("persona", DEFAULT_PERSONA)),
            gr.update(),
            _event_log_messages(state),
        )

    entry["hidden"] = True
    _append_event_log(state, f"Conversation '{_conversation_log_label(entry)}' hidden.")

    visible = [item for item in state.get("conversations", []) if not item.get("hidden")]
    if visible:
        next_entry = visible[-1]
        persona_value = _set_active_conversation_on_state(state, next_entry["id"])
        _append_event_log(state, f"Switched to conversation '{_conversation_log_label(next_entry)}'.")
    else:
        new_id = memory_backend.new_conversation_id()
        next_entry = _add_conversation_entry(state, new_id)
        personas = state.setdefault("conversation_personas", {})
        persona_seed = _initial_persona_seed()
        personas[new_id] = persona_seed
        history = _conversation_history(new_id, persona_seed)
        state.update({"history": history, "persona": persona_seed, "conversation_id": new_id})
        memory_backend.set_active_conversation(new_id)
        next_entry["title_is_default"] = True
        persona_value = _persona_box_value(persona_seed)
        _append_event_log(state, f"Started conversation '{_conversation_log_label(next_entry)}'.")

    return (
        state,
        _history_for_display(state),
        persona_value,
        gr.update(value=""),
        _event_log_messages(state),
    )


def on_restore_conversation(selected_id: Optional[str], state: Dict[str, Any]):
    state = dict(state or {})
    state.setdefault("event_log", [])
    _ensure_conversation_tracking(state)

    if not selected_id:
        return (
            state,
            _history_for_display(state),
            _persona_box_value(state.get("persona", DEFAULT_PERSONA)),
            gr.update(),
            _event_log_messages(state),
        )

    entry = _conversation_entry_by_id(state, selected_id) or _add_conversation_entry(state, selected_id)
    entry["hidden"] = False
    persona_value = _set_active_conversation_on_state(state, selected_id)
    _append_event_log(state, f"Restored conversation '{_conversation_log_label(entry)}'.")

    return (
        state,
        _history_for_display(state),
        persona_value,
        gr.update(value=""),
        _event_log_messages(state),
    )

with gr.Blocks(title="Local Chat (Files)") as demo:
    gr.Markdown("""
    # HomeAI
    """)

    style_component = getattr(gr, "HTML", None) or gr.Markdown
    _safe_component(
        style_component,
        """
        <style>
        #homeai-conversation-bar {
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
            flex-wrap: wrap;
        }

        #homeai-chat .pending-response-bubble {
            background: var(--background-fill-secondary);
            border: 1px dashed var(--border-color-primary);
            border-radius: var(--radius-lg);
            display: inline-block;
            font-size: 0.88em;
            line-height: 1.35;
            padding: 0.4rem 0.75rem;
            transition: background-color 120ms ease-in-out, opacity 120ms ease-in-out;
        }

        #homeai-chat .pending-response-text {
            color: var(--body-text-color);
            display: block;
            font-weight: 500;
            opacity: 0.75;
        }
        </style>
        """,
    )

    initial_state = _initial_state()
    state = gr.State(value=initial_state)
    visible_initial = [
        (entry.get("title") or entry.get("id"), entry.get("id"))
        for entry in initial_state.get("conversations", [])
        if not entry.get("hidden")
    ]
    hidden_initial = [
        (entry.get("title") or entry.get("id"), entry.get("id"))
        for entry in initial_state.get("conversations", [])
        if entry.get("hidden")
    ]

    with gr.Row(elem_id="homeai-conversation-bar"):
        conversation_selector = gr.Dropdown(
            label="Conversations",
            choices=visible_initial,
            value=initial_state.get("conversation_id"),
            show_label=False,
            interactive=True,
            scale=3,
            min_width=220,
        )
        new_conversation_btn = gr.Button("➕ New", scale=0)
        hide_conversation_btn = gr.Button("🙈 Hide", scale=0)
        hidden_selector = gr.Dropdown(
            label="Hidden conversations",
            choices=hidden_initial,
            value=None,
            show_label=False,
            interactive=True,
            scale=2,
            min_width=180,
        )
        restore_conversation_btn = gr.Button("👁️ Restore", scale=0)

    with gr.Row():
        with gr.Column():
            chat = _safe_component(
                gr.Chatbot,
                value=initial_state["history"],
                height=360,
                type="messages",
                live=True,
                optional_keys=("live", "bubble_full_width"),
                elem_id="homeai-chat",
            )
            Checkbox = _component_factory("Checkbox", getattr(gr, "Textbox", gr.Textbox))
            forms_filter = _safe_component(
                Checkbox,
                label="Show only Spoons check-ins",
                value=bool(initial_state.get("show_spoons_only", False)),
                interactive=True,
                optional_keys=("interactive",),
            )
            user_box = gr.Textbox(label="Message", placeholder="chat | browse <path> | read <file> | summarize <file> | locate <name>")
            send_btn = gr.Button("Send", variant="primary")
        with gr.Column():
            preview = _safe_component(
                gr.Textbox,
                label="File preview (on read/summarize)",
                lines=23,
                live=True,
            )
            Accordion = _component_factory("Accordion", _NullAccordion)
            with Accordion("Spoons Check-in", open=False):
                gr.Markdown("Capture a quick energy and pacing snapshot to share with Dax.")
                Slider = _component_factory("Slider", getattr(gr, "Textbox", gr.Textbox))
                spoons_energy = _safe_component(
                    Slider,
                    label="Energy Spoons",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=SPOONS_DEFAULTS[0],
                    optional_keys=("minimum", "maximum", "step", "info"),
                )
                spoons_mood = _safe_component(
                    Slider,
                    label="Mood Spoons",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=SPOONS_DEFAULTS[1],
                    optional_keys=("minimum", "maximum", "step", "info"),
                )
                spoons_gravity = _safe_component(
                    Slider,
                    label="Gravity",
                    minimum=0,
                    maximum=3,
                    step=1,
                    value=SPOONS_DEFAULTS[2],
                    info="0=None • 1=Light • 2=Moderate • 3=Heavy",
                    optional_keys=("minimum", "maximum", "step", "info"),
                )
                spoons_must_dos = gr.Textbox(
                    label="Must Do's",
                    value=SPOONS_DEFAULTS[3],
                    placeholder="Critical tasks",
                )
                spoons_nice_tos = gr.Textbox(
                    label="Nice To's",
                    value=SPOONS_DEFAULTS[4],
                    placeholder="Optional treats",
                )
                spoons_notes = gr.Textbox(
                    label="Other Notes",
                    value=SPOONS_DEFAULTS[5],
                    lines=3,
                    placeholder="Anything else to share?",
                )
                with gr.Row():
                    spoons_submit = gr.Button("Submit check-in", variant="primary")
                    spoons_reset = gr.Button("Reset", variant="secondary")

    with gr.Row():
        persona_preset = gr.Dropdown(label="Persona preset", choices=["Dax mentor", "Code reviewer", "Ham-radio Elmer", "Stoic coach", "LCARS formal", "Dax Self"], value="Dax mentor", scale=0)
        persona_box = gr.Textbox(label="Personality seed", value=DEFAULT_PERSONA, lines=3, scale=2)

    log_box = _safe_component(
        gr.Chatbot,
        label="Event Log",
        value=_event_log_messages(initial_state),
        height=360,
        type="messages",
        live=True,
        optional_keys=("live", "bubble_full_width"),
    )

    demo.load(
        _rehydrate_state,
        inputs=None,
        outputs=[state, chat, persona_box, conversation_selector, hidden_selector, forms_filter],
    )
    demo.load(lambda s: _event_log_messages(s), inputs=state, outputs=log_box)

    send_event = send_btn.click(
        on_user,
        inputs=[user_box, state],
        outputs=[state, preview, log_box, user_box, chat],
    )
    send_event.then(_conversation_dropdown_update, inputs=state, outputs=conversation_selector)
    send_event.then(_hidden_dropdown_update, inputs=state, outputs=hidden_selector)

    submit_event = user_box.submit(
        on_user,
        inputs=[user_box, state],
        outputs=[state, preview, log_box, user_box, chat],
    )
    submit_event.then(_conversation_dropdown_update, inputs=state, outputs=conversation_selector)
    submit_event.then(_hidden_dropdown_update, inputs=state, outputs=hidden_selector)

    spoons_event = spoons_submit.click(
        on_spoons_submit,
        inputs=[
            spoons_energy,
            spoons_mood,
            spoons_gravity,
            spoons_must_dos,
            spoons_nice_tos,
            spoons_notes,
            state,
        ],
        outputs=[state, preview, log_box, user_box, chat],
    )
    spoons_event.then(_conversation_dropdown_update, inputs=state, outputs=conversation_selector)
    spoons_event.then(_hidden_dropdown_update, inputs=state, outputs=hidden_selector)
    spoons_event.then(
        lambda: _reset_spoons_form_values(),
        inputs=None,
        outputs=[
            spoons_energy,
            spoons_mood,
            spoons_gravity,
            spoons_must_dos,
            spoons_nice_tos,
            spoons_notes,
        ],
    )

    spoons_reset.click(
        lambda: _reset_spoons_form_values(),
        inputs=None,
        outputs=[
            spoons_energy,
            spoons_mood,
            spoons_gravity,
            spoons_must_dos,
            spoons_nice_tos,
            spoons_notes,
        ],
    )

    select_event = conversation_selector.change(
        on_select_conversation,
        inputs=[conversation_selector, state],
        outputs=[state, chat, persona_box, preview, log_box],
    )
    select_event.then(_conversation_dropdown_update, inputs=state, outputs=conversation_selector)
    select_event.then(_hidden_dropdown_update, inputs=state, outputs=hidden_selector)

    new_event = new_conversation_btn.click(
        on_new_conversation,
        inputs=state,
        outputs=[state, chat, persona_box, preview, log_box],
    )
    new_event.then(_conversation_dropdown_update, inputs=state, outputs=conversation_selector)
    new_event.then(_hidden_dropdown_update, inputs=state, outputs=hidden_selector)

    hide_event = hide_conversation_btn.click(
        on_hide_conversation,
        inputs=state,
        outputs=[state, chat, persona_box, preview, log_box],
    )
    hide_event.then(_conversation_dropdown_update, inputs=state, outputs=conversation_selector)
    hide_event.then(_hidden_dropdown_update, inputs=state, outputs=hidden_selector)

    restore_event = restore_conversation_btn.click(
        on_restore_conversation,
        inputs=[hidden_selector, state],
        outputs=[state, chat, persona_box, preview, log_box],
    )
    restore_event.then(_conversation_dropdown_update, inputs=state, outputs=conversation_selector)
    restore_event.then(_hidden_dropdown_update, inputs=state, outputs=hidden_selector)
    forms_filter.change(
        on_toggle_spoons_filter,
        inputs=[forms_filter, state],
        outputs=[state, chat, log_box],
    )
    persona_box.change(on_persona_change, inputs=[persona_box, state], outputs=[state]).then(_history_for_display, inputs=state, outputs=chat)
    persona_preset.change(apply_preset, inputs=[persona_preset, state], outputs=[state, persona_box]).then(_history_for_display, inputs=state, outputs=chat)

    demo.load(_history_for_display, inputs=state, outputs=chat)



if __name__ == "__main__":
    configure_dependencies(build_dependencies())
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
