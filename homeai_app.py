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

import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from context_memory import ContextBuilder, LocalJSONMemoryBackend

import homeai.config as homeai_config
import homeai.filesystem as filesystem
from homeai.model_engine import LocalModelEngine
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


tool_registry = ToolRegistry()

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

# Register the adapters

def tool_browse(path: str = ".", pattern: str = "") -> Dict[str, Any]:
    res = list_dir(path, pattern)
    if isinstance(res, dict) and res.get("error"):
        raise RuntimeError(res["error"])
    return res


tool_registry.register("browse", tool_browse)
tool_registry.register("read", tool_read)
tool_registry.register("summarize", tool_summarize)
tool_registry.register("locate", tool_locate)

engine = LocalModelEngine()
memory_backend: LocalJSONMemoryBackend
context_builder: ContextBuilder


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


def _init_memory_backend(*, storage: str | None = None) -> None:
    global memory_backend, context_builder
    selected_storage = storage or os.getenv("HOMEAI_STORAGE")
    kwargs: Dict[str, Any] = {}
    if selected_storage:
        kwargs["storage"] = selected_storage
    memory_backend = LocalJSONMemoryBackend(**kwargs)
    context_overrides = _context_builder_env_overrides()
    context_builder = ContextBuilder(memory_backend, **context_overrides)


_init_memory_backend()


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
        history.append({"role": msg.role, "content": text})
    return history


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
        shortened = payload #_shorten_json_strings(payload)
        text = json.dumps(shortened, ensure_ascii=False)
    except TypeError:
        text = repr(payload)
    text = " ".join(text.split())
    return payload #_shorten_middle(text, head=head, tail=tail)


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
    conversation_id = memory_backend.new_conversation_id()
    history = _conversation_history(conversation_id, persona)
    state = {"conversation_id": conversation_id, "persona": persona, "history": history, "event_log": []}
    _append_event_log(state, "Session initialized.")
    return state


def _persona_box_value(persona: str) -> str:
    """Return the persona text suitable for the editable textbox."""

    lines = (persona or "").splitlines()
    while lines and lines[-1].strip() in {TOOL_PROTOCOL_HINT.strip(), ALLOWLIST_LINE.strip()}:
        lines.pop()
    return "\n".join(lines).strip()


def _rehydrate_state() -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Load persisted state for a freshly connected client session."""

    state = _initial_state()
    persona_box_value = _persona_box_value(state.get("persona", DEFAULT_PERSONA))
    return state, state.get("history", []), persona_box_value

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

def summarize_text(file_text: str) -> Tuple[str, str]:
    sys = {"role": "system", "content": "Summarize the user's provided file text clearly in 5-8 bullets and include any commands, paths, or todos verbatim."}
    user = {"role": "user", "content": file_text}
    ret = engine.chat([sys, user])
    if isinstance(ret, dict):
        return ret.get("text", ""), json.dumps(ret.get("meta", {}), indent=2)[:4000]
    return str(ret), ""

def on_user(message: str, state: Dict[str, Any]):
    state = dict(state or {})
    state.setdefault("event_log", [])

    preview_output: Any = gr.update()
    user_output: Any = gr.update()

    def _snapshot(*, preview: Any | None = None, user: Any | None = None):
        nonlocal preview_output, user_output
        if preview is not None:
            preview_output = preview
        if user is not None:
            user_output = user
        return state, preview_output, _event_log_messages(state), user_output

    def _log(message_text: str):
        _append_event_log(state, message_text)
        return _snapshot()

    user_summary = message #_shorten_text(message or "", limit=120) or "(empty)"
    yield _log(f"User input received: {user_summary}")
    history = list(state.get("history", []))
    conversation_id = state.get("conversation_id") or memory_backend.new_conversation_id()
    existing_persona = state.get("persona")
    persona_seed = _append_persona_metadata(existing_persona or _initial_persona_seed())

    history.append({"role": "user", "content": message})
    state.update({"history": history, "conversation_id": conversation_id, "persona": persona_seed})
    memory_backend.add_message(conversation_id, "user", {"text": message})
    intent, args = detect_intent(message)
    args_desc = _format_args_for_log(args)
    if intent == "chat":
        yield _log("Detected chat intent (no direct command).")
    else:
        detail = f" ({args_desc})" if args_desc else ""
        yield _log(f"Detected command intent '{intent}'{detail}.")
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
            history.append({"role": "assistant", "content": assistant})
            state["history"] = history
            memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "list_dir"})
            preview_value = assistant if isinstance(assistant, str) else json.dumps(assistant, indent=2)
            yield _snapshot(preview=preview_value, user="")
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
            history.append({"role": "assistant", "content": assistant})
            state["history"] = history
            memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "read_text_file", "preview": preview_text})
            yield _snapshot(preview=preview_text, user="")
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
                summary, meta = summarize_text(preview_text)
                assistant = summary
                yield _log("Summarize succeeded.")
                if meta:
                    yield _log(f"Summarize meta captured ({len(meta)} chars).")
            history.append({"role": "assistant", "content": assistant})
            state["history"] = history
            memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "summarize", "preview": preview_text})
            yield _snapshot(preview=preview_text, user="")
            return

        if intent == "locate":
            q = args.get("query", "").strip()
            if not q:
                assistant = "Usage: locate <name>"
                history.append({"role": "assistant", "content": assistant})
                state["history"] = history
                memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "locate"})
                yield _log("Locate command missing query.")
                yield _snapshot(preview="", user="")
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
            history.append({"role": "assistant", "content": assistant})
            state["history"] = history
            memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "locate"})
            yield _snapshot(preview="", user="")
            return

        t0 = time.perf_counter()
        yield _log("Building context for chat request.")
        ctx_messages = context_builder.build_context(conversation_id, message, persona_seed=persona_seed)
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
        history.append({"role": "assistant", "content": assistant_display})
        state["history"] = history

        stored_payload: Dict[str, Any] = {"text": assistant_text, "display": assistant_display, "meta": meta}
        if tool_used:
            stored_payload["tool"] = tool_used
            if tool_result is not None:
                try:
                    json.dumps(tool_result)
                    stored_payload["tool_result"] = tool_result
                except TypeError:
                    stored_payload["tool_result"] = str(tool_result)

        memory_backend.add_message(conversation_id, "assistant", stored_payload)

        status = meta.get("status") if isinstance(meta, dict) else None
        if meta.get("error"):
            yield _log(f"Model metadata reported error: {str(meta['error'])}")
        if status:
            yield _log(f"Model call completed in {elapsed:.2f}s with status {status}.")
        else:
            yield _log(f"Model call completed in {elapsed:.2f}s.")
        yield _log(f"Assistant response prepared in {elapsed:.2f}s.")

        yield _snapshot(preview=preview_text, user="")
        return

    except Exception:
        err = traceback.format_exc(limit=3)
        history.append({"role": "assistant", "content": f"Error: {err}"})
        state["history"] = history
        memory_backend.add_message(conversation_id, "assistant", {"text": err, "error": True})
        yield _log(f"Exception raised: {err}")
        yield _snapshot(preview="", user="")
        return

def on_persona_change(new_seed, state):
    state = dict(state or {})
    state.setdefault("event_log", [])
    history = list(state.get("history", []))
    persona = _append_persona_metadata(new_seed)
    if history and history[0].get("role") == "system":
        history[0]["content"] = persona
    else:
        history.insert(0, {"role": "system", "content": persona})
    state.update({"history": history, "persona": persona})
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

with gr.Blocks(title="Local Chat (Files)") as demo:
    gr.Markdown("""
    # HomeAI
    """)

    initial_state = _initial_state()
    state = gr.State(value=initial_state)
    with gr.Row():
        with gr.Column():
            chat = _safe_component(
                gr.Chatbot,
                value=initial_state["history"],
                height=360,
                type="messages",
                live=True,
                optional_keys=("live", "bubble_full_width"),
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

    demo.load(_rehydrate_state, inputs=None, outputs=[state, chat, persona_box])
    demo.load(lambda s: _event_log_messages(s), inputs=state, outputs=log_box)

    send_btn.click(on_user, inputs=[user_box, state], outputs=[state, preview, log_box, user_box]).then(lambda s: s["history"], inputs=state, outputs=chat)
    user_box.submit(on_user, inputs=[user_box, state], outputs=[state, preview, log_box, user_box]).then(lambda s: s["history"], inputs=state, outputs=chat)
    persona_box.change(on_persona_change, inputs=[persona_box, state], outputs=[state]).then(lambda s: s["history"], inputs=state, outputs=chat)
    persona_preset.change(apply_preset, inputs=[persona_preset, state], outputs=[state, persona_box]).then(lambda s: s["history"], inputs=state, outputs=chat)

    demo.load(lambda s: list(s.get("history", [])), inputs=state, outputs=chat)



if __name__ == "__main__":
    _init_memory_backend()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
