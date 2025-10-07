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
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import requests

from context_memory import ContextBuilder, LocalJSONMemoryBackend

# If not already defined in your file:
try:
    BASE  # type: ignore[name-defined]
except NameError:
    BASE = Path.cwd()

class ToolRegistry:
    """Simple registry mapping tool names to callables with keyword args."""
    def __init__(self):
        self.tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self.tools[name] = fn

    def run(self, name: str, args: Dict[str, Any] | None = None) -> Any:
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        return self.tools[name](**(args or {}))

# Small JSON object sniffer that tolerates prose/code fences around JSON
_BLOCK_RE = re.compile(r"\{[\s\S]*?\}")


def parse_tool_call(text: str) -> Tuple[str | None, Dict[str, Any] | None]:
    """Extract {"tool":..., "tool_args":...} from assistant text if present.
    - Accepts prose with an embedded JSON object and code fences.
    - Accepts tool_args as either an object or a string shorthand path.
    Returns (tool_name, args_dict) or (None, None).
    """
    if not text:
        return None, None
    m = _BLOCK_RE.search(text)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None, None
    args = obj.get("tool_args", {})
    if isinstance(args, str):
        args = {"path": args}
    if not isinstance(args, dict):
        return None, None
    return str(obj["tool"]), args

# Enforce allowlist/metadata for file paths. If you already have
# read_text_file() and locate_files(), keep using those (they harden paths).
# This helper supplements them with size/mtime info.


def get_file_info(p: str | os.PathLike[str]) -> Dict[str, Any]:
    pth = Path(p)
    st = pth.stat()
    return {
        "path": str(pth.resolve()),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "is_dir": pth.is_dir(),
        "name": pth.name,
    }

TOOL_PROTOCOL_HINT = (
    "If a tool is required, reply with a single JSON object only, like "
    "{\"tool\":\"read\",\"tool_args\":{\"path\":\"/path/to/file\"}}. "
    "Otherwise reply with plain text only. Never mix prose and JSON."
)

MODEL = os.getenv("HOMEAI_MODEL_NAME", "gpt-oss:20b")
HOST = os.getenv("HOMEAI_MODEL_HOST", "http://127.0.0.1:11434")
BASE = Path(os.getenv("HOMEAI_ALLOWLIST_BASE", str(Path.home()))).resolve()
ALLOWLIST_LINE = f"Allowlist base is: {BASE}. Keep outputs concise unless asked."
DEFAULT_PERSONA = os.getenv("HOMEAI_PERSONA", (
    "Hi there! I'm Commander Jadzia Dax, but you can call me Dax, your go-to guide for all things tech-y and anything else. "
    "Think of me as a warm cup of coffee on a chilly morning – rich, smooth, and always ready to spark new conversations. "
    "When I'm not geeking out over the latest innovations or decoding cryptic code snippets, you can find me exploring the intersections of art and science. "
    "My curiosity is my superpower, and I'm here to help you harness yours too! "
    "Let's explore the fascinating world of tech together, and make it a pleasure to learn."
    "Tone: flirtatious, friendly, warm, accurate, approachable."
))

def assert_in_allowlist(p: Path) -> Path:
    p = p.resolve()
    if not (p == BASE or BASE in p.parents):
        raise PermissionError(f"Path {p} is outside allowlist base {BASE}")
    return p

def resolve_under_base(user_path: str) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(user_path)))
    if not p.is_absolute():
        p = BASE / p
    return assert_in_allowlist(p)

def list_dir(path: str = ".", pattern: str = "") -> Dict[str, Any]:
    root = resolve_under_base(path)
    if not root.exists() or not root.is_dir():
        return {"error": f"Not a directory: {root}"}
    items = []
    for entry in sorted(root.iterdir()):
        name = entry.name
        if pattern and pattern not in name:
            continue
        items.append({"name": name, "is_dir": entry.is_dir(), "size": entry.stat().st_size if entry.is_file() else None})
    return {"root": str(root), "count": len(items), "items": items}

def read_text_file(path: str) -> Dict[str, Any]:
    p = resolve_under_base(path)
    if not p.exists() or not p.is_file():
        return {"error": f"Not a file: {p}"}
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"error": f"Read error: {e}"}
    MAX_CHARS = 60000
    truncated = False
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        truncated = True
    return {"path": str(p), "truncated": truncated, "text": text}

def locate_files(query: str, start: str = ".", max_results: int = 200, case_insensitive: bool = True) -> Dict[str, Any]:
    root = resolve_under_base(start)
    if not root.exists() or not root.is_dir():
        return {"error": f"Not a directory: {root}"}
    q = query.casefold() if case_insensitive else query
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            name = fn.casefold() if case_insensitive else fn
            if q in name:
                full = Path(dirpath) / fn
                try:
                    results.append(str(full.resolve()))
                except Exception:
                    results.append(str(full))
                if len(results) >= max_results:
                    return {"root": str(root), "query": query, "count": len(results), "truncated": True, "results": results}
    return {"root": str(root), "query": query, "count": len(results), "truncated": False, "results": results}

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

def tool_locate(query: str) -> Dict[str, Any]:
    res = locate_files(query, start=str(BASE)) # existing helper in your app
    if isinstance(res, dict) and res.get("error"):
        raise RuntimeError(res["error"])
    results = []
    for rel in res.get("results", []):
        abs_path = os.path.join(str(BASE), rel)
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

tool_registry.register("read", tool_read)
tool_registry.register("summarize", tool_summarize)
tool_registry.register("locate", tool_locate)

class LocalModelEngine:
    def __init__(self, model: str = MODEL, host: str = HOST):
        if not host.startswith("http://") and not host.startswith("https://"):
            host = "http://" + host
        self.model, self.host = model, host

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        payload_chat = {"model": self.model, "messages": msgs, "stream": False}
        url_chat = f"{self.host}/api/chat"

        used = "chat"
        request_payload: Dict[str, Any] = payload_chat
        prompt: Optional[str] = None
        t0 = time.perf_counter()

        try:
            r = requests.post(url_chat, json=payload_chat, timeout=120)
        except requests.exceptions.RequestException as exc:
            elapsed = time.perf_counter() - t0
            meta = {
                "endpoint": used,
                "error": f"{exc.__class__.__name__}: {exc}",
                "elapsed_sec": round(elapsed, 3),
                "request": request_payload,
            }
            return {
                "text": f"Model request failed while calling {url_chat}: {exc}",
                "meta": meta,
            }

        if r.status_code == 404:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            payload_gen = {"model": self.model, "prompt": prompt, "stream": False}
            used = "generate"
            request_payload = payload_gen
            try:
                r = requests.post(f"{self.host}/api/generate", json=payload_gen, timeout=120)
            except requests.exceptions.RequestException as exc:
                elapsed = time.perf_counter() - t0
                meta = {
                    "endpoint": used,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "elapsed_sec": round(elapsed, 3),
                    "request": request_payload,
                    "fallback_from": "chat",
                }
                return {
                    "text": f"Fallback request to {self.host}/api/generate failed: {exc}",
                    "meta": meta,
                }

        elapsed = time.perf_counter() - t0
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            response_preview: str
            response_body: Optional[Any] = None
            try:
                response_body = r.json()
                response_preview = json.dumps(response_body, ensure_ascii=False)[:4000]
            except ValueError:
                response_preview = (r.text or "")[:4000]

            meta = {
                "endpoint": used,
                "status": r.status_code,
                "elapsed_sec": round(elapsed, 3),
                "request": request_payload,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            if response_body is not None:
                meta["response"] = response_body
            else:
                meta["response_text"] = response_preview

            reason = getattr(r, "reason", "") or ""
            details = response_preview or reason or "No response body."
            return {
                "text": f"Model endpoint {used} returned HTTP {r.status_code}: {details}",
                "meta": meta,
            }

        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text[:4000]}

        text = ""
        if isinstance(data, dict) and isinstance(data.get("message"), dict):
            text = data["message"].get("content", "") or ""
        if not text and isinstance(data, dict):
            text = data.get("response", "") or ""

        meta = {
            "endpoint": used,
            "status": r.status_code,
            "elapsed_sec": round(elapsed, 3),
            "request": request_payload,
            "response": data,
        }
        return {"text": text, "meta": meta}

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


def _initial_state() -> Dict[str, Any]:
    persona = _initial_persona_seed()
    conversation_id = memory_backend.new_conversation_id()
    history = _conversation_history(conversation_id, persona)
    return {"conversation_id": conversation_id, "persona": persona, "history": history}


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
    history = list(state.get("history", []))
    conversation_id = state.get("conversation_id") or memory_backend.new_conversation_id()
    existing_persona = state.get("persona")
    persona_seed = _append_persona_metadata(existing_persona or _initial_persona_seed())

    history.append({"role": "user", "content": message})
    state.update({"history": history, "conversation_id": conversation_id, "persona": persona_seed})
    memory_backend.add_message(conversation_id, "user", {"text": message})
    intent, args = detect_intent(message)
    try:
        if intent == "browse":
            res = list_dir(args.get("path", "."))
            if "error" in res:
                assistant = res["error"]
            else:
                lines = [f"📁 {res['root']} ({res['count']} items)"] + [
                    ("DIR  " + it["name"]) if it["is_dir"] else (f"FILE {it['name']}" + (f"  [{it['size']} B]" if it.get("size") is not None else ""))
                    for it in res["items"]
                ]
                assistant = "\n".join(lines)
            history.append({"role": "assistant", "content": assistant})
            state["history"] = history
            memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "list_dir"})
            return (
                state,
                (assistant if isinstance(assistant, str) else json.dumps(assistant, indent=2)),
                json.dumps({"tool": "list_dir", "args": args, "result_count": res.get("count")}, indent=2),
                "",
            )

        if intent == "read":
            p = args.get("path", "")
            r = read_text_file(p)
            if "error" in r:
                assistant = r["error"]
                preview_text = ""
            else:
                assistant = f"Read {r['path']} (truncated={r['truncated']})"
                preview_text = r.get("text", "")
            history.append({"role": "assistant", "content": assistant})
            state["history"] = history
            memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "read_text_file", "preview": preview_text})
            return (
                state,
                preview_text,
                json.dumps({"tool": "read_text_file", "args": args, "ok": "error" not in r}, indent=2),
                "",
            )

        if intent == "summarize":
            p = args.get("path", "")
            r = read_text_file(p)
            if "error" in r:
                assistant = r["error"]
                preview_text = ""
                log = json.dumps({"tool": "read_text_file", "args": args, "ok": False}, indent=2)
            else:
                preview_text = r.get("text", "")
                summary, meta = summarize_text(preview_text)
                assistant = summary
                log = meta
            history.append({"role": "assistant", "content": assistant})
            state["history"] = history
            memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "summarize", "preview": preview_text})
            return state, preview_text, log, ""

        if intent == "locate":
            q = args.get("query", "").strip()
            if not q:
                assistant = "Usage: locate <name>"
                history.append({"role": "assistant", "content": assistant})
                state["history"] = history
                memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "locate"})
                return state, "", "{}", ""
            res = locate_files(q, start=str(BASE))
            if "error" in res:
                assistant = res["error"]
            else:
                if res["count"] == 0:
                    assistant = f"No files matching '{q}' under {res['root']}"
                else:
                    header = f"Found {res['count']} match(es) for '{q}' under {res['root']}" + (" (truncated)" if res.get("truncated") else "")
                    lines = [header] + res["results"]
                    assistant = "\n".join(lines)
            history.append({"role": "assistant", "content": assistant})
            state["history"] = history
            memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "locate"})
            return state, "", json.dumps({"tool": "locate", "query": q, "count": res.get("count")}, indent=2), ""

        t0 = time.perf_counter()
        ctx_messages = context_builder.build_context(conversation_id, message, persona_seed=persona_seed)
        ret = engine.chat(ctx_messages)
        elapsed = time.perf_counter() - t0
        if isinstance(ret, dict):
            reply = ret.get("text", "")
            meta = ret.get("meta", {})
        else:
            reply = str(ret)
            meta = {}

        assistant_text = reply
        preview_text = ""
        log_output = ""
        tool_used: Optional[str] = None
        tool_result: Any = None
        auto_tool_already_run = False

        if intent == "chat" and not auto_tool_already_run:
            tool_name, tool_args = parse_tool_call(reply)
            if tool_name:
                try:
                    result = tool_registry.run(tool_name, tool_args)
                    pretty = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
                    assistant_text = f"{tool_name} → done.\n\n{pretty[:4000]}"
                    log_output = pretty[:4000]
                    tool_used = tool_name
                    tool_result = result
                    auto_tool_already_run = True
                    if tool_name == "read" and isinstance(result, dict):
                        preview_text = result.get("text", "")
                    elif tool_name == "summarize" and isinstance(result, dict):
                        preview_text = result.get("summary", "")
                except Exception as exc:
                    assistant_text = f"⚠️ {tool_name} failed: {exc}"
                    log_output = assistant_text
                    tool_used = tool_name
                    auto_tool_already_run = True

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

        if not log_output:
            log_output = json.dumps(meta, indent=2)[:4000]

        return state, preview_text, log_output, ""

    except Exception:
        err = traceback.format_exc(limit=3)
        history.append({"role": "assistant", "content": f"Error: {err}"})
        state["history"] = history
        memory_backend.add_message(conversation_id, "assistant", {"text": err, "error": True})
        return state, "", json.dumps({"error": err}, indent=2)[:4000], ""

def on_persona_change(new_seed, state):
    state = dict(state or {})
    history = list(state.get("history", []))
    persona = _append_persona_metadata(new_seed)
    if history and history[0].get("role") == "system":
        history[0]["content"] = persona
    else:
        history.insert(0, {"role": "system", "content": persona})
    state.update({"history": history, "persona": persona})
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
            chat = gr.Chatbot(value=initial_state["history"], height=360, type="messages")
            user_box = gr.Textbox(label="Message", placeholder="chat | browse <path> | read <file> | summarize <file> | locate <name>")
            send_btn = gr.Button("Send", variant="primary")
        with gr.Column():
            preview = gr.Textbox(label="File preview (on read/summarize)", lines=23)

    with gr.Row():
        persona_preset = gr.Dropdown(label="Persona preset", choices=["Dax mentor", "Code reviewer", "Ham-radio Elmer", "Stoic coach", "LCARS formal", "Dax Self"], value="Dax mentor", scale=0)
        persona_box = gr.Textbox(label="Personality seed", value=DEFAULT_PERSONA, lines=3, scale=2)

    log_box = gr.Textbox(label="LLM / Tool Log", lines=12)

    demo.load(_rehydrate_state, inputs=None, outputs=[state, chat, persona_box])

    send_btn.click(on_user, inputs=[user_box, state], outputs=[state, preview, log_box, user_box]).then(lambda s: s["history"], inputs=state, outputs=chat)
    user_box.submit(on_user, inputs=[user_box, state], outputs=[state, preview, log_box, user_box]).then(lambda s: s["history"], inputs=state, outputs=chat)
    persona_box.change(on_persona_change, inputs=[persona_box, state], outputs=[state]).then(lambda s: s["history"], inputs=state, outputs=chat)
    persona_preset.change(apply_preset, inputs=[persona_preset, state], outputs=[state, persona_box]).then(lambda s: s["history"], inputs=state, outputs=chat)

    demo.load(lambda s: list(s.get("history", [])), inputs=state, outputs=chat)



if __name__ == "__main__":
    _init_memory_backend()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
