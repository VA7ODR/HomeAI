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

import os
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple

import gradio as gr
import requests

from context_memory import ContextBuilder, LocalJSONMemoryBackend

MODEL = os.getenv("TBH_OLLAMA_MODEL", "gpt-oss:20b")
HOST = os.getenv("TBH_OLLAMA_HOST", "http://192.168.1.100:11434")
BASE = Path(os.getenv("TBH_ALLOWLIST_BASE", str(Path.home()))).resolve()
DEFAULT_PERSONA = os.getenv("TBH_PERSONA", (
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

class OllamaEngine:
    def __init__(self, model: str = MODEL, host: str = HOST):
        if not host.startswith("http://") and not host.startswith("https://"):
            host = "http://" + host
        self.model, self.host = model, host

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        payload_chat = {"model": self.model, "messages": msgs, "stream": False}
        url_chat = f"{self.host}/api/chat"

        t0 = time.perf_counter()
        r = requests.post(url_chat, json=payload_chat, timeout=120)
        used = "chat"
        prompt = None

        if r.status_code == 404:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            payload_gen = {"model": self.model, "prompt": prompt, "stream": False}
            r = requests.post(f"{self.host}/api/generate", json=payload_gen, timeout=120)
            used = "generate"

        elapsed = time.perf_counter() - t0
        r.raise_for_status()
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
            "request": payload_chat if used == "chat" else {"model": self.model, "prompt": prompt, "stream": False},
            "response": data,
        }
        return {"text": text, "meta": meta}

engine = OllamaEngine()
memory_backend = LocalJSONMemoryBackend()
context_builder = ContextBuilder(memory_backend)


def _initial_persona_seed() -> str:
    return f"{DEFAULT_PERSONA}\nAllowlist base is: {BASE}. Keep outputs concise unless asked."


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

def detect_intent(text: str) -> Tuple[str, Dict[str, str]]:
    t = text.strip()
    low = t.lower()
    if low.startswith("browse ") or low.startswith("list ") or low.startswith("ls "):
        path = t.split(" ", 1)[1].strip() if " " in t else "."
        return "browse", {"path": path}
    if low.startswith("read "):
        path = t.split(" ", 1)[1].strip()
        return "read", {"path": path}
    if low.startswith("summarize ") or low.startswith("summarise "):
        path = t.split(" ", 1)[1].strip()
        return "summarize", {"path": path}
    if low.startswith("locate ") or low.startswith("find "):
        query = t.split(" ", 1)[1].strip()
        return "locate", {"query": query}
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
    persona_seed = state.get("persona") or _initial_persona_seed()

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
            return state, (assistant if isinstance(assistant, str) else json.dumps(assistant, indent=2)), json.dumps({"tool":"list_dir","args":args,"result_count":res.get("count")}, indent=2)

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
            return state, preview_text, json.dumps({"tool":"read_text_file","args":args,"ok": "error" not in r}, indent=2)

        if intent == "summarize":
            p = args.get("path", "")
            r = read_text_file(p)
            if "error" in r:
                assistant = r["error"]
                preview_text = ""
                log = json.dumps({"tool":"read_text_file","args":args,"ok":False}, indent=2)
            else:
                preview_text = r.get("text", "")
                summary, meta = summarize_text(preview_text)
                assistant = summary
                log = meta
            history.append({"role": "assistant", "content": assistant})
            state["history"] = history
            memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "summarize", "preview": preview_text})
            return state, preview_text, log

        if intent == "locate":
            q = args.get("query", "").strip()
            if not q:
                assistant = "Usage: locate <name>"
                history.append({"role": "assistant", "content": assistant})
                state["history"] = history
                memory_backend.add_message(conversation_id, "assistant", {"text": assistant, "tool": "locate"})
                return state, "", "{}"
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
            return state, "", json.dumps({"tool":"locate","query":q,"count":res.get("count")}, indent=2)

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
        assistant = f"{reply}\n\n— local in {elapsed:.2f}s"
        history.append({"role": "assistant", "content": assistant})
        state["history"] = history
        memory_backend.add_message(conversation_id, "assistant", {"text": reply, "display": assistant, "meta": meta})
        return state, "", json.dumps(meta, indent=2)[:4000]

    except Exception:
        err = traceback.format_exc(limit=3)
        history.append({"role": "assistant", "content": f"Error: {err}"})
        state["history"] = history
        memory_backend.add_message(conversation_id, "assistant", {"text": err, "error": True})
        return state, "", json.dumps({"error": err}, indent=2)[:4000]

def on_persona_change(new_seed, state):
    state = dict(state or {})
    history = list(state.get("history", []))
    persona = new_seed + "\n" + (f"Allowlist base is: {BASE}. Keep outputs concise unless asked.")
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
    gr.Markdown(f"""
    # Local Chat (Ollama)
    **Allowlist base:** `{BASE}`
    """)

    with gr.Row():
        persona_preset = gr.Dropdown(label="Persona preset", choices=["Dax mentor", "Code reviewer", "Ham-radio Elmer", "Stoic coach", "LCARS formal", "Dax Self"], value="Dax mentor", scale=0)
        persona_box = gr.Textbox(label="Personality seed", value=DEFAULT_PERSONA, lines=3, scale=2)

    state = gr.State(value=_initial_state())

    chat = gr.Chatbot(height=520, type="messages")
    user_box = gr.Textbox(label="Message", placeholder="chat | browse <path> | read <file> | summarize <file> | locate <name>")
    send_btn = gr.Button("Send", variant="primary")
    preview = gr.Textbox(label="File preview (on read/summarize)", lines=18)
    log_box = gr.Textbox(label="LLM / Tool Log", lines=12)

    send_btn.click(on_user, inputs=[user_box, state], outputs=[state, preview, log_box]).then(lambda s: s["history"], inputs=state, outputs=chat)
    user_box.submit(on_user, inputs=[user_box, state], outputs=[state, preview, log_box]).then(lambda s: s["history"], inputs=state, outputs=chat)
    persona_box.change(on_persona_change, inputs=[persona_box, state], outputs=[state]).then(lambda s: s["history"], inputs=state, outputs=chat)
    persona_preset.change(apply_preset, inputs=[persona_preset, state], outputs=[state, persona_box]).then(lambda s: s["history"], inputs=state, outputs=chat)



if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
