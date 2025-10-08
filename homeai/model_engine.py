from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any, Dict, Generator, Iterable, List, Optional

import requests

from . import config


class LocalModelEngine:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        selected_model = model or config.MODEL
        selected_host = host or config.HOST
        if not selected_host.startswith("http://") and not selected_host.startswith("https://"):
            selected_host = "http://" + selected_host
        self.model, self.host = selected_model, selected_host
        session_factory = getattr(requests, "Session", None)
        if callable(session_factory):
            self._session = session_factory()
            self._close_session = getattr(self._session, "close", lambda: None)
        else:
            # Test environments monkeypatch ``requests`` with a lightweight stub
            # that only exposes ``post``. Reuse that module directly so the engine
            # keeps working without a full ``requests.Session`` implementation.
            self._session = requests
            self._close_session = lambda: None

    def close(self) -> None:
        self._close_session()

    def __enter__(self) -> "LocalModelEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _prepare_chat_payload(self, messages: List[Dict[str, str]], *, stream: bool) -> Dict[str, Any]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        return {"model": self.model, "messages": msgs, "stream": stream}

    def _prepare_generate_payload(self, messages: List[Dict[str, str]], *, stream: bool) -> Dict[str, Any]:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return {"model": self.model, "prompt": prompt, "stream": stream}

    def _iter_sse_payload(self, response) -> Iterable[str]:
        iter_lines = getattr(response, "iter_lines", None)
        if not callable(iter_lines):
            return []
        return iter_lines(decode_unicode=True)

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload_chat = self._prepare_chat_payload(messages, stream=False)
        url_chat = f"{self.host}/api/chat"

        used = "chat"
        request_payload: Dict[str, Any] = payload_chat
        t0 = time.perf_counter()

        try:
            response = self._session.post(url_chat, json=payload_chat, timeout=120)
        except requests.exceptions.RequestException as exc:
            elapsed = time.perf_counter() - t0
            meta = {
                "endpoint": used,
                "error": f"{exc.__class__.__name__}: {exc}",
                "elapsed_sec": round(elapsed, 3),
                "request": request_payload,
            }
            return {"text": f"Model request failed while calling {url_chat}: {exc}", "meta": meta}

        if response.status_code == 404:
            payload_gen = self._prepare_generate_payload(messages, stream=False)
            used = "generate"
            request_payload = payload_gen
            try:
                response = self._session.post(
                    f"{self.host}/api/generate", json=payload_gen, timeout=120
                )
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
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            try:
                response_body: Any = response.json()
                response_preview = json.dumps(response_body, ensure_ascii=False)[:4000]
            except ValueError:
                response_body = None
                response_preview = (response.text or "")[:4000]

            meta = {
                "endpoint": used,
                "status": response.status_code,
                "elapsed_sec": round(elapsed, 3),
                "request": request_payload,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            if response_body is not None:
                meta["response"] = response_body
            else:
                meta["response_text"] = response_preview

            reason = getattr(response, "reason", "") or ""
            details = response_preview or reason or "No response body."
            return {
                "text": f"Model endpoint {used} returned HTTP {response.status_code}: {details}",
                "meta": meta,
            }

        try:
            data = response.json()
        except Exception:
            data = {"raw": response.text[:4000]}

        text = ""
        if isinstance(data, dict) and isinstance(data.get("message"), dict):
            text = data["message"].get("content", "") or ""
        if not text and isinstance(data, dict):
            text = data.get("response", "") or ""

        meta = {
            "endpoint": used,
            "status": response.status_code,
            "elapsed_sec": round(elapsed, 3),
            "request": request_payload,
            "response": data,
        }
        return {"text": text, "meta": meta}

    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator[Dict[str, Any], None, None]:
        payload_chat = self._prepare_chat_payload(messages, stream=True)
        url_chat = f"{self.host}/api/chat"
        used = "chat"
        request_payload: Dict[str, Any] = payload_chat
        t0 = time.perf_counter()

        try:
            response = self._session.post(
                url_chat, json=payload_chat, timeout=120, stream=True
            )
        except requests.exceptions.RequestException:
            yield {"event": "complete", "data": self.chat(messages)}
            return

        if response.status_code == 404:
            payload_gen = self._prepare_generate_payload(messages, stream=True)
            used = "generate"
            request_payload = payload_gen
            try:
                response = self._session.post(
                    f"{self.host}/api/generate", json=payload_gen, timeout=120, stream=True
                )
            except requests.exceptions.RequestException:
                yield {"event": "complete", "data": self.chat(messages)}
                return

        if response.status_code >= 400:
            response.close()
            yield {"event": "complete", "data": self.chat(messages)}
            return

        content_type = response.headers.get("content-type", "") if hasattr(response, "headers") else ""
        if "text/event-stream" not in content_type.lower():
            body = None
            try:
                body = response.json()
            except Exception:
                pass
            response.close()
            if body is not None:
                meta = {
                    "endpoint": used,
                    "status": response.status_code,
                    "elapsed_sec": round(time.perf_counter() - t0, 3),
                    "request": request_payload,
                    "response": body,
                }
                text = ""
                if isinstance(body, dict) and isinstance(body.get("message"), dict):
                    text = body["message"].get("content", "") or ""
                if not text and isinstance(body, dict):
                    text = body.get("response", "") or ""
                yield {"event": "complete", "data": {"text": text, "meta": meta}}
                return
            yield {"event": "complete", "data": self.chat(messages)}
            return

        chunks: List[str] = []
        last_payload: Optional[Any] = None
        meta_extra: Dict[str, Any] = {}
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        tool_call_ids: Dict[str, int] = {}
        next_tool_call_index = 0

        def _iter_tool_call_lists(obj: Any) -> Iterable[List[Dict[str, Any]]]:
            if not isinstance(obj, dict):
                return []

            def _walk(node: Any) -> Iterable[List[Dict[str, Any]]]:
                if not isinstance(node, dict):
                    return
                tool_calls = node.get("tool_calls")
                if isinstance(tool_calls, list):
                    yield tool_calls
                for key in ("message", "delta"):
                    child = node.get(key)
                    if isinstance(child, dict):
                        yield from _walk(child)
                choices = node.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if isinstance(choice, dict):
                            yield from _walk(choice)

            return list(_walk(obj))

        def _merge_tool_calls(calls: List[Dict[str, Any]]) -> None:
            nonlocal next_tool_call_index
            for call in calls:
                if not isinstance(call, dict):
                    continue
                idx_value = call.get("index")
                idx: Optional[int]
                if isinstance(idx_value, int):
                    idx = idx_value
                else:
                    try:
                        idx = int(idx_value)
                    except (TypeError, ValueError):
                        idx = None
                call_id = call.get("id")
                if idx is None and isinstance(call_id, str) and call_id in tool_call_ids:
                    idx = tool_call_ids[call_id]
                if idx is None and isinstance(call_id, str):
                    idx = tool_call_ids[call_id] = next_tool_call_index
                    next_tool_call_index += 1
                if idx is None:
                    idx = next_tool_call_index
                    next_tool_call_index += 1
                entry = tool_calls_acc.setdefault(idx, {"index": idx})
                if isinstance(call_id, str):
                    entry["id"] = call_id
                    tool_call_ids[call_id] = idx
                call_type = call.get("type")
                if isinstance(call_type, str):
                    entry["type"] = call_type
                function_payload = call.get("function")
                if isinstance(function_payload, dict):
                    entry_function = entry.setdefault("function", {})
                    name = function_payload.get("name")
                    if isinstance(name, str) and name:
                        entry_function["name"] = name
                    if "arguments" in function_payload:
                        args_val = function_payload["arguments"]
                        if isinstance(args_val, str):
                            replace_existing = False
                            if args_val:
                                stripped = args_val.strip()
                                if stripped:
                                    try:
                                        json.loads(args_val)
                                    except json.JSONDecodeError:
                                        replace_existing = False
                                    else:
                                        replace_existing = True
                            if replace_existing or not isinstance(
                                entry_function.get("arguments"), str
                            ):
                                entry_function["arguments"] = args_val
                            else:
                                entry_function["arguments"] += args_val
                        else:
                            entry_function["arguments"] = args_val

        try:
            for raw_line in self._iter_sse_payload(response):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if not line.startswith("data:"):
                    continue
                payload_text = line[len("data:") :].strip()
                if not payload_text:
                    continue
                if payload_text == "[DONE]":
                    break
                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    continue
                last_payload = payload
                for call_list in _iter_tool_call_lists(payload):
                    _merge_tool_calls(call_list)
                delta: str = ""
                if isinstance(payload, dict):
                    if isinstance(payload.get("message"), dict):
                        delta = payload["message"].get("content", "") or ""
                    elif isinstance(payload.get("delta"), dict):
                        delta = (
                            payload["delta"].get("content")
                            or payload["delta"].get("text")
                            or ""
                        )
                    elif isinstance(payload.get("response"), str):
                        delta = payload.get("response", "")
                    if payload.get("meta") and isinstance(payload["meta"], dict):
                        meta_extra.update(payload["meta"])
                    if payload.get("error") and isinstance(payload["error"], str):
                        meta_extra.setdefault("error", payload["error"])
                if delta:
                    chunks.append(delta)
                    yield {"event": "delta", "data": delta}
                if isinstance(payload, dict) and payload.get("done"):
                    break
        finally:
            response.close()

        elapsed = time.perf_counter() - t0
        meta = {
            "endpoint": used,
            "status": response.status_code,
            "elapsed_sec": round(elapsed, 3),
            "request": request_payload,
        }
        if meta_extra:
            meta.update(meta_extra)
        aggregated_response: Optional[Dict[str, Any]] = None
        if isinstance(last_payload, dict):
            aggregated_response = deepcopy(last_payload)
        elif tool_calls_acc:
            aggregated_response = {}
        if aggregated_response is not None and tool_calls_acc:
            ordered_calls = [
                deepcopy(tool_calls_acc[idx]) for idx in sorted(tool_calls_acc.keys())
            ]
            message_payload = aggregated_response.get("message")
            if not isinstance(message_payload, dict):
                message_payload = {}
            message_payload["tool_calls"] = ordered_calls
            aggregated_response["message"] = message_payload
            aggregated_response["tool_calls"] = ordered_calls
        if aggregated_response is not None:
            meta["response"] = aggregated_response
        elif last_payload is not None:
            meta.setdefault("response", last_payload)

        text = "".join(chunks)
        yield {"event": "complete", "data": {"text": text, "meta": meta, "raw": last_payload}}


__all__ = ["LocalModelEngine"]
