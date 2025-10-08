from __future__ import annotations

import json
import time
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
        if last_payload is not None:
            meta.setdefault("response", last_payload)

        text = "".join(chunks)
        yield {"event": "complete", "data": {"text": text, "meta": meta, "raw": last_payload}}


__all__ = ["LocalModelEngine"]
