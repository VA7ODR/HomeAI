from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests

from . import config


class LocalModelEngine:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        selected_model = model or config.MODEL
        selected_host = host or config.HOST
        if not selected_host.startswith("http://") and not selected_host.startswith("https://"):
            selected_host = "http://" + selected_host
        self.model, self.host = selected_model, selected_host

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        payload_chat = {"model": self.model, "messages": msgs, "stream": False}
        url_chat = f"{self.host}/api/chat"

        used = "chat"
        request_payload: Dict[str, Any] = payload_chat
        t0 = time.perf_counter()

        try:
            response = requests.post(url_chat, json=payload_chat, timeout=120)
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
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            payload_gen = {"model": self.model, "prompt": prompt, "stream": False}
            used = "generate"
            request_payload = payload_gen
            try:
                response = requests.post(f"{self.host}/api/generate", json=payload_gen, timeout=120)
            except requests.exceptions.RequestException as exc:
                elapsed = time.perf_counter() - t0
                meta = {
                    "endpoint": used,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "elapsed_sec": round(elapsed, 3),
                    "request": request_payload,
                    "fallback_from": "chat",
                }
                return {"text": f"Fallback request to {self.host}/api/generate failed: {exc}", "meta": meta}

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
            return {"text": f"Model endpoint {used} returned HTTP {response.status_code}: {details}", "meta": meta}

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


__all__ = ["LocalModelEngine"]
