from __future__ import annotations

import json

from typing import Any, Dict, List, Tuple

import pytest
import requests

from homeai.model_engine import LocalModelEngine
from homeai.tool_utils import parse_structured_tool_call


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        json_payload: Any = None,
        text: str = "",
        raise_http: bool = False,
        headers: Dict[str, str] | None = None,
        iter_lines_payload: List[str] | None = None,
    ):
        self.status_code = status_code
        self._json_payload = json_payload
        self.text = text
        self._raise_http = raise_http
        self.reason = "Server Error" if status_code >= 400 else "OK"
        self.headers = headers or {}
        self._iter_lines_payload = list(iter_lines_payload or [])

    def raise_for_status(self) -> None:
        if self._raise_http and self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> Any:
        if isinstance(self._json_payload, BaseException):
            raise self._json_payload
        if self._json_payload is None:
            raise ValueError("no json")
        return self._json_payload

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._iter_lines_payload:
            if decode_unicode and isinstance(line, bytes):
                yield line.decode("utf-8")
            else:
                yield line

    def close(self) -> None:  # pragma: no cover - included for interface completeness
        return None


class _QueuedSession:
    def __init__(self, responses: List[Tuple[str, Dict[str, Any]]]):
        self._responses = responses
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def post(self, url: str, json: Dict[str, Any], timeout: int, **kwargs) -> _FakeResponse:
        stream_flag = kwargs.get("stream", False)
        self.calls.append((url, json))
        if not self._responses:
            raise AssertionError("No queued responses left")
        expected_url, payload = self._responses.pop(0)
        assert expected_url == url
        expected_payload = payload.get("payload")
        if expected_payload is not None:
            assert expected_payload == json
        expected_stream = payload.get("stream")
        if expected_stream is not None:
            assert expected_stream == stream_flag
        response = payload.get("response")
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, _FakeResponse)
        return response

    def close(self) -> None:  # pragma: no cover - nothing to clean up in tests
        pass


@pytest.fixture
def fake_session(monkeypatch):
    sessions: List[_QueuedSession] = []

    def factory() -> _QueuedSession:
        if not sessions:
            raise AssertionError("Test did not seed session responses")
        return sessions.pop(0)

    monkeypatch.setattr(requests, "Session", lambda: factory())

    def enqueue(responses: List[Tuple[str, Dict[str, Any]]]) -> None:
        sessions.append(_QueuedSession(responses))

    return enqueue


def _build_chat_messages() -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "Ping"},
    ]


def test_local_model_engine_returns_metadata_on_http_error(fake_session) -> None:
    fake_session(
        [
            (
                "http://127.0.0.1:8000/api/chat",
                {
                    "payload": {"model": "gpt4", "messages": _build_chat_messages(), "stream": False},
                    "response": _FakeResponse(
                        status_code=500,
                        json_payload={"detail": "boom"},
                        raise_http=True,
                    ),
                },
            )
        ]
    )

    engine = LocalModelEngine(model="gpt4", host="127.0.0.1:8000")
    result = engine.chat(_build_chat_messages())

    assert "HTTP 500" in result["text"]
    meta = result["meta"]
    assert meta["status"] == 500
    assert meta["endpoint"] == "chat"
    assert meta["request"]["model"] == "gpt4"


def test_local_model_engine_falls_back_to_generate(fake_session) -> None:
    chat_payload = {"model": "gpt4", "messages": _build_chat_messages(), "stream": False}
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in _build_chat_messages()])
    gen_payload = {"model": "gpt4", "prompt": prompt, "stream": False}

    fake_session(
        [
            (
                "http://localhost:9999/api/chat",
                {
                    "payload": chat_payload,
                    "response": _FakeResponse(status_code=404, json_payload={"error": "missing"}),
                },
            ),
            (
                "http://localhost:9999/api/generate",
                {
                    "payload": gen_payload,
                    "response": _FakeResponse(
                        status_code=200,
                        json_payload={"message": {"content": "pong"}},
                    ),
                },
            ),
        ]
    )

    engine = LocalModelEngine(model="gpt4", host="localhost:9999")
    result = engine.chat(_build_chat_messages())

    assert result["text"] == "pong"
    assert result["meta"]["endpoint"] == "generate"
    assert result["meta"]["request"] == gen_payload


def test_local_model_engine_reports_request_exception(fake_session) -> None:
    fake_session(
        [
            (
                "http://test:1000/api/chat",
                {
                    "payload": {"model": "gpt4", "messages": _build_chat_messages(), "stream": False},
                    "response": requests.exceptions.ConnectTimeout("boom"),
                },
            )
        ]
    )

    engine = LocalModelEngine(model="gpt4", host="test:1000")
    result = engine.chat(_build_chat_messages())

    assert "Model request failed" in result["text"]
    assert result["meta"]["error"].startswith("ConnectTimeout")


def test_chat_stream_emits_deltas_and_meta(fake_session) -> None:
    payload = {"model": "gpt4", "messages": _build_chat_messages(), "stream": True}
    fake_session(
        [
            (
                "http://127.0.0.1:8000/api/chat",
                {
                    "payload": payload,
                    "stream": True,
                    "response": _FakeResponse(
                        status_code=200,
                        headers={"content-type": "text/event-stream"},
                        iter_lines_payload=[
                            "data: {\"message\": {\"content\": \"Hel\"}}",
                            "",
                            "data: {\"delta\": {\"content\": \"lo\"}}",
                            "data: {\"done\": true, \"meta\": {\"status\": 200}}",
                            "data: [DONE]",
                        ],
                    ),
                },
            )
        ]
    )

    engine = LocalModelEngine(model="gpt4", host="127.0.0.1:8000")
    events = list(engine.chat_stream(_build_chat_messages()))

    assert events[-1]["event"] == "complete"
    deltas = "".join(event["data"] for event in events if event["event"] == "delta")
    assert deltas == "Hello"
    complete = events[-1]["data"]
    assert complete["text"] == "Hello"
    assert complete["meta"]["endpoint"] == "chat"
    assert complete["meta"]["status"] == 200
    assert "elapsed_sec" in complete["meta"]


def test_chat_stream_accumulates_tool_calls(fake_session) -> None:
    payload = {"model": "gpt4", "messages": _build_chat_messages(), "stream": True}
    fake_session(
        [
            (
                "http://127.0.0.1:8000/api/chat",
                {
                    "payload": payload,
                    "stream": True,
                    "response": _FakeResponse(
                        status_code=200,
                        headers={"content-type": "text/event-stream"},
                        iter_lines_payload=[
                            "data: {\"delta\": {\"tool_calls\": [{\"index\": 0, \"id\": \"call_A\", \"function\": {\"name\": \"my_tool\", \"arguments\": \"{\"}}]}}",
                            "data: {\"delta\": {\"tool_calls\": [{\"index\": 0, \"function\": {\"arguments\": \"\\\"path\\\": \\\"./foo\\\"\"}}]}}",
                            "data: {\"delta\": {\"tool_calls\": [{\"index\": 0, \"function\": {\"arguments\": \"}\"}}]}}",
                            "data: {\"delta\": {\"tool_calls\": [{\"index\": 1, \"id\": \"call_B\", \"function\": {\"name\": \"second_tool\", \"arguments\": \"{\\\"value\\\": 5}\"}}]}}",
                            "data: {\"done\": true, \"meta\": {\"status\": 200}}",
                            "data: [DONE]",
                        ],
                    ),
                },
            )
        ]
    )

    engine = LocalModelEngine(model="gpt4", host="127.0.0.1:8000")
    events = list(engine.chat_stream(_build_chat_messages()))

    complete = events[-1]["data"]
    response_meta = complete["meta"]["response"]
    tool_calls = response_meta["message"]["tool_calls"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["arguments"] == '{"path": "./foo"}'
    assert tool_calls[0]["function"]["name"] == "my_tool"
    assert tool_calls[1]["id"] == "call_B"
    assert tool_calls[1]["function"]["arguments"] == '{"value": 5}'
    tool_name, tool_args = parse_structured_tool_call(response_meta)
    assert tool_name == "my_tool"
    assert tool_args == {"path": "./foo"}


def test_chat_stream_replaces_arguments_when_final_payload_complete(fake_session) -> None:
    payload = {"model": "gpt4", "messages": _build_chat_messages(), "stream": True}
    streamed_chunks = [
        {
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_A",
                        "function": {"name": "my_tool", "arguments": '{"path": '},
                    }
                ]
            }
        },
        {
            "delta": {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": '"./foo"}'}},
                ]
            }
        },
        {
            "message": {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_A",
                        "function": {
                            "name": "my_tool",
                            "arguments": '{"path": "./foo"}',
                        },
                    }
                ]
            }
        },
    ]
    iter_lines_payload = ["data: " + json.dumps(chunk) for chunk in streamed_chunks]
    iter_lines_payload.append("data: [DONE]")

    fake_session(
        [
            (
                "http://127.0.0.1:8000/api/chat",
                {
                    "payload": payload,
                    "stream": True,
                    "response": _FakeResponse(
                        status_code=200,
                        headers={"content-type": "text/event-stream"},
                        iter_lines_payload=iter_lines_payload,
                    ),
                },
            )
        ]
    )

    engine = LocalModelEngine(model="gpt4", host="127.0.0.1:8000")
    events = list(engine.chat_stream(_build_chat_messages()))

    complete = events[-1]["data"]
    response_meta = complete["meta"]["response"]
    tool_calls = response_meta["message"]["tool_calls"]
    assert len(tool_calls) == 1
    args = tool_calls[0]["function"]["arguments"]
    assert args == '{"path": "./foo"}'
    tool_name, tool_args = parse_structured_tool_call(response_meta)
    assert tool_name == "my_tool"
    assert tool_args == {"path": "./foo"}


def test_chat_stream_falls_back_to_standard_call(fake_session) -> None:
    stream_payload = {"model": "gpt4", "messages": _build_chat_messages(), "stream": True}
    chat_payload = {"model": "gpt4", "messages": _build_chat_messages(), "stream": False}
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in _build_chat_messages()])
    generate_payload = {"model": "gpt4", "prompt": prompt, "stream": True}
    fake_session(
        [
            (
                "http://localhost:8080/api/chat",
                {
                    "payload": stream_payload,
                    "stream": True,
                    "response": _FakeResponse(status_code=404, json_payload={"error": "nope"}),
                },
            ),
            (
                "http://localhost:8080/api/generate",
                {
                    "payload": generate_payload,
                    "stream": True,
                    "response": requests.exceptions.ConnectTimeout("fail"),
                },
            ),
            (
                "http://localhost:8080/api/chat",
                {
                    "payload": chat_payload,
                    "response": _FakeResponse(
                        status_code=200,
                        json_payload={"message": {"content": "fallback"}},
                    ),
                },
            ),
        ]
    )

    engine = LocalModelEngine(model="gpt4", host="localhost:8080")
    events = list(engine.chat_stream(_build_chat_messages()))

    assert len(events) == 1
    assert events[0]["event"] == "complete"
    assert events[-1]["data"]["text"] == "fallback"
    assert events[-1]["data"]["meta"]["endpoint"] == "chat"
