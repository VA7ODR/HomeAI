import importlib
from typing import List


def test_context_builder_env_overrides_parses_ints(monkeypatch):
    homeai_app = importlib.import_module("homeai_app")

    monkeypatch.setenv("HOMEAI_CONTEXT_RECENT_LIMIT", "64")
    monkeypatch.setenv("HOMEAI_CONTEXT_TOKEN_BUDGET", " 20000 ")
    monkeypatch.setenv("HOMEAI_CONTEXT_RESERVE_FOR_RESPONSE", "1800")
    monkeypatch.setenv("HOMEAI_CONTEXT_FTS_LIMIT", "oops")  # ignored

    overrides = homeai_app._context_builder_env_overrides()

    assert overrides["recent_limit"] == 64
    assert overrides["token_budget"] == 20000
    assert overrides["reserve_for_response"] == 1800
    assert "fts_limit" not in overrides


def test_detect_intent_supports_slash_read():
    homeai_app = importlib.import_module("homeai_app")

    intent, args = homeai_app.detect_intent("/read docs/file.txt")

    assert intent == "read"
    assert args == {"path": "docs/file.txt"}


def test_detect_intent_slash_defaults_browse_path():
    homeai_app = importlib.import_module("homeai_app")

    intent, args = homeai_app.detect_intent("/ls")

    assert intent == "browse"
    assert args == {"path": "."}


def test_detect_intent_slash_ignores_empty_command():
    homeai_app = importlib.import_module("homeai_app")

    intent, args = homeai_app.detect_intent("/   ")

    assert intent == "chat"
    assert args == {}


def test_detect_intent_slash_tolerates_space_after_slash():
    homeai_app = importlib.import_module("homeai_app")

    intent, args = homeai_app.detect_intent("/ read docs/file.txt")

    assert intent == "read"
    assert args == {"path": "docs/file.txt"}


def test_detect_intent_plain_text_treated_as_chat():
    homeai_app = importlib.import_module("homeai_app")

    intent, args = homeai_app.detect_intent("Summarise notes/today.md")

    assert intent == "chat"
    assert args == {}


def test_detect_intent_unknown_slash_falls_back_to_chat():
    homeai_app = importlib.import_module("homeai_app")

    intent, args = homeai_app.detect_intent("/dance party time")

    assert intent == "chat"
    assert args == {}


def test_detect_intent_requires_slash_prefix_for_commands():
    homeai_app = importlib.import_module("homeai_app")

    intent, args = homeai_app.detect_intent("read docs/file.txt")

    assert intent == "chat"
    assert args == {}


def test_on_user_streams_event_log(tmp_path, monkeypatch):
    monkeypatch.setenv("HOMEAI_ALLOWLIST_BASE", str(tmp_path))
    monkeypatch.setenv("HOMEAI_DATA_DIR", str(tmp_path / "memory"))

    module = importlib.import_module("homeai_app")
    homeai_app = importlib.reload(module)

    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")

    state = homeai_app._initial_state()
    updates = list(homeai_app.on_user("/ls", state))

    assert len(updates) >= 4
    logs = [entry[2] for entry in updates]
    assert any("Executing 'browse'" in log for log in logs)

    final_state, _, final_log, cleared, chat_history = updates[-1]
    assert cleared == ""
    assert "Browse succeeded" in final_log
    assert isinstance(final_state.get("event_log"), list)
    assert isinstance(chat_history, list)
    assert chat_history[-1]["role"] == "assistant"


def test_on_user_marks_pending_bubble(tmp_path, monkeypatch):
    monkeypatch.setenv("HOMEAI_ALLOWLIST_BASE", str(tmp_path))
    monkeypatch.setenv("HOMEAI_DATA_DIR", str(tmp_path / "memory"))

    module = importlib.import_module("homeai_app")
    homeai_app = importlib.reload(module)

    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")

    state = homeai_app._initial_state()
    updates = list(homeai_app.on_user("/ls", state))

    pending_found = False
    for update in updates:
        chat_history = update[-1]
        for message in chat_history:
            if message["role"] == "assistant" and "pending-response-bubble" in message["content"]:
                pending_found = True
                break
        if pending_found:
            break

    assert pending_found, "pending styling should appear while response is streaming"

    final_state = updates[-1][0]
    final_content = final_state["history"][-1]["content"]
    assert "pending-response-bubble" not in final_content


def test_empty_model_reply_surfaces_fallback_message(tmp_path, monkeypatch):
    monkeypatch.setenv("HOMEAI_ALLOWLIST_BASE", str(tmp_path))
    monkeypatch.setenv("HOMEAI_DATA_DIR", str(tmp_path / "memory"))

    module = importlib.import_module("homeai_app")
    homeai_app = importlib.reload(module)

    monkeypatch.setattr(
        homeai_app.engine,
        "chat",
        lambda *_: {"text": "", "meta": {"status": 200}},
    )
    monkeypatch.setattr(
        homeai_app.engine,
        "chat_stream",
        lambda *_: iter([{"event": "complete", "data": {"text": "", "meta": {"status": 200}}}]),
    )

    state = homeai_app._initial_state()
    updates = list(homeai_app.on_user("Do you remember anything?", state))

    final_state, _, _, _, chat_history = updates[-1]
    assistant_turn = final_state["history"][-1]["content"]

    assert "I didn't receive any text back from the model" in assistant_turn
    assert any(
        "Warning: model returned empty response text." in entry
        for entry in final_state.get("event_log", [])
    )
    assert chat_history[-1]["role"] == "assistant"


def test_streaming_updates_pending_bubble(tmp_path, monkeypatch):
    monkeypatch.setenv("HOMEAI_ALLOWLIST_BASE", str(tmp_path))
    monkeypatch.setenv("HOMEAI_DATA_DIR", str(tmp_path / "memory"))

    module = importlib.import_module("homeai_app")
    homeai_app = importlib.reload(module)

    def _stream(*_args, **_kwargs):
        yield {"event": "delta", "data": "Hel"}
        yield {"event": "delta", "data": "lo"}
        yield {
            "event": "complete",
            "data": {"text": "Hello", "meta": {"status": 200}},
        }

    monkeypatch.setattr(homeai_app.engine, "chat_stream", lambda *_: _stream())

    state = homeai_app._initial_state()
    updates = list(homeai_app.on_user("Hello there?", state))

    pending_samples = []
    for update in updates:
        chat_history = update[-1]
        for message in chat_history:
            if message["role"] == "assistant" and "pending-response-bubble" in message["content"]:
                pending_samples.append(message["content"])

    assert any("Hel" in sample or "lo" in sample for sample in pending_samples)

    final_message = updates[-1][0]["history"][-1]["content"]
    assert "Hello" in final_message
    assert "pending-response-bubble" not in final_message


def test_streaming_placeholder_appears_before_final_chunk(tmp_path, monkeypatch):
    monkeypatch.setenv("HOMEAI_ALLOWLIST_BASE", str(tmp_path))
    monkeypatch.setenv("HOMEAI_DATA_DIR", str(tmp_path / "memory"))

    module = importlib.import_module("homeai_app")
    homeai_app = importlib.reload(module)

    def _stream(*_args, **_kwargs):
        yield {
            "event": "complete",
            "data": {"text": "Done", "meta": {"status": 200}},
        }

    monkeypatch.setattr(homeai_app.engine, "chat_stream", lambda *_: _stream())

    state = homeai_app._initial_state()
    updates = list(homeai_app.on_user("Ping?", state))

    pending_indices: List[int] = []
    for idx, update in enumerate(updates):
        chat_history = update[-1]
        for message in chat_history:
            if message["role"] == "assistant" and "pending-response-bubble" in message["content"]:
                pending_indices.append(idx)
                break

    assert pending_indices, "should show a pending bubble before completion"
    assert pending_indices[0] < len(updates) - 1

    final_message = updates[-1][0]["history"][-1]["content"]
    assert "Done" in final_message
    assert "pending-response-bubble" not in final_message
