import importlib


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


def test_detect_intent_slash_requires_keyword_without_whitespace():
    homeai_app = importlib.import_module("homeai_app")

    intent, args = homeai_app.detect_intent("/ locate something")

    assert intent == "chat"
    assert args == {}


def test_detect_intent_handles_text_synonyms():
    homeai_app = importlib.import_module("homeai_app")

    intent, args = homeai_app.detect_intent("Summarise notes/today.md")

    assert intent == "summarize"
    assert args == {"path": "notes/today.md"}


def test_detect_intent_unknown_slash_falls_back_to_chat():
    homeai_app = importlib.import_module("homeai_app")

    intent, args = homeai_app.detect_intent("/dance party time")

    assert intent == "chat"
    assert args == {}
