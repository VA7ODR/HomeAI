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
