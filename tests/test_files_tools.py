"""Tests for file utility functions in gradio_tbh_canvas."""

from __future__ import annotations

from pathlib import Path
import importlib
import sys
import types

import pytest


@pytest.fixture()
def canvas_module(tmp_path, monkeypatch):
    """Import ``gradio_tbh_canvas`` with a controlled environment."""

    class _DummyComponent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def click(self, *args, **kwargs):
            return self

        def submit(self, *args, **kwargs):
            return self

        def change(self, *args, **kwargs):
            return self

        def then(self, *args, **kwargs):
            return self

        def launch(self, *args, **kwargs):
            return None

    class _DummyBlocks(_DummyComponent):
        def load(self, *args, **kwargs):
            return _DummyComponent()

    dummy_gradio = types.ModuleType("gradio")
    dummy_gradio.Blocks = _DummyBlocks
    dummy_gradio.Markdown = lambda *args, **kwargs: _DummyComponent()
    dummy_gradio.Row = lambda *args, **kwargs: _DummyComponent()
    dummy_gradio.Column = lambda *args, **kwargs: _DummyComponent()
    dummy_gradio.Dropdown = lambda *args, **kwargs: _DummyComponent()
    dummy_gradio.Textbox = lambda *args, **kwargs: _DummyComponent()
    dummy_gradio.State = lambda *args, **kwargs: _DummyComponent()
    dummy_gradio.Chatbot = lambda *args, **kwargs: _DummyComponent()
    dummy_gradio.Button = lambda *args, **kwargs: _DummyComponent()

    class _DummyResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    dummy_requests = types.ModuleType("requests")
    dummy_requests.post = lambda *args, **kwargs: _DummyResponse()

    monkeypatch.setitem(sys.modules, "gradio", dummy_gradio)
    monkeypatch.setitem(sys.modules, "requests", dummy_requests)
    monkeypatch.setenv("TBH_ALLOWLIST_BASE", str(tmp_path))

    import gradio_tbh_canvas

    module = importlib.reload(gradio_tbh_canvas)
    return module


def test_resolve_under_base_allows_relative_and_absolute(canvas_module, tmp_path):
    base = tmp_path
    nested_dir = base / "nested"
    nested_dir.mkdir()
    absolute_path = nested_dir / "file.txt"
    absolute_path.write_text("content", encoding="utf-8")

    resolved_relative = canvas_module.resolve_under_base("nested/file.txt")
    resolved_absolute = canvas_module.resolve_under_base(str(absolute_path))

    assert resolved_relative == absolute_path
    assert resolved_absolute == absolute_path


def test_resolve_under_base_blocks_outside_base(canvas_module, tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir()

    with pytest.raises(PermissionError):
        canvas_module.resolve_under_base(str(outside))


def test_list_dir_lists_files_and_filters_pattern(canvas_module, tmp_path):
    (tmp_path / "folder").mkdir()
    text_file = tmp_path / "file_a.txt"
    text_file.write_text("hi", encoding="utf-8")
    log_file = tmp_path / "file_b.log"
    log_file.write_text("hello", encoding="utf-8")

    result = canvas_module.list_dir(pattern=".txt")

    assert result["root"] == str(tmp_path)
    assert result["count"] == 1
    assert result["items"] == [
        {"name": "file_a.txt", "is_dir": False, "size": text_file.stat().st_size}
    ]


def test_list_dir_reports_error_for_non_directory(canvas_module, tmp_path):
    text_file = tmp_path / "file.txt"
    text_file.write_text("hello", encoding="utf-8")

    result = canvas_module.list_dir(str(text_file))

    assert "error" in result
    assert "Not a directory" in result["error"]


def test_read_text_file_returns_contents(canvas_module, tmp_path):
    text_file = tmp_path / "file.txt"
    text_file.write_text("hello", encoding="utf-8")

    result = canvas_module.read_text_file("file.txt")

    assert result["path"] == str(text_file)
    assert result["truncated"] is False
    assert result["text"] == "hello"


def test_read_text_file_truncates_large_files(canvas_module, tmp_path):
    long_text = "a" * 60050
    text_file = tmp_path / "long.txt"
    text_file.write_text(long_text, encoding="utf-8")

    result = canvas_module.read_text_file("long.txt")

    assert result["truncated"] is True
    assert len(result["text"]) == 60000


def test_read_text_file_reports_missing_file(canvas_module):
    result = canvas_module.read_text_file("missing.txt")

    assert "error" in result
    assert "Not a file" in result["error"]


def test_locate_files_finds_matches_case_insensitive(canvas_module, tmp_path):
    subdir = tmp_path / "Project"
    subdir.mkdir()
    file_a = subdir / "notes.TXT"
    file_a.write_text("hello", encoding="utf-8")
    file_b = tmp_path / "data.md"
    file_b.write_text("content", encoding="utf-8")

    result = canvas_module.locate_files("txt")

    assert result["root"] == str(tmp_path)
    assert result["count"] == 1
    assert result["truncated"] is False
    assert {Path(p) for p in result["results"]} == {file_a.resolve()}


def test_locate_files_honors_max_results_and_truncates(canvas_module, tmp_path):
    for idx in range(3):
        f = tmp_path / f"match_{idx}.txt"
        f.write_text("x", encoding="utf-8")

    result = canvas_module.locate_files("match", max_results=2)

    assert result["count"] == 2
    assert result["truncated"] is True
    assert len(result["results"]) == 2


def test_locate_files_reports_error_for_non_directory(canvas_module, tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = canvas_module.locate_files("file", start=str(file_path))

    assert "error" in result
    assert "Not a directory" in result["error"]
