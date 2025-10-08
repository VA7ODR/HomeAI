from __future__ import annotations

import pytest

from homeai import filesystem


def test_get_file_info_enforces_allowlist(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    allowed_file = base / "note.txt"
    allowed_file.write_text("hello")

    monkeypatch.setattr(filesystem.config, "BASE", base.resolve())

    info = filesystem.get_file_info("note.txt")
    assert info["path"] == str(allowed_file.resolve())
    assert info["size"] == len("hello")
    assert not info["is_dir"]

    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside_file = outside_root / "secret.txt"
    outside_file.write_text("nope")

    with pytest.raises(PermissionError):
        filesystem.get_file_info(outside_file)


def test_get_file_info_missing_file(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    base.mkdir()

    monkeypatch.setattr(filesystem.config, "BASE", base.resolve())

    with pytest.raises(FileNotFoundError):
        filesystem.get_file_info("missing.txt")

