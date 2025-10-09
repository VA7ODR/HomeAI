from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from . import config


def assert_in_allowlist(p: Path) -> Path:
    """Ensure ``p`` sits within the configured allowlist base directory."""

    base = config.BASE
    p = p.resolve()
    if not (p == base or base in p.parents):
        raise PermissionError(f"Path {p} is outside allowlist base {base}")
    return p


def resolve_under_base(user_path: str) -> Path:
    """Expand user input to an absolute Path scoped to the allowlist base."""

    p = Path(os.path.expandvars(os.path.expanduser(user_path)))
    if not p.is_absolute():
        p = config.BASE / p
    return assert_in_allowlist(p)


def list_dir(path: str = ".", pattern: str = "") -> Dict[str, Any]:
    """Return a deterministic directory listing filtered by ``pattern``."""

    root = resolve_under_base(path)
    if not root.exists() or not root.is_dir():
        return {"error": f"Not a directory: {root}"}
    items = []
    for entry in sorted(root.iterdir()):
        name = entry.name
        if pattern and pattern not in name:
            continue
        items.append(
            {
                "name": name,
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if entry.is_file() else None,
            }
        )
    return {"root": str(root), "count": len(items), "items": items}


def read_text_file(path: str) -> Dict[str, Any]:
    """Read a UTF-8 text file and return its contents with truncation metadata."""

    p = resolve_under_base(path)
    if not p.exists() or not p.is_file():
        return {"error": f"Not a file: {p}"}
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        return {"error": f"Read error: {exc}"}
    max_chars = 60000
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    return {"path": str(p), "truncated": truncated, "text": text}


def locate_files(
    query: str,
    start: str = ".",
    max_results: int = 200,
    case_insensitive: bool = True,
) -> Dict[str, Any]:
    """Walk ``start`` looking for filenames that contain ``query``."""

    root = resolve_under_base(start)
    if not root.exists() or not root.is_dir():
        return {"error": f"Not a directory: {root}"}
    normalized_query = query.casefold() if case_insensitive else query
    results = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            comparison_name = filename.casefold() if case_insensitive else filename
            if normalized_query in comparison_name:
                full = Path(dirpath) / filename
                try:
                    results.append(str(full.resolve()))
                except Exception:
                    results.append(str(full))
                if len(results) >= max_results:
                    return {
                        "root": str(root),
                        "query": query,
                        "count": len(results),
                        "truncated": True,
                        "results": results,
                    }
    return {
        "root": str(root),
        "query": query,
        "count": len(results),
        "truncated": False,
        "results": results,
    }


def get_file_info(p: str | os.PathLike[str]) -> Dict[str, Any]:
    """Return metadata for ``p`` after enforcing the allowlist."""

    path = resolve_under_base(os.fspath(p))
    stat_result = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat_result.st_size,
        "mtime": int(stat_result.st_mtime),
        "is_dir": path.is_dir(),
        "name": path.name,
    }


__all__ = [
    "assert_in_allowlist",
    "get_file_info",
    "list_dir",
    "locate_files",
    "read_text_file",
    "resolve_under_base",
]
