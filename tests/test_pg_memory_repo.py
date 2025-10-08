from __future__ import annotations

import json
import sys
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional

import pytest

from context_memory import MemoryItem, PgMemoryRepo


class _FakeSQLModule:
    class SQL(str):
        def format(self, identifier: "_FakeSQLModule.Identifier") -> "_FakeSQLModule.SQL":
            return _FakeSQLModule.SQL(str(self).replace("{}", identifier.as_string()))

    class Identifier:
        def __init__(self, name: str) -> None:
            self._name = name

        def as_string(self) -> str:
            return self._name


class _FakeDB:
    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {}

    def _record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "id": payload["id"],
            "kind": payload["kind"],
            "source": payload["source"],
            "user_id": payload["user_id"],
            "session_id": payload["session_id"],
            "tags": json.loads(payload["tags"]),
            "content": json.loads(payload["content"]),
            "plain_text": payload["plain_text"],
            "created_at": payload["created_at"],
            "updated_at": payload["updated_at"],
        }
        existing = self.rows.get(record["id"])
        if existing:
            record["created_at"] = min(existing["created_at"], record["created_at"])
        self.rows[record["id"]] = record
        return dict(record)

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        normalized = " ".join(query.split()).strip().lower()
        params = params or {}
        if normalized.startswith("insert into"):
            return [self._record(params)]
        if "where id =" in normalized:
            item_id = params.get("item_id")
            row = self.rows.get(item_id)
            return [dict(row)] if row else []
        if "where session_id =" in normalized and "order by created_at" in normalized:
            session_id = params.get("session_id")
            rows = [dict(row) for row in self.rows.values() if row["session_id"] == session_id]
            rows.sort(key=lambda r: r["created_at"])
            return rows
        if normalized.startswith("select session_id, max"):
            summary: Dict[str, str] = {}
            for row in self.rows.values():
                sid = row["session_id"]
                summary[sid] = max(summary.get(sid, ""), row["created_at"])
            ordered = sorted(summary.items(), key=lambda it: it[1])
            return [{"session_id": sid, "last_created": created} for sid, created in ordered]
        if "plain_text ilike" in normalized:
            tokens = [val.strip("% ").lower() for key, val in params.items() if key.startswith("tok_")]
            session_id = params.get("session_id")
            rows = [dict(row) for row in self.rows.values()]
            results = []
            for row in rows:
                if session_id and row["session_id"] != session_id:
                    continue
                haystack = row["plain_text"].lower()
                if all(tok in haystack for tok in tokens):
                    results.append(row)
            return results
        if "order by created_at" in normalized:
            session_id = params.get("session_id")
            rows = [dict(row) for row in self.rows.values() if not session_id or row["session_id"] == session_id]
            rows.sort(key=lambda r: r["created_at"])
            return rows
        if normalized.startswith("select"):
            rows = [dict(row) for row in self.rows.values()]
            if params.get("session_id"):
                rows = [row for row in rows if row["session_id"] == params["session_id"]]
            return rows
        return []


class _Cursor:
    def __init__(self, db: _FakeDB):
        self._db = db
        self._rows: List[Dict[str, Any]] = []
        self.description: List[Iterable[str]] = []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._rows = []

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._rows = self._db.execute(query, params)
        if self._rows:
            self.description = [(key,) for key in self._rows[0].keys()]

    def fetchone(self) -> Optional[Dict[str, Any]]:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> List[Dict[str, Any]]:
        return list(self._rows)


class _Connection:
    def __init__(self, db: _FakeDB):
        self._db = db
        self._homeai_schema_set = False

    def cursor(self, row_factory=None) -> _Cursor:  # pragma: no cover - row_factory ignored intentionally
        return _Cursor(self._db)

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._db.execute(query, params)


class _ConnectionContext:
    def __init__(self, db: _FakeDB):
        self._db = db

    def __enter__(self) -> _Connection:
        return _Connection(self._db)

    def __exit__(self, exc_type, exc, tb) -> None:
        pass


class _FakePool:
    def __init__(self, *, conninfo: str, min_size: int, max_size: int, kwargs: Dict[str, Any]):
        self.conninfo = conninfo
        self.min_size = min_size
        self.max_size = max_size
        self.kwargs = kwargs
        self._db = _FakeDB()

    def wait(self) -> None:
        return None

    def connection(self) -> _ConnectionContext:
        return _ConnectionContext(self._db)


class _FakeRowsModule:
    dict_row = object()


class _FakePsycopgModule:
    sql = _FakeSQLModule
    rows = _FakeRowsModule


@pytest.fixture(autouse=False)
def fake_psycopg(monkeypatch):
    modules = {
        "psycopg": _FakePsycopgModule,
        "psycopg.sql": _FakeSQLModule,
        "psycopg.rows": _FakeRowsModule,
        "psycopg_pool": type("psycopg_pool", (), {"ConnectionPool": _FakePool}),
    }

    with monkeypatch.context() as m:
        for name, module in modules.items():
            m.setitem(sys.modules, name, module)
        yield


def _build_item(session_id: str = "session") -> MemoryItem:
    return MemoryItem(
        id="item-1",
        kind="note",
        source="agent",
        session_id=session_id,
        plain_text="Important note",
        content={"text": "Important note"},
        tags=["tag"],
    )


def test_pg_repo_round_trip(fake_psycopg) -> None:
    repo = PgMemoryRepo("postgresql://example", schema="public")

    item = _build_item(session_id="alpha")
    stored = repo.upsert(item)
    assert stored.session_id == "alpha"

    fetched = repo.get(item.id)
    assert fetched is not None
    assert fetched.plain_text == item.plain_text

    search_hits = repo.search_text("important", filters={"session_id": "alpha"}, k=5)
    assert [hit.id for hit in search_hits] == [item.id]

    sessions = repo.list_session_ids()
    assert "alpha" in sessions


def test_pg_repo_updates_existing_rows(fake_psycopg) -> None:
    repo = PgMemoryRepo("postgresql://example", schema="public")

    first = _build_item(session_id="beta")
    repo.upsert(first)

    updated = replace(first, plain_text="Refined note", content={"text": "Refined note"})
    repo.upsert(updated)

    results = repo.list_session("beta")
    assert len(results) == 1
    assert results[0].plain_text == "Refined note"
