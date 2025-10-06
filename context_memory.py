"""Lightweight conversation memory and context building utilities.

This module provides a disk-backed memory backend and a context builder that
mirrors the ranking/merging logic from the original project canvas.  The goal
is to keep the code self-contained and dependency-light so it can operate even
when PostgreSQL or a vector database is unavailable.  The backend stores
messages as JSON files on disk and offers basic retrieval primitives used by
``ContextBuilder`` to construct the messages payload that we send to the LLM.

The implementation purposefully keeps the search heuristics simple.  When a
more sophisticated database is available the backend can be swapped for one
that performs FTS/semantic search, while the high-level context assembly stays
the same.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, replace, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from abc import ABC, abstractmethod
import threading
import re

def _sanitize_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s) or "conversation"

def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)  # atomic on same filesystem

def _rough_tokens(text: str) -> int:
    """Heuristic token estimate used for budget trimming.

    We do not have access to the model tokenizer, therefore we approximate
    tokens by counting words and punctuation.  This mirrors the helper that was
    present in the original markdown snippet.
    """

    if not text:
        return 0
    # Rough approximation: words + half the punctuation count.
    word_count = len(re.findall(r"\w+", text))
    punct_bonus = len(re.findall(r"[.,;:!?]", text)) // 2
    return word_count + punct_bonus


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: Optional[Any]) -> datetime:
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return datetime.fromtimestamp(0, tz=timezone.utc)
    if not ts:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(ts))
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


@dataclass
class MemoryItem:
    id: str
    kind: str
    source: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    content: Dict[str, Any] = field(default_factory=dict)
    plain_text: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tags": list(self.tags),
            "content": self.content,
            "plain_text": self.plain_text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(
            id=data.get("id") or uuid.uuid4().hex,
            kind=data.get("kind", "note"),
            source=data.get("source", "agent"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            tags=list(data.get("tags") or []),
            content=dict(data.get("content") or {}),
            plain_text=str(data.get("plain_text") or ""),
            created_at=data.get("created_at") or _now_iso(),
            updated_at=data.get("updated_at") or _now_iso(),
            score=data.get("score"),
        )


class MemoryRepo(ABC):
    @abstractmethod
    def upsert(self, item: MemoryItem) -> MemoryItem:
        ...

    @abstractmethod
    def get(self, item_id: str) -> Optional[MemoryItem]:
        ...

    @abstractmethod
    def search_text(self, query: str, filters: Optional[Dict[str, Any]] = None, k: int = 20) -> List[MemoryItem]:
        ...

    @abstractmethod
    def search_semantic(self, query: str, filters: Optional[Dict[str, Any]] = None, k: int = 20) -> List[MemoryItem]:
        ...

    # Stage 0 helper to keep conversation features working without PG yet.
    @abstractmethod
    def list_session(self, session_id: str) -> List[MemoryItem]:
        ...

    @abstractmethod
    def list_session_ids(self) -> List[str]:
        ...

def _extract_plain_text(item: MemoryItem) -> str:
    if item.plain_text:
        return item.plain_text
    content = item.content
    if isinstance(content, dict):
        return str(
            content.get("text")
            or content.get("body")
            or content.get("summary")
            or content.get("display")
            or ""
        )
    return str(content or "")


@dataclass
class MemoryMessage:
    """Normalized representation of a stored chat message."""

    id: str
    role: str
    content: Dict[str, Any]
    created_at: float
    score: Optional[float] = None
    type: str = "message"


def _message_text(msg: MemoryMessage) -> str:
    content = msg.content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("display") or "")
    return str(content or "")


class FsMemoryRepo(MemoryRepo):
    """Filesystem-backed implementation of the ``MemoryRepo`` interface."""

    def __init__(self, base_dir: Optional[Path] = None, *, logger: Optional[logging.Logger] = None):
        env_dir = os.getenv("HOMEAI_DATA_DIR")
        resolved = base_dir or (Path(env_dir).expanduser() if env_dir else Path.home() / ".homeai" / "memory")
        self.base_dir = Path(resolved).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def conversation_path(self, session_id: str) -> Path:
        return self.base_dir / f"{_sanitize_id(session_id)}.json"

    def _quarantine_corrupt(self, path: Path, exc: Exception) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quarantined = path.with_suffix(path.suffix + f".corrupt-{ts}")
        try:
            shutil.move(str(path), str(quarantined))
            self._logger.warning("Quarantined corrupt memory file %s: %s", path, exc)
        except Exception as move_exc:  # pragma: no cover - defensive
            self._logger.error("Failed to quarantine %s: %s", path, move_exc)

    def _write_session_items(self, session_id: str, items: Sequence[MemoryItem]) -> None:
        path = self.conversation_path(session_id)
        serialisable = [item.to_dict() for item in items]
        _atomic_write(path, json.dumps(serialisable, ensure_ascii=False, indent=2))

    def _load_session_items(self, session_id: str) -> List[MemoryItem]:
        path = self.conversation_path(session_id)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [MemoryItem.from_dict(obj) for obj in raw]
        except json.JSONDecodeError as exc:
            self._quarantine_corrupt(path, exc)
            return []

    def _load_all_items(self) -> List[MemoryItem]:
        items: List[MemoryItem] = []
        for session_id in self.list_session_ids():
            items.extend(self._load_session_items(session_id))
        return items

    # ------------------------------------------------------------------
    # MemoryRepo interface
    # ------------------------------------------------------------------
    def upsert(self, item: MemoryItem) -> MemoryItem:
        session_id = item.session_id or "conversation"
        with self._lock:
            items = self._load_session_items(session_id)
            items_by_id = {existing.id: existing for existing in items}
            stored = items_by_id.get(item.id)
            now_iso = _now_iso()
            if stored:
                item.created_at = stored.created_at
            elif not item.created_at:
                item.created_at = now_iso
            item.updated_at = now_iso
            items_by_id[item.id] = item
            ordered = sorted(items_by_id.values(), key=lambda it: _parse_iso(it.created_at))
            self._write_session_items(session_id, ordered)
        return item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        with self._lock:
            for session_id in self.list_session_ids():
                for item in self._load_session_items(session_id):
                    if item.id == item_id:
                        return item
        return None

    def search_text(
        self, query: str, filters: Optional[Dict[str, Any]] = None, k: int = 20
    ) -> List[MemoryItem]:
        filters = filters or {}
        session_id = filters.get("session_id")
        if session_id:
            universe = self.list_session(session_id)
        else:
            universe = self._load_all_items()

        if k is not None and k <= 0:
            return []

        q = (query or "").strip().lower()
        if not q:
            ordered = sorted(universe, key=lambda it: _parse_iso(it.created_at))
            if not k:
                return ordered
            return ordered[-k:]

        tokens = [tok for tok in re.split(r"\W+", q) if tok]
        if not tokens:
            return []

        results: List[MemoryItem] = []
        now = datetime.now(timezone.utc)
        for item in universe:
            text = _extract_plain_text(item)
            if not text:
                continue
            low = text.lower()
            tf = sum(low.count(tok) for tok in tokens)
            if not tf:
                continue
            created = _parse_iso(item.created_at)
            age_days = max(0.0, (now - created).total_seconds() / 86400.0)
            recency = 0.5 / (1.0 + age_days)
            score = tf + recency
            results.append(replace(item, score=float(score)))

        results.sort(key=lambda it: it.score or 0.0, reverse=True)
        if k is None:
            return results
        return results[:k]

    def search_semantic(
        self, query: str, filters: Optional[Dict[str, Any]] = None, k: int = 20
    ) -> List[MemoryItem]:
        # Stage 0 fallback: reuse lexical hits and reinterpret scores as distances.
        if k is not None and k <= 0:
            return []
        hits = self.search_text(query, filters=filters, k=k * 2 if k else None)
        semantic: List[MemoryItem] = []
        for item in hits:
            score = item.score or 0.0
            distance = 1.0 / (1.0 + score)
            semantic.append(replace(item, score=float(distance)))
        if k is None:
            return semantic
        return semantic[:k]

    def list_session(self, session_id: str) -> List[MemoryItem]:
        with self._lock:
            items = self._load_session_items(session_id)
        items.sort(key=lambda it: _parse_iso(it.created_at))
        return items

    def list_session_ids(self) -> List[str]:
        try:
            files = [p for p in self.base_dir.glob("*.json") if p.is_file()]
        except OSError:
            files = []
        files.sort(key=lambda p: p.stat().st_mtime)
        return [p.stem for p in files]


class PgMemoryRepo(MemoryRepo):
    """Placeholder Postgres implementation for Stage 0."""

    def __init__(
        self,
        dsn: Optional[str],
        *,
        schema: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.dsn = dsn
        self.schema = schema
        self._logger = logger or logging.getLogger(__name__)
        if not dsn:
            self._logger.warning(
                "HOMEAI_STORAGE=pg but HOMEAI_PG_DSN not configured; using in-memory placeholder store."
            )
        self._store: Dict[str, MemoryItem] = {}
        self._lock = threading.Lock()

    def _values(self) -> List[MemoryItem]:
        return list(self._store.values())

    def upsert(self, item: MemoryItem) -> MemoryItem:
        with self._lock:
            now_iso = _now_iso()
            existing = self._store.get(item.id)
            if existing:
                item.created_at = existing.created_at
            elif not item.created_at:
                item.created_at = now_iso
            item.updated_at = now_iso
            self._store[item.id] = item
        return item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        with self._lock:
            return self._store.get(item_id)

    def search_text(
        self, query: str, filters: Optional[Dict[str, Any]] = None, k: int = 20
    ) -> List[MemoryItem]:
        filters = filters or {}
        session_id = filters.get("session_id")
        with self._lock:
            universe = [
                item
                for item in self._store.values()
                if not session_id or item.session_id == session_id
            ]
        if k is not None and k <= 0:
            return []

        if not query:
            universe.sort(key=lambda it: _parse_iso(it.created_at))
            if k is None:
                return universe
            return universe[-k:]
        tokens = [tok for tok in re.split(r"\W+", query.lower()) if tok]
        if not tokens:
            return []
        results: List[MemoryItem] = []
        now = datetime.now(timezone.utc)
        for item in universe:
            text = _extract_plain_text(item)
            low = text.lower()
            tf = sum(low.count(tok) for tok in tokens)
            if not tf:
                continue
            created = _parse_iso(item.created_at)
            age_days = max(0.0, (now - created).total_seconds() / 86400.0)
            recency = 0.5 / (1.0 + age_days)
            score = tf + recency
            results.append(replace(item, score=float(score)))
        results.sort(key=lambda it: it.score or 0.0, reverse=True)
        if k is None:
            return results
        return results[:k]

    def search_semantic(
        self, query: str, filters: Optional[Dict[str, Any]] = None, k: int = 20
    ) -> List[MemoryItem]:
        if k is not None and k <= 0:
            return []
        hits = self.search_text(query, filters=filters, k=k * 2 if k else None)
        semantic: List[MemoryItem] = []
        for item in hits:
            score = item.score or 0.0
            distance = 1.0 / (1.0 + score)
            semantic.append(replace(item, score=float(distance)))
        if k is None:
            return semantic
        return semantic[:k]

    def list_session(self, session_id: str) -> List[MemoryItem]:
        with self._lock:
            items = [item for item in self._store.values() if item.session_id == session_id]
        items.sort(key=lambda it: _parse_iso(it.created_at))
        return items

    def list_session_ids(self) -> List[str]:
        with self._lock:
            sessions = {item.session_id or "conversation" for item in self._store.values()}
        if not sessions:
            return []
        return sorted(sessions)


def make_repo(
    *,
    storage: Optional[str] = None,
    base_dir: Optional[Path] = None,
    dsn: Optional[str] = None,
    schema: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> MemoryRepo:
    mode = (storage or os.getenv("HOMEAI_STORAGE", "fs")).lower()
    log = logger or logging.getLogger(__name__)
    if mode == "pg":
        dsn = dsn or os.getenv("HOMEAI_PG_DSN")
        schema = schema or os.getenv("HOMEAI_PG_SCHEMA")
        return PgMemoryRepo(dsn, schema=schema, logger=log)
    # default to filesystem storage
    data_dir_env = os.getenv("HOMEAI_DATA_DIR")
    chosen_dir = base_dir or (Path(data_dir_env).expanduser() if data_dir_env else None)
    return FsMemoryRepo(chosen_dir, logger=log)


def _item_to_message(item: MemoryItem) -> MemoryMessage:
    created = _parse_iso(item.created_at)
    content = item.content or {}
    if not content and item.plain_text:
        content = {"text": item.plain_text}
    return MemoryMessage(
        id=item.id,
        role=item.source,
        content=content,
        created_at=created.timestamp(),
        score=item.score,
        type=item.kind,
    )


def _build_memory_item(
    *,
    conversation_id: str,
    role: str,
    content: Dict[str, Any],
    created_at: Optional[str] = None,
) -> MemoryItem:
    plain = content.get("text") if isinstance(content, dict) else None
    if not plain:
        plain = json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else str(content)
    return MemoryItem(
        id=uuid.uuid4().hex,
        kind="interaction",
        source=role,
        session_id=conversation_id,
        content=content,
        plain_text=str(plain or ""),
        created_at=created_at or _now_iso(),
        updated_at=created_at or _now_iso(),
    )


class LocalJSONMemoryBackend:
    """Conversation memory service backed by a ``MemoryRepo`` implementation."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        *,
        repo: Optional[MemoryRepo] = None,
        storage: Optional[str] = None,
        data_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        repo_base = data_dir or base_dir
        if repo is not None:
            self.repo = repo
        else:
            self.repo = make_repo(
                storage=storage,
                base_dir=repo_base,
                dsn=os.getenv("HOMEAI_PG_DSN"),
                schema=os.getenv("HOMEAI_PG_SCHEMA"),
                logger=self._logger,
            )
        if isinstance(self.repo, FsMemoryRepo):
            self.base_dir = self.repo.base_dir
        else:
            resolved_base = repo_base or Path(
                os.getenv("HOMEAI_DATA_DIR", str(Path.home() / ".homeai" / "memory"))
            )
            self.base_dir = Path(resolved_base).expanduser().resolve()
        self._lock = threading.Lock()
        self._primary_conversation_id = self._select_primary_conversation_id()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _conversation_path(self, conversation_id: str) -> Path:
        if isinstance(self.repo, FsMemoryRepo):
            return self.repo.conversation_path(conversation_id)
        return self.base_dir / f"{_sanitize_id(conversation_id)}.json"

    def _select_primary_conversation_id(self) -> str:
        sessions = self.repo.list_session_ids()
        if sessions:
            return sessions[-1]
        return "conversation"

    def _load_messages(self, conversation_id: str) -> List[MemoryMessage]:
        items = self.repo.list_session(conversation_id)
        return [_item_to_message(item) for item in items]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def new_conversation_id(self) -> str:
        return self._primary_conversation_id

    def add_message(self, conversation_id: str, role: str, content: Dict[str, Any]) -> MemoryMessage:
        item = _build_memory_item(conversation_id=conversation_id, role=role, content=content)
        stored = self.repo.upsert(item)
        return _item_to_message(stored)

    def get_recent_messages(self, conversation_id: str, limit: int) -> List[MemoryMessage]:
        items = self.repo.list_session(conversation_id)
        messages = [_item_to_message(item) for item in items]
        if limit <= 0:
            return messages
        return messages[-limit:]

    def search_fts(self, conversation_id: str, query: str, limit: int) -> List[MemoryMessage]:
        filters = {"session_id": conversation_id}
        hits = self.repo.search_text(query, filters=filters, k=limit)
        return [_item_to_message(item) for item in hits]

    def search_semantic(self, conversation_id: str, query: str, limit: int) -> List[MemoryMessage]:
        if limit <= 0:
            return []
        filters = {"session_id": conversation_id}
        hits = self.repo.search_semantic(query, filters=filters, k=limit)
        return [_item_to_message(item) for item in hits]

    def get_memories(self, conversation_id: str, limit: int) -> List[MemoryMessage]:
        # Stage 0 keeps the lightweight behaviour of returning no durable memories.
        return []

def _msg_row_to_snip(msg: MemoryMessage, *, tag: str = "message") -> Dict[str, Any]:
    return {
        "type": tag,
        "id": msg.id,
        "role": msg.role,
        "created_at": msg.created_at,
        "content": msg.content,
        "score": msg.score,
    }


def _mem_row_to_snip(msg: MemoryMessage) -> Dict[str, Any]:
    snip = _msg_row_to_snip(msg, tag="memory")
    snip.setdefault("content", {})
    return snip


class ContextBuilder:
    """Assemble compact chat prompts from stored conversation memory."""

    def __init__(
        self,
        backend: LocalJSONMemoryBackend,
        *,
        recent_limit: int = 8,
        fts_limit: int = 6,
        vector_limit: int = 6,
        memory_limit: int = 4,
        token_budget: int = 4096,
        reserve_for_response: int = 800,
    ) -> None:
        self.backend = backend
        self.recent_limit = recent_limit
        self.fts_limit = fts_limit
        self.vector_limit = vector_limit
        self.memory_limit = memory_limit
        self.token_budget = token_budget
        self.reserve_for_response = reserve_for_response

    def build_context(
        self,
        conversation_id: str,
        user_prompt: str,
        *,
        persona_seed: str,
    ) -> List[Dict[str, Any]]:
        # Fetch candidates -------------------------------------------------
        recent = list(self.backend.get_recent_messages(conversation_id, self.recent_limit))
        if recent:
            latest = recent[-1]
            if latest.role == "user" and _message_text(latest) == user_prompt:
                recent = recent[:-1]
        fts_hits = self.backend.search_fts(conversation_id, user_prompt, self.fts_limit)
        vec_hits = self.backend.search_semantic(conversation_id, user_prompt, self.vector_limit)
        mem_hits = self.backend.get_memories(conversation_id, self.memory_limit)

        recent_snips = [_msg_row_to_snip(m, tag="recent") for m in recent]
        fts_snips = [_msg_row_to_snip(m) for m in fts_hits]
        vec_snips = [_msg_row_to_snip(m, tag="vector") for m in vec_hits]
        mem_snips = [_mem_row_to_snip(m) for m in mem_hits]

        # Deduplicate ------------------------------------------------------
        buckets = [
            ("recent", recent_snips),
            ("fts",    fts_snips),
            ("vector", vec_snips),
            ("memory", mem_snips),
        ]

        seen_ids = set()
        merged: List[Dict[str, Any]] = []
        for bucket_name, bucket in buckets:
            for snip in bucket:
                snip["bucket"] = bucket_name
                if snip["id"] in seen_ids:
                    continue
                seen_ids.add(snip["id"])
                merged.append(snip)

        # Ranking ----------------------------------------------------------
        bucket_pri = {"recent": 0, "memory": 1, "fts": 2, "vector": 3}

        def _rank_key(snip: Dict[str, Any]) -> tuple:
            b = bucket_pri.get(snip.get("bucket"), 9)
            score = float(snip.get("score") or 0.0)
            # FTS: higher score is better; vector: lower "distance" is better
            if snip.get("bucket") == "vector":
                primary = score            # lower is better
            else:
                primary = -score           # higher is better
            created = float(snip.get("created_at") or 0.0)
            return (b, primary, created)

        merged.sort(key=_rank_key)

        # Build messages ---------------------------------------------------
        messages: List[Dict[str, Any]] = []

        sys_text = (
            persona_seed
            if persona_seed
            else "You are a concise, helpful local copilot. Use tools when needed; otherwise reply directly."
        )
        for mem in mem_snips:
            content = mem.get("content", {})
            if isinstance(content, dict) and content.get("kind") == "summary":
                sys_text += "\nSUMMARY:\n" + (content.get("text") or "")
                break
        messages.append({"role": "system", "content": sys_text})

        # Add conversation replay (chronological order)
        ordered_recent = list(recent)
        ordered_recent.sort(key=lambda m: m.created_at)
        for msg in ordered_recent:
            text = _message_text(msg)
            if not text:
                continue
            messages.append({"role": msg.role, "content": text})

        # Attach retrieved snippets as a single system block
        snippets_texts: List[str] = []
        for snip in merged:
            if snip.get("type") == "recent":
                continue
            content = snip.get("content", {})
            if isinstance(content, dict):
                text = content.get("text") or content.get("summary") or json.dumps(content, ensure_ascii=False)[:800]
            else:
                text = str(content)
            if text:
                snippets_texts.append(text)
            if len(snippets_texts) >= 12:
                break

        if snippets_texts:
            joined = "Relevant notes:\n- " + "\n- ".join(snippets_texts)
            messages.append({"role": "system", "content": joined})

        messages.append({"role": "user", "content": user_prompt})

        # Token budget trimming -------------------------------------------
        def _total_tokens(msgs: Iterable[Dict[str, Any]]) -> int:
            return sum(_rough_tokens(str(m.get("content", ""))) for m in msgs)

        budget = max(1000, self.token_budget - self.reserve_for_response)

        def _find_oldest_drop_idx(msgs: List[Dict[str, Any]]) -> Optional[int]:
            # keep the first system message and the last user message
            last_user_idx = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=None)
            candidates = [
                (i, m) for i, m in enumerate(msgs)
                if not (m.get("role") == "system" or i == last_user_idx)
            ]
            if not candidates:
                return None
            # drop the earliest (oldest in order) non-system/non-final-user message
            return min(candidates, key=lambda t: t[0])[0]

        while _total_tokens(messages) > budget and len(messages) > 3:
            idx = _find_oldest_drop_idx(messages)
            if idx is None:
                break
            messages.pop(idx)

        return messages

