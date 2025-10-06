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
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from datetime import datetime
import os, tempfile, shutil

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


class LocalJSONMemoryBackend:
    """Tiny JSON-on-disk conversation store.

    Messages are stored per-conversation under ``~/.homeai/memory``.  Each file
    contains a list of dictionaries representing the ``MemoryMessage`` dataclass
    above.  The backend exposes simple retrieval helpers expected by
    ``ContextBuilder``; for now the FTS/semantic lookups fall back to lexical
    heuristics so the application continues to work without PostgreSQL.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        default_dir = Path.home() / ".homeai" / "memory"
        self.base_dir = (base_dir or default_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._primary_conversation_id = self._select_primary_conversation_id()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _conversation_path(self, conversation_id: str) -> Path:
        return self.base_dir / f"{_sanitize_id(conversation_id)}.json"

    def _select_primary_conversation_id(self) -> str:
        """Pick the conversation file we treat as the shared default."""

        try:
            existing = sorted(
                [p for p in self.base_dir.glob("*.json") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            existing = []
        if existing:
            return existing[-1].stem
        return "conversation"

    def _quarantine_corrupt(path: Path, exc: Exception) -> None:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        quarantined = path.with_suffix(path.suffix + f".corrupt-{ts}")
        try:
            shutil.move(str(path), str(quarantined))
        except Exception:
            pass  # last resort: leave it be

    def _write_messages(self, conversation_id: str, messages: Sequence[MemoryMessage]) -> None:
        path = self._conversation_path(conversation_id)
        serialisable = [msg.__dict__ for msg in messages]
        _atomic_write(path, json.dumps(serialisable, ensure_ascii=False, indent=2))

    def _load_messages(self, conversation_id: str) -> List[MemoryMessage]:
        path = self._conversation_path(conversation_id)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [MemoryMessage(**msg) for msg in raw]
        except json.JSONDecodeError as e:
            _quarantine_corrupt(path, e)
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def new_conversation_id(self) -> str:
        """Return the shared conversation identifier.

        The lightweight client currently treats the chat as a single ongoing
        thread.  ``new_conversation_id`` therefore returns the primary
        conversation identifier selected during initialisation.  When
        multi-chat support lands this method can grow an option to force a new
        ID, while existing callers keep their behaviour.
        """

        return self._primary_conversation_id

    def add_message(self, conversation_id: str, role: str, content: Dict[str, Any]) -> MemoryMessage:
        message = MemoryMessage(id=uuid.uuid4().hex, role=role, content=content, created_at=time.time())
        with self._lock:
            messages = self._load_messages(conversation_id)
            messages.append(message)
            self._write_messages(conversation_id, messages)
        return message

    def get_recent_messages(self, conversation_id: str, limit: int) -> List[MemoryMessage]:
        with self._lock:
            messages = self._load_messages(conversation_id)
        if limit <= 0:
            return list(messages)
        return list(messages[-limit:])

    def search_fts(self, conversation_id: str, query: str, limit: int) -> List[MemoryMessage]:
        """Simple keyword search used as an FTS stand-in."""

        query = (query or "").strip().lower()
        if not query:
            return []
        tokens = [tok for tok in re.split(r"\W+", query) if tok]
        if not tokens:
            return []

        with self._lock:
            messages = self._load_messages(conversation_id)

        scored: List[MemoryMessage] = []
        for msg in messages:
            text = _message_text(msg)
            if not text:
                continue
            low = text.lower()
            tf = sum(low.count(tok) for tok in tokens)
            if not tf:
                continue
            age_days = max(0.0, (time.time() - msg.created_at) / 86400.0)
            recency = 0.5 / (1.0 + age_days)  # ~0.5 today, decays over time
            score = tf + recency
            scored.append(MemoryMessage(**{**msg.__dict__, "score": float(score)}))

        scored.sort(key=lambda m: m.score or 0.0, reverse=True)
        return scored[:limit]

    def search_semantic(self, conversation_id: str, query: str, limit: int) -> List[MemoryMessage]:
        """Placeholder vector search.

        Until a vector DB is wired in we reuse the FTS hits and tag them as
        ``vector`` results with a fake distance so the ranking logic works.
        """

        if limit <= 0:
            return []
        hits = self.search_fts(conversation_id, query, limit * 2)
        # Reinterpret the score as a distance (lower is better). Use inverse to
        # avoid division by zero and keep deterministic ordering.
        transformed: List[MemoryMessage] = []
        for msg in hits:
            score = msg.score or 0.0
            distance = 1.0 / (1.0 + score)
            transformed.append(MemoryMessage(**{**msg.__dict__, "score": distance}))
        return transformed[:limit]

    def get_memories(self, conversation_id: str, limit: int) -> List[MemoryMessage]:
        """Return stored long-term memories if present.

        The lightweight backend does not yet support durable memories, so we
        simply return an empty list.  The method is provided for API parity with
        the intended Postgres-backed implementation.
        """

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

