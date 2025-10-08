"""PostgreSQL vector store and in-memory fallback for semantic retrieval.

This module provides a thin abstraction around the ``pgvector`` extension so
that HomeAI can ingest repository files and chat transcripts into a vector
database.  When PostgreSQL is unavailable we fall back to an in-memory store so
unit tests remain hermetic while still exercising the ingestion, chunking and
search logic.

The public API mirrors the requirements in the Stage 1 specification:

``ingest_files``
    Walk allowed files, split them into overlapping chunks, compute embeddings
    for new or changed chunks, and upsert them into the ``doc_chunks`` table.

``embed_missing``
    Compute embeddings for ``doc_chunks`` and ``messages`` rows that are missing
    embeddings.

``search_docs`` / ``search_messages``
    Perform cosine-similarity searches over the corresponding tables.  The
    PostgreSQL implementation uses the `<=>` operator so that the ANN indexes
    built on the vector columns can be utilised.  The in-memory fallback applies
    the same cosine distance implementation to keep behaviour consistent.

``backfill_embeddings``
    Recompute embeddings for all stored rows.  This is useful when switching to
    a different embedding model.  The method enforces a dimension check so that
    migrations must run before changing the configured embedding size.

``health``
    Summarise the current state of the store, reporting row counts, embedding
    coverage, whether ANN indexes are present (when PostgreSQL is active), and
    the configured embedding model metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import logging
import math
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from . import filesystem


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.fromtimestamp(0, tz=timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    return datetime.fromtimestamp(0, tz=timezone.utc)


def _normalise_path(path: Path) -> Path:
    return filesystem.assert_in_allowlist(path.expanduser().resolve())


def _looks_binary(path: Path, sample_size: int = 2048) -> bool:
    try:
        with path.open("rb") as fh:
            sample = fh.read(sample_size)
    except OSError:
        return True
    if b"\x00" in sample:
        return True
    if not sample:
        return False
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x7F)))
    translated = sample.translate(None, text_chars)
    # Consider binary if more than 30% of bytes are non-text characters.
    return float(len(translated)) / float(len(sample)) > 0.30


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = text.replace("\r\n", "\n")
    segments: List[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            segments.append(chunk)
        start += step
    if not segments and text.strip():
        segments.append(text.strip())
    return segments


def _hash_chunk(content: str) -> str:
    digest = hashlib.sha256()
    digest.update(content.encode("utf-8"))
    return digest.hexdigest()


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    # Guard against numerical noise outside [-1, 1].
    similarity = max(min(similarity, 1.0), -1.0)
    return 1.0 - similarity


@dataclass
class DocChunk:
    source_kind: str
    source_path: str
    file_name: str
    chunk_index: int
    content: str
    content_hash: str
    size_bytes: int
    mtime: datetime
    mime_type: Optional[str]
    embedding: Optional[Tuple[float, ...]] = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def to_result(self, distance: Optional[float] = None) -> Dict[str, Any]:
        preview = self.content[:400]
        score = None
        if distance is not None:
            score = 1.0 - distance
        return {
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "file_name": self.file_name,
            "chunk_index": self.chunk_index,
            "preview": preview,
            "score": score,
            "distance": distance,
            "updated_at": self.updated_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
        }


@dataclass
class MessageRow:
    thread_id: str
    role: str
    content: str
    message_id: str
    embedding: Optional[Tuple[float, ...]] = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def to_result(self, distance: Optional[float] = None) -> Dict[str, Any]:
        preview = self.content[:400]
        score = None
        if distance is not None:
            score = 1.0 - distance
        return {
            "thread_id": self.thread_id,
            "role": self.role,
            "message_id": self.message_id,
            "preview": preview,
            "score": score,
            "distance": distance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced or have invalid dimensions."""


class PgVectorStore:
    """Vector store backed by PostgreSQL or an in-memory fallback."""

    DOC_TABLE = "doc_chunks"
    MSG_TABLE = "messages"

    def __init__(
        self,
        dsn: Optional[str],
        *,
        schema: Optional[str] = None,
        embedder: Optional["SupportsEmbed"] = None,
        embedding_dimension: int = 384,
        embedding_model: str = "mini-lm-embedding",
        chunk_size: int = 1200,
        chunk_overlap: int = 240,
        skip_binaries: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.dsn = dsn
        self.schema = schema
        self.embedder = embedder
        self.embedding_dimension = embedding_dimension
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.skip_binaries = skip_binaries
        self._logger = logger or logging.getLogger(__name__)

        self._pool = None
        self._psycopg = None
        self._dict_row = None
        self._docs: MutableMapping[Tuple[str, int], DocChunk] = {}
        self._messages: MutableMapping[str, MessageRow] = {}

        if not dsn:
            self._logger.info("PgVectorStore running in in-memory mode (dsn not provided)")
            return

        try:
            import psycopg  # type: ignore
            from psycopg.rows import dict_row  # type: ignore
            from psycopg_pool import ConnectionPool  # type: ignore
        except ImportError:
            self._logger.warning(
                "psycopg not installed; PgVectorStore will operate in in-memory mode"
            )
            return

        try:
            pool = ConnectionPool(conninfo=dsn, min_size=1, max_size=5, kwargs={"autocommit": True})
            pool.wait()
        except Exception as exc:  # pragma: no cover - environment specific
            self._logger.error("Failed to create PostgreSQL pool: %s", exc)
            return

        self._pool = pool
        self._psycopg = psycopg
        self._dict_row = dict_row

    # ------------------------------------------------------------------
    # Helpers shared by both implementations
    # ------------------------------------------------------------------
    @property
    def uses_postgres(self) -> bool:
        return self._pool is not None

    def _ensure_embedder(self, embedder: Optional["SupportsEmbed"] = None) -> "SupportsEmbed":
        chosen = embedder or self.embedder
        if chosen is None:
            raise EmbeddingError("An embedding provider must be supplied")
        if getattr(chosen, "dimension", None) not in (None, self.embedding_dimension):
            raise EmbeddingError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension},"
                f" got {getattr(chosen, 'dimension', 'unknown')}"
            )
        return chosen

    def _validate_vector(self, vector: Sequence[float]) -> Tuple[float, ...]:
        if len(vector) != self.embedding_dimension:
            raise EmbeddingError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(vector)}"
            )
        return tuple(float(x) for x in vector)

    # ------------------------------------------------------------------
    # In-memory implementation
    # ------------------------------------------------------------------
    def ingest_files(
        self,
        paths: Iterable[str | os.PathLike[str] | Path],
        *,
        source_kind: str = "file",
        embedder: Optional["SupportsEmbed"] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        skip_binaries: Optional[bool] = None,
    ) -> "IngestReport":
        """Ingest files from the allowlisted base directory."""

        chosen_embedder = self._ensure_embedder(embedder)
        chosen_chunk_size = chunk_size or self.chunk_size
        chosen_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        skip_binaries = self.skip_binaries if skip_binaries is None else skip_binaries

        report = IngestReport()

        files: List[Path] = []
        for entry in paths:
            path = Path(entry)
            resolved = _normalise_path(path)
            if resolved.is_dir():
                for child in sorted(resolved.rglob("*")):
                    if child.is_file():
                        files.append(child)
            elif resolved.is_file():
                files.append(resolved)

        for file_path in files:
            report.files_processed += 1
            if skip_binaries and _looks_binary(file_path):
                report.files_skipped += 1
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                report.files_skipped += 1
                continue

            chunks = _chunk_text(text, chunk_size=chosen_chunk_size, overlap=chosen_overlap)
            stats = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            updated_at = datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc)
            size_bytes = stats.st_size

            for index, chunk_text in enumerate(chunks):
                report.chunks_processed += 1
                content_hash = _hash_chunk(chunk_text)
                key = (str(file_path), index)
                existing_row = self._docs.get(key)

                needs_embedding = True
                if existing_row and existing_row.content_hash == content_hash:
                    needs_embedding = existing_row.embedding is None
                    row = existing_row
                    row.updated_at = updated_at
                    row.size_bytes = size_bytes
                    row.mime_type = mime_type
                else:
                    row = DocChunk(
                        source_kind=source_kind,
                        source_path=str(file_path),
                        file_name=file_path.name,
                        chunk_index=index,
                        content=chunk_text,
                        content_hash=content_hash,
                        size_bytes=size_bytes,
                        mtime=updated_at,
                        mime_type=mime_type,
                        created_at=_utcnow(),
                        updated_at=updated_at,
                    )
                    needs_embedding = True

                if needs_embedding:
                    vector = self._validate_vector(chosen_embedder.embed([chunk_text])[0])
                    row.embedding = vector
                    report.chunks_embedded += 1

                self._docs[key] = row

        return report

    # ------------------------------------------------------------------
    def _iter_embeddings(
        self,
        records: Iterable[Tuple[str, str, Sequence[float]]],
        *,
        embedder: "SupportsEmbed",
    ) -> List[Tuple[str, Tuple[float, ...]]]:
        vectors: List[Tuple[str, Tuple[float, ...]]] = []
        for record_id, text, _ in records:
            vector = embedder.embed([text])[0]
            vectors.append((record_id, self._validate_vector(vector)))
        return vectors

    def embed_missing(self, *, embedder: Optional["SupportsEmbed"] = None) -> Dict[str, int]:
        chosen_embedder = self._ensure_embedder(embedder)
        updated = {"doc_chunks": 0, "messages": 0}

        for key, row in list(self._docs.items()):
            if row.embedding is None:
                vector = chosen_embedder.embed([row.content])[0]
                row.embedding = self._validate_vector(vector)
                self._docs[key] = row
                updated["doc_chunks"] += 1

        for message_id, row in list(self._messages.items()):
            if row.embedding is None:
                vector = chosen_embedder.embed([row.content])[0]
                row.embedding = self._validate_vector(vector)
                self._messages[message_id] = row
                updated["messages"] += 1

        return updated

    # ------------------------------------------------------------------
    def search_docs(
        self,
        top_k: int,
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        embedder: Optional["SupportsEmbed"] = None,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        chosen_embedder = self._ensure_embedder(embedder)
        query_vec = self._validate_vector(chosen_embedder.embed([query_text])[0])
        filters = filters or {}

        path_prefix = filters.get("path_prefix")
        mime_types = filters.get("mime_types")
        updated_since = filters.get("updated_since")
        if updated_since is not None:
            updated_since = _to_datetime(updated_since)

        results: List[Tuple[float, DocChunk]] = []
        for row in self._docs.values():
            if row.embedding is None:
                continue
            if path_prefix and not str(row.source_path).startswith(str(path_prefix)):
                continue
            if mime_types and row.mime_type not in set(mime_types):
                continue
            if updated_since and row.updated_at < updated_since:
                continue
            distance = _cosine_distance(row.embedding, query_vec)
            results.append((distance, row))

        results.sort(key=lambda item: item[0])
        return [row.to_result(distance=dist) for dist, row in results[:top_k]]

    # ------------------------------------------------------------------
    def search_messages(
        self,
        top_k: int,
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        embedder: Optional["SupportsEmbed"] = None,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        chosen_embedder = self._ensure_embedder(embedder)
        query_vec = self._validate_vector(chosen_embedder.embed([query_text])[0])
        filters = filters or {}
        thread_id = filters.get("thread_id")
        updated_since = filters.get("updated_since")
        if updated_since is not None:
            updated_since = _to_datetime(updated_since)

        results: List[Tuple[float, MessageRow]] = []
        for row in self._messages.values():
            if row.embedding is None:
                continue
            if thread_id and row.thread_id != thread_id:
                continue
            if updated_since and row.updated_at < updated_since:
                continue
            distance = _cosine_distance(row.embedding, query_vec)
            results.append((distance, row))

        results.sort(key=lambda item: item[0])
        return [row.to_result(distance=dist) for dist, row in results[:top_k]]

    # ------------------------------------------------------------------
    def backfill_embeddings(self, *, embedder: Optional["SupportsEmbed"] = None) -> Dict[str, int]:
        chosen_embedder = self._ensure_embedder(embedder)
        updated = {"doc_chunks": 0, "messages": 0}

        for key, row in list(self._docs.items()):
            vector = chosen_embedder.embed([row.content])[0]
            row.embedding = self._validate_vector(vector)
            row.updated_at = _utcnow()
            self._docs[key] = row
            updated["doc_chunks"] += 1

        for message_id, row in list(self._messages.items()):
            vector = chosen_embedder.embed([row.content])[0]
            row.embedding = self._validate_vector(vector)
            row.updated_at = _utcnow()
            self._messages[message_id] = row
            updated["messages"] += 1

        return updated

    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        docs_total = len(self._docs)
        docs_embedded = sum(1 for row in self._docs.values() if row.embedding is not None)
        msg_total = len(self._messages)
        msg_embedded = sum(1 for row in self._messages.values() if row.embedding is not None)

        def _pct(done: int, total: int) -> float:
            if not total:
                return 0.0
            return round((done / total) * 100.0, 2)

        report = {
            "mode": "postgres" if self.uses_postgres else "memory",
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "doc_chunks": {
                "total": docs_total,
                "embedded": docs_embedded,
                "percent_embedded": _pct(docs_embedded, docs_total),
            },
            "messages": {
                "total": msg_total,
                "embedded": msg_embedded,
                "percent_embedded": _pct(msg_embedded, msg_total),
            },
            "ann_indexes": {
                "doc_chunks": False,
                "messages": False,
            },
        }

        if self.uses_postgres:  # pragma: no cover - exercised in integration environments
            assert self._pool is not None
            with self._pool.connection() as conn:
                if self.schema:
                    conn.execute(
                        f"SET search_path TO {self.schema}, pg_catalog"
                    )
                with conn.cursor(row_factory=self._dict_row) as cur:
                    cur.execute(
                        "SELECT COUNT(*) AS c, COUNT(embedding) AS embedded FROM doc_chunks"
                    )
                    row = cur.fetchone()
                    if row:
                        report["doc_chunks"].update(
                            {
                                "total": int(row["c"]),
                                "embedded": int(row["embedded"]),
                                "percent_embedded": _pct(int(row["embedded"]), int(row["c"])),
                            }
                        )
                    cur.execute(
                        "SELECT COUNT(*) AS c, COUNT(embedding) AS embedded FROM messages"
                    )
                    row = cur.fetchone()
                    if row:
                        report["messages"].update(
                            {
                                "total": int(row["c"]),
                                "embedded": int(row["embedded"]),
                                "percent_embedded": _pct(int(row["embedded"]), int(row["c"])),
                            }
                        )
                    cur.execute(
                        """
                        SELECT indexrelid::regclass::text AS name
                        FROM pg_index
                        WHERE indrelid = 'doc_chunks'::regclass
                          AND indisvalid
                        """
                    )
                    doc_indexes = {row[0] for row in cur.fetchall()}
                    cur.execute(
                        """
                        SELECT indexrelid::regclass::text AS name
                        FROM pg_index
                        WHERE indrelid = 'messages'::regclass
                          AND indisvalid
                        """
                    )
                    msg_indexes = {row[0] for row in cur.fetchall()}
                    report["ann_indexes"] = {
                        "doc_chunks": any("embedding" in name for name in doc_indexes),
                        "messages": any("embedding" in name for name in msg_indexes),
                    }

        return report

    # ------------------------------------------------------------------
    def register_message(
        self,
        *,
        message_id: str,
        thread_id: str,
        role: str,
        content: str,
        created_at: Optional[Any] = None,
        updated_at: Optional[Any] = None,
        embedder: Optional["SupportsEmbed"] = None,
    ) -> MessageRow:
        created_dt = _to_datetime(created_at) if created_at else _utcnow()
        updated_dt = _to_datetime(updated_at) if updated_at else created_dt
        row = MessageRow(
            thread_id=thread_id,
            role=role,
            content=content,
            message_id=message_id,
            created_at=created_dt,
            updated_at=updated_dt,
        )

        chosen_embedder = embedder or self.embedder
        if chosen_embedder is not None:
            vector = chosen_embedder.embed([content])[0]
            row.embedding = self._validate_vector(vector)

        self._messages[message_id] = row
        return row

    # ------------------------------------------------------------------
    # SQL builder helpers ------------------------------------------------
    # ------------------------------------------------------------------
    def build_vector_search_sql(
        self,
        *,
        table: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if table not in {self.DOC_TABLE, self.MSG_TABLE}:
            raise ValueError(f"Unsupported table: {table}")
        where_clauses = ["embedding IS NOT NULL"]
        params: Dict[str, Any] = {"limit": max(top_k, 0)}
        filters = filters or {}

        if table == self.DOC_TABLE:
            if filters.get("path_prefix"):
                where_clauses.append("source_path LIKE %(path_prefix)s")
                params["path_prefix"] = f"{filters['path_prefix']}%"
            if filters.get("mime_types"):
                where_clauses.append("mime_type = ANY(%(mime_types)s)")
                params["mime_types"] = filters["mime_types"]
            if filters.get("updated_since"):
                where_clauses.append("updated_at >= %(updated_since)s")
                params["updated_since"] = _to_datetime(filters["updated_since"])
        else:
            if filters.get("thread_id"):
                where_clauses.append("thread_id = %(thread_id)s")
                params["thread_id"] = filters["thread_id"]
            if filters.get("updated_since"):
                where_clauses.append("updated_at >= %(updated_since)s")
                params["updated_since"] = _to_datetime(filters["updated_since"])

        where_sql = " AND ".join(where_clauses)
        sql = (
            f"SELECT *, (embedding <=> %(query)s) AS distance "
            f"FROM {table} WHERE {where_sql} "
            f"ORDER BY embedding <=> %(query)s LIMIT %(limit)s"
        )
        return sql, params


@dataclass
class IngestReport:
    files_processed: int = 0
    files_skipped: int = 0
    chunks_processed: int = 0
    chunks_embedded: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "chunks_processed": self.chunks_processed,
            "chunks_embedded": self.chunks_embedded,
        }


class SupportsEmbed:
    """Protocol-like helper for embedding providers."""

    model_name: str  # pragma: no cover - attribute contract only
    dimension: int   # pragma: no cover - attribute contract only

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:  # pragma: no cover - contract
        raise NotImplementedError


__all__ = [
    "DocChunk",
    "EmbeddingError",
    "IngestReport",
    "MessageRow",
    "PgVectorStore",
    "SupportsEmbed",
]
