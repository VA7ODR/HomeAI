from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pytest

from homeai import config
from homeai import filesystem
from homeai.pgvector_store import EmbeddingError, PgVectorStore


class FakeEmbedder:
    def __init__(self, *, dimension: int = 8, model_name: str = "fake-embed") -> None:
        self.dimension = dimension
        self.model_name = model_name
        self.calls = 0

    def embed(self, texts):
        vectors = []
        for text in texts:
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=self.dimension * 4).digest()
            vector = []
            for i in range(self.dimension):
                chunk = digest[i * 4 : (i + 1) * 4]
                value = int.from_bytes(chunk, "little") / 0xFFFFFFFF
                vector.append((value * 2.0) - 1.0)
            vectors.append(vector)
            self.calls += 1
        return vectors


def _make_store(tmp_path: Path, monkeypatch, *, dimension: int = 8) -> PgVectorStore:
    embedder = FakeEmbedder(dimension=dimension)
    monkeypatch.setattr(config, "BASE", tmp_path)
    monkeypatch.setattr(filesystem.config, "BASE", tmp_path)
    store = PgVectorStore(
        dsn=None,
        embedder=embedder,
        embedding_dimension=dimension,
        embedding_model=embedder.model_name,
        chunk_size=64,
        chunk_overlap=16,
    )
    return store


def test_ingest_and_search_docs(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path, monkeypatch)
    embedder = store.embedder  # type: ignore[assignment]
    assert embedder is not None

    file_a = tmp_path / "notes" / "alpha.txt"
    file_a.parent.mkdir(parents=True, exist_ok=True)
    file_a.write_text("OAuth settings live in config/oauth.yaml. Remember the callback URI.")

    file_b = tmp_path / "notes" / "beta.txt"
    file_b.write_text("Rate limiting discussions happened last spring in standup logs.")

    report = store.ingest_files([tmp_path], source_kind="repo", embedder=embedder)
    assert report.files_processed == 2
    assert report.chunks_processed >= 2
    first_call_count = embedder.calls
    assert report.chunks_embedded == report.chunks_processed

    # Re-ingesting without changes should not trigger additional embeddings.
    second_report = store.ingest_files([tmp_path], source_kind="repo", embedder=embedder)
    assert second_report.chunks_embedded == 0
    assert embedder.calls == first_call_count

    hits = store.search_docs(
        top_k=3,
        query_text="Where are the OAuth settings defined?",
        filters={"path_prefix": str(tmp_path / "notes")},
        embedder=embedder,
    )

    assert hits, "expected at least one semantic match"
    top_hit = hits[0]
    assert top_hit["source_path"].endswith("alpha.txt")
    assert 0.0 <= top_hit["score"] <= 1.0
    assert top_hit["distance"] is not None


def test_dimension_mismatch_backfill(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path, monkeypatch, dimension=6)
    good_embedder = store.embedder  # type: ignore[assignment]
    assert good_embedder is not None

    file_path = tmp_path / "doc.txt"
    file_path.write_text("Semantic embeddings with pgvector.")
    store.ingest_files([file_path], embedder=good_embedder)

    wrong_embedder = FakeEmbedder(dimension=4)
    with pytest.raises(EmbeddingError):
        store.backfill_embeddings(embedder=wrong_embedder)


def test_build_vector_search_sql_uses_cosine_operator(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path, monkeypatch)
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    sql, params = store.build_vector_search_sql(
        table=store.DOC_TABLE,
        top_k=8,
        filters={
            "path_prefix": str(tmp_path / "configs"),
            "mime_types": ["text/plain"],
            "updated_since": since,
        },
    )

    assert "embedding <=> %(query)s" in sql
    assert "ORDER BY embedding <=> %(query)s" in sql
    assert params["path_prefix"].endswith("%")
    assert params["mime_types"] == ["text/plain"]
    assert isinstance(params["updated_since"], datetime)


def test_embed_missing_handles_messages(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path, monkeypatch)
    embedder = store.embedder  # type: ignore[assignment]
    assert embedder is not None

    store.embedder = None  # simulate message saved before the embedder is configured
    store.register_message(
        message_id="m1",
        thread_id="thread-1",
        role="user",
        content="TLS failures were mitigated by rotating certificates.",
        embedder=None,
    )
    store.embedder = embedder

    counts = store.embed_missing(embedder=embedder)
    assert counts["messages"] == 1

    hits = store.search_messages(
        top_k=1,
        query_text="How did we fix TLS failures?",
        filters={"thread_id": "thread-1"},
        embedder=embedder,
    )
    assert hits and hits[0]["message_id"] == "m1"


def test_register_message_continues_on_embedding_failure(tmp_path: Path, monkeypatch, caplog) -> None:
    store = _make_store(tmp_path, monkeypatch)

    class BoomEmbedder(FakeEmbedder):
        def embed(self, texts):  # type: ignore[override]
            super().embed(texts)
            raise RuntimeError("boom")

    failing_embedder = BoomEmbedder(dimension=store.embedding_dimension)
    store.embedder = failing_embedder  # type: ignore[assignment]

    with caplog.at_level("WARNING"):
        row = store.register_message(
            message_id="m-fail",
            thread_id="thread-x",
            role="assistant",
            content="Message that should persist even when embedding fails.",
        )

    assert row.message_id == "m-fail"
    assert row.embedding is None
    assert "Failed to embed message" in caplog.text
    assert store._messages["m-fail"].content.startswith("Message that")


def test_ingest_files_records_chunk_when_embedding_fails(tmp_path: Path, monkeypatch, caplog) -> None:
    store = _make_store(tmp_path, monkeypatch)

    class BoomEmbedder(FakeEmbedder):
        def embed(self, texts):  # type: ignore[override]
            super().embed(texts)
            raise RuntimeError("boom")

    failing_embedder = BoomEmbedder(dimension=store.embedding_dimension)

    doc = tmp_path / "docs" / "note.txt"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("The embedding service is offline right now.")

    with caplog.at_level("WARNING"):
        report = store.ingest_files([doc], embedder=failing_embedder)

    assert report.files_processed == 1
    assert report.chunks_processed == 1
    assert report.chunks_embedded == 0
    assert "Failed to embed chunk" in caplog.text
    key = (str(doc), 0)
    assert store._docs[key].content.startswith("The embedding")
    assert store._docs[key].embedding is None
