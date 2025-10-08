from __future__ import annotations

from typing import Any, Dict, Sequence

import pytest

from homeai.embeddings import OllamaEmbedder


class _StubResponse:
    def __init__(self, *, payload: Dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise ValueError("boom")

    def json(self) -> Dict[str, Any]:
        return self._payload


class _StubSession:
    def __init__(self, embeddings: Sequence[Sequence[float]]) -> None:
        self._embeddings = embeddings
        self.closed = False
        self.calls = []

    def post(self, url: str, json: Dict[str, Any], timeout: float):
        self.calls.append((url, json, timeout))
        index = len(self.calls) - 1
        payload = {"embedding": list(self._embeddings[index])}
        return _StubResponse(payload=payload)

    def close(self) -> None:
        self.closed = True


def test_embedder_validates_timeout() -> None:
    with pytest.raises(ValueError):
        OllamaEmbedder(model_name="test", host="http://localhost", dimension=3, timeout=0)


def test_embedder_context_manager_closes_session() -> None:
    session = _StubSession([[1.0, 2.0, 3.0]])
    with OllamaEmbedder(
        model_name="test",
        host="http://localhost",
        dimension=3,
        session=session,
    ) as embedder:
        result = embedder.embed(["hello"])

    assert session.closed is True
    assert result == [[1.0, 2.0, 3.0]]


def test_embedder_detects_dimension_mismatch() -> None:
    session = _StubSession([[1.0, 2.0]])
    embedder = OllamaEmbedder(
        model_name="test",
        host="http://localhost",
        dimension=3,
        session=session,
    )

    with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
        embedder.embed(["hello"])

