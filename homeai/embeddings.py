from __future__ import annotations

import json
import logging
from typing import List, Optional, Sequence

import requests

from .pgvector_store import SupportsEmbed


class OllamaEmbedder(SupportsEmbed):
    """Embedding client that talks to a local Ollama host."""

    def __init__(
        self,
        *,
        model_name: str,
        host: str,
        dimension: int,
        timeout: float = 60.0,
        session: Optional[requests.Session] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be provided")
        if not host:
            raise ValueError("host must be provided")
        self.model_name = model_name
        self.dimension = dimension
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._logger = logger or logging.getLogger(__name__)
        if session is None:
            self._session = requests.Session()
            self._close_session = self._session.close
        else:
            self._session = session
            self._close_session = getattr(session, "close", lambda: None)

    def close(self) -> None:
        self._close_session()

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for text in texts:
            prompt = text or ""
            payload = {"model": self.model_name, "prompt": prompt}
            try:
                response = self._session.post(
                    f"{self.host}/api/embeddings",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                self._logger.error("Ollama embedding request failed: %s", exc)
                raise RuntimeError(f"Embedding request failed: {exc}") from exc
            except ValueError as exc:
                self._logger.error("Failed to decode embedding response: %s", exc)
                raise RuntimeError("Invalid embedding response payload") from exc

            embedding = None
            if isinstance(data, dict):
                embedding = data.get("embedding")
            if embedding is None:
                self._logger.error("Embedding response missing 'embedding': %s", json.dumps(data)[:400])
                raise RuntimeError("Embedding response missing 'embedding' field")
            try:
                vector = [float(x) for x in embedding]
            except (TypeError, ValueError) as exc:
                self._logger.error("Embedding response had invalid values: %s", exc)
                raise RuntimeError("Embedding response contained non-numeric values") from exc

            if self.dimension and len(vector) != self.dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )

            vectors.append(vector)

        return vectors


__all__ = ["OllamaEmbedder"]
