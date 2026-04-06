from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI
except ImportError:  # optional dependency for OpenAIEmbeddings
    OpenAI = None  # type: ignore[misc, assignment]


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        pass


class SentenceTransformerEmbeddings(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.astype("float32")


def _l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row; matches cosine / inner-product search when using IndexFlatIP."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vectors / norms).astype("float32", copy=False)


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI API embeddings (`text-embedding-3-small`, `text-embedding-ada-002`, etc.).

    Set ``OPENAI_API_KEY`` in the environment, or pass ``api_key=...``.

    Embeddings are L2-normalized by default so they behave like
    ``SentenceTransformerEmbeddings`` with ``normalize_embeddings=True`` for FAISS IP search.
    """

    # OpenAI allows large batches; keep moderate for payload size and retries.
    _DEFAULT_BATCH_SIZE = 256

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        dimensions: Optional[int] = None,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "OpenAIEmbeddings requires the 'openai' package. "
                "Install with: pip install openai"
            )
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAIEmbeddings: set OPENAI_API_KEY or pass api_key=..."
            )
        self._model = model
        self._normalize = normalize
        self._batch_size = max(1, batch_size)
        self._dimensions = dimensions
        kwargs = {"api_key": key}
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        create_kwargs = {"model": self._model, "input": list(texts)}
        if self._dimensions is not None:
            create_kwargs["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**create_kwargs)
        ordered = sorted(response.data, key=lambda item: item.index)
        mat = np.array([row.embedding for row in ordered], dtype=np.float32)
        if self._normalize:
            mat = _l2_normalize_rows(mat)
        return mat

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        parts: List[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            parts.append(self._encode_batch(batch))
        return np.vstack(parts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode_batch([text])


def build_embedder_from_config() -> EmbeddingProvider:
    """Construct the configured `EmbeddingProvider` (see `src.config`)."""
    from src import config

    backend = getattr(config, "EMBEDDING_BACKEND", "sentence_transformers").lower()
    if backend == "openai":
        return OpenAIEmbeddings(model=getattr(config, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    if backend == "sentence_transformers":
        return SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}")