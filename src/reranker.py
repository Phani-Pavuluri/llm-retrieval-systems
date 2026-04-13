"""
Optional second-stage (query, chunk) reranking. Modular: swap BaseReranker impl later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src import config


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query_text: str, candidates: pd.DataFrame) -> pd.DataFrame:
        """Return same rows reordered with new column `rerank_score` (higher = better)."""


class CrossEncoderReranker(BaseReranker):
    """Local cross-encoder (sentence-transformers)."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or config.RERANK_MODEL
        self._model = None

    def _lazy_model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query_text: str, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty or "text" not in candidates.columns:
            out = candidates.copy()
            if not out.empty and "rerank_score" not in out.columns:
                out["rerank_score"] = 0.0
            return out

        out = candidates.copy()
        texts = out["text"].fillna("").astype(str).tolist()
        pairs = [(query_text, t) for t in texts]
        bs = getattr(config, "RERANK_BATCH_SIZE", 16) or 16
        scores = self._lazy_model().predict(pairs, batch_size=int(bs))
        out["rerank_score"] = [float(s) for s in scores]
        out = out.sort_values("rerank_score", ascending=False).reset_index(drop=True)
        return out


def effective_rerank(request) -> bool:  # type: ignore[no-untyped-def]
    ur = getattr(request, "use_rerank", None)
    if ur is not None:
        return bool(ur)
    return bool(config.RERANK_ENABLED)


def effective_rerank_model(request) -> str:  # type: ignore[no-untyped-def]
    """Cross-encoder id: request override, else config.RERANK_MODEL."""
    rm = getattr(request, "rerank_model", None)
    if rm is not None and str(rm).strip():
        return str(rm).strip()
    return str(config.RERANK_MODEL)


def try_load_cross_encoder(model_name: str) -> tuple[bool, str | None]:
    """
    Try to construct a CrossEncoder (may download weights). For experiment probes.
    Returns (ok, error_message_if_failed).
    """
    try:
        from sentence_transformers import CrossEncoder

        CrossEncoder(model_name)
        return True, None
    except Exception as e:  # pragma: no cover - optional dependency / OOM paths
        return False, f"{type(e).__name__}: {e}"


def rerank_top_n_for_request(request) -> int:  # type: ignore[no-untyped-def]
    rn = getattr(request, "rerank_top_n", None)
    if rn is not None:
        return int(rn)
    return int(config.RERANK_TOP_N)


def apply_rerank_to_candidates(
    reranker: BaseReranker,
    query_text: str,
    candidates: pd.DataFrame,
    final_top_k: int,
) -> pd.DataFrame:
    """Rerank full candidate frame, then keep top `final_top_k` rows."""
    ranked = reranker.rerank(query_text, candidates)
    out = ranked.head(int(final_top_k)).reset_index(drop=True)
    out["final_rank"] = range(1, len(out) + 1)
    return out
