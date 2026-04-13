"""
Hybrid ranking on top of vector candidates: normalize semantic + keyword scores, fuse.

Keeps vector-only path untouched when hybrid is disabled (see Retriever).
"""
from __future__ import annotations

import re

import pandas as pd


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def keyword_overlap_score(query: str, document: str) -> float:
    """Fraction of distinct query tokens that appear in the document (0–1)."""
    q = tokenize(query)
    if not q:
        return 0.0
    d = tokenize(document)
    return len(q & d) / len(q)


def _min_max_normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo = float(s.min())
    hi = float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(1.0, index=s.index)
    return (s - lo) / (hi - lo)


def apply_hybrid_scoring(
    df: pd.DataFrame,
    query_text: str,
    alpha: float,
    beta: float,
    text_col: str = "text",
    semantic_col: str = "score",
) -> pd.DataFrame:
    """
    Expects FAISS similarity in semantic_col (higher = better).
    Adds semantic_score, keyword_score; overwrites score with fused ranking score.
    """
    out = df.copy()
    out["semantic_score"] = out[semantic_col].astype(float)
    out["keyword_score"] = out[text_col].map(
        lambda t: keyword_overlap_score(query_text, str(t))
    )
    sem_n = _min_max_normalize(out["semantic_score"])
    kw_n = _min_max_normalize(out["keyword_score"])
    out["score"] = float(alpha) * sem_n + float(beta) * kw_n
    return out
