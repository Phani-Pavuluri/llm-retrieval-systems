"""
Retrieval evaluation metrics for ranked chunk_id lists vs gold sets.

Precision@k, Recall@k (when gold non-empty), MRR.
"""
from __future__ import annotations


def precision_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float | None:
    """|top-k ∩ gold| / k. None if k <= 0."""
    if k <= 0:
        return None
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for cid in top if cid in gold_ids)
    return hits / k


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float | None:
    """|top-k ∩ gold| / |gold|. None if gold empty."""
    if not gold_ids:
        return None
    top = set(ranked_ids[:k])
    return len(top & gold_ids) / len(gold_ids)


def mean_reciprocal_rank(ranked_ids: list[str], gold_ids: set[str]) -> float:
    """Reciprocal rank of first gold hit; 0.0 if none."""
    if not gold_ids:
        return 0.0
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in gold_ids:
            return 1.0 / i
    return 0.0


def aggregate_mean(values: list[float | None]) -> float | None:
    """Mean of non-None values; None if empty."""
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)
