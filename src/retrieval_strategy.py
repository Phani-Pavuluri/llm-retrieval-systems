"""
Rule-based retrieval mode and hybrid weight selection.

Inspects task_type + query text only (no LLM). Replace with learned or LLM
routing later; keep the apply_strategy_to_request entrypoint.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.retrieval_request import RetrievalRequest

# Exact-issue lookups: lexical match helps more than broad embedding similarity.
_ISSUE_KEYWORDS = (
    "counterfeit",
    "fake",
    "defect",
    "defective",
    "rash",
    "burn",
    "chunks",
    "texture",
)

# Broad / summarization phrasing: prefer semantics (or very light lexical fuse).
_SUMMARY_PHRASES = (
    "summary",
    "common complaints",
    "overall issues",
    "negative experiences",
    "worth it",
)


@dataclass(frozen=True)
class RetrievalStrategy:
    use_hybrid: bool
    hybrid_alpha: float | None  # None → config defaults when hybrid on
    hybrid_beta: float | None
    reason: str


def select_retrieval_strategy(task_type: str, query_text: str) -> RetrievalStrategy:
    q = (query_text or "").lower()

    if any(kw in q for kw in _ISSUE_KEYWORDS):
        return RetrievalStrategy(
            use_hybrid=True,
            hybrid_alpha=0.6,
            hybrid_beta=0.4,
            reason="issue_keywords",
        )

    if any(phrase in q for phrase in _SUMMARY_PHRASES):
        return RetrievalStrategy(
            use_hybrid=False,
            hybrid_alpha=None,
            hybrid_beta=None,
            reason="abstract_summary_vector",
        )

    if task_type == "complaint_summary":
        return RetrievalStrategy(
            use_hybrid=True,
            hybrid_alpha=0.85,
            hybrid_beta=0.15,
            reason="complaint_task_light_hybrid",
        )

    return RetrievalStrategy(
        use_hybrid=False,
        hybrid_alpha=None,
        hybrid_beta=None,
        reason="default_vector",
    )


def _candidate_pool_multiplier_for_request(
    request: RetrievalRequest, strategy_reason: str
) -> float:
    """Larger pools for issue-style queries and tight rating filters (traceable, lightly tuned)."""
    if strategy_reason == "issue_keywords":
        return 1.5
    if request.filters and "review_rating" in request.filters:
        return 1.4
    if strategy_reason == "abstract_summary_vector":
        return 1.0
    if strategy_reason == "complaint_task_light_hybrid":
        return 1.2
    return 1.0


def apply_strategy_to_request(request: RetrievalRequest) -> None:
    """Mutates request with selected mode, hybrid weights, and pool multiplier."""
    s = select_retrieval_strategy(request.task_type, request.query_text)
    request.use_hybrid = s.use_hybrid
    request.hybrid_alpha = s.hybrid_alpha
    request.hybrid_beta = s.hybrid_beta
    request.strategy_reason = s.reason
    request.candidate_pool_multiplier = _candidate_pool_multiplier_for_request(
        request, s.reason
    )
