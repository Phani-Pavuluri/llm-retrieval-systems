"""Structured retrieval request passed between parser, retriever, and pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalRequest:
    """Represents a single vector retrieval operation (dataset-agnostic)."""

    query_text: str
    top_k: int
    filters: dict[str, Any] = field(default_factory=dict)
    task_type: str = "general_qa"
    original_query: str | None = None
    # None → use config.HYBRID_RETRIEVAL; True/False forces hybrid vs vector-only ranking
    use_hybrid: bool | None = None
    # When hybrid: None → config HYBRID_ALPHA / HYBRID_BETA
    hybrid_alpha: float | None = None
    hybrid_beta: float | None = None
    strategy_reason: str = ""
    # Applied to oversample formula: None/1.0 = no extra pull (see retriever._candidate_count)
    candidate_pool_multiplier: float | None = None
    # None → config.RERANK_ENABLED
    use_rerank: bool | None = None
    # None → config.RERANK_TOP_N (max rows after filter passed to reranker)
    rerank_top_n: int | None = None
    # None → config.RERANK_MODEL (cross-encoder id for sentence-transformers)
    rerank_model: str | None = None
    # Optional labeled-eval / parser hint for selective rerank (see rerank_policy).
    query_family: str | None = None
    # Why rerank was on/off (e.g. selective:family_on:value_complaint, cli_override)
    rerank_reason: str = ""
    rerank_skipped_due_to_query_family: bool = False
    rerank_skipped_due_to_metadata_filters: bool = False

    def __post_init__(self) -> None:
        if self.original_query is None:
            self.original_query = self.query_text

    @classmethod
    def from_raw(cls, query: str, top_k: int) -> RetrievalRequest:
        """Build a request without parsing (backward-compatible / bypass path)."""
        return cls(
            query_text=query,
            top_k=top_k,
            filters={},
            task_type="general_qa",
            original_query=query,
            use_hybrid=None,
            hybrid_alpha=None,
            hybrid_beta=None,
            strategy_reason="",
            candidate_pool_multiplier=None,
            use_rerank=None,
            rerank_top_n=None,
            rerank_model=None,
            query_family=None,
            rerank_reason="",
            rerank_skipped_due_to_query_family=False,
            rerank_skipped_due_to_metadata_filters=False,
        )
