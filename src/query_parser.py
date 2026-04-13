"""
Rule-based query → RetrievalRequest.

Designed to be replaced or augmented later (e.g. LLM-based parsing, routing).
"""
from __future__ import annotations

import re
from typing import Any

from src.rerank_policy import infer_query_family
from src.retrieval_request import RetrievalRequest

# Keywords → task_type (extend without touching retriever / LLM)
_COMPLAINT_TASK_KEYWORDS = (
    "complaint",
    "complaints",
    "issue",
    "issues",
    "defect",
    "defects",
    "defective",
    "counterfeit",
    "fake",
    "problem",
    "problems",
    "bad",
    "negative",
)

# Filter keys are metadata column names; retriever ignores unknown keys.
_RATING_COLUMN = "review_rating"
# Only "from" / "by" — "for …" matches intents like "for sensitive skin" too often.
_BRAND_CAPTURE = re.compile(
    r"(?i)\b(?:from|by)\s+([A-Z0-9][\w'&.-]*(?:\s+[A-Z0-9][\w'&.-]*){0,3})"
)


class QueryParser:
    """Convert natural language into a structured RetrievalRequest (rules only)."""

    def parse(self, user_query: str, top_k: int = 5) -> RetrievalRequest:
        text = (user_query or "").strip()
        filters: dict[str, Any] = {}
        task_type = self._infer_task_type(text)
        self._maybe_add_rating_filters(text, filters)
        self._maybe_add_brand_filter(text, filters)

        req = RetrievalRequest(
            query_text=text,
            top_k=top_k,
            filters=filters,
            task_type=task_type,
            original_query=user_query,
            rerank_model=None,
            query_family=None,
            rerank_reason="",
            rerank_skipped_due_to_query_family=False,
            rerank_skipped_due_to_metadata_filters=False,
        )
        req.query_family = infer_query_family(req)
        return req

    def _infer_task_type(self, text: str) -> str:
        lower = text.lower()
        for kw in _COMPLAINT_TASK_KEYWORDS:
            if kw in lower:
                return "complaint_summary"
        return "general_qa"

    def _maybe_add_rating_filters(self, text: str, filters: dict[str, Any]) -> None:
        lower = text.lower()
        # Literal one-star intent — must run before broad "negative" (e.g. "negative one-star experiences").
        # Use equality so retrieval matches review_rating == 1 (not max<=1 with float edge cases).
        if any(
            p in lower
            for p in ("1-star", "1 star", "one star", "one-star", "single star", "single-star")
        ):
            filters[_RATING_COLUMN] = 1
            return
        if any(p in lower for p in ("low-rated", "low rated", "bad reviews")):
            # Cap at 3 stars for "low" phrasing (heuristic)
            filters[_RATING_COLUMN] = {"max": 3}
            return
        # "negative" alone (no explicit one-star above) — still treat as low-rating window
        if "negative" in lower:
            filters[_RATING_COLUMN] = {"max": 3}

    def _maybe_add_brand_filter(self, text: str, filters: dict[str, Any]) -> None:
        m = _BRAND_CAPTURE.search(text)
        if not m:
            return
        candidate = m.group(1).strip()
        if len(candidate) < 2:
            return
        # Heuristic column name; absent columns are ignored downstream
        filters["brand"] = candidate
