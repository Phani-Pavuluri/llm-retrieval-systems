"""
Task-aware selective reranking (rule-based). Does not change the cross-encoder model.

Policy is driven by query_family (when set), else heuristics from task_type, filters,
and retrieval strategy_reason. Explicit request.use_rerank True/False is never overwritten.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src import config
from src.retrieval_request import RetrievalRequest

# Labeled-eval-aligned families: rerank ON / OFF from evaluation evidence.
RERANK_ON_QUERY_FAMILIES = frozenset(
    {"value_complaint", "abstract_complaint_summary", "buyer_risk_issues"}
)
RERANK_OFF_QUERY_FAMILIES = frozenset(
    {"rating_scoped_summary", "exact_issue_lookup", "symptom_issue_extraction"}
)

# Thematic / broad complaint summaries (not price-focused).
_SUMMARY_PHRASES = (
    "summary",
    "common complaints",
    "overall issues",
    "negative experiences",
    "themes in",
)

# Only when the user clearly asks about money, worth, or value framing.
_VALUE_INTENT_MARKERS = (
    "value for money",
    "good value",
    "worth the price",
    "worth it",
    "overpriced",
    "price point",
    "too expensive",
    "cheap enough",
    "quality for the price",
    "value-related",
    "for the price",
    "is it worth",
    "worth the",
)

# Narrow defect / authenticity / attribute lookups (not structured symptom lists).
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

# List- or extraction-style symptom / skin-reaction questions (Phase 4.5).
_SYMPTOM_SKIN_TERMS = (
    "rash",
    "irritation",
    "skin irritation",
    "burning",
    "itch",
    "hives",
    "bumps",
    "blisters",
    "allergic reaction",
)

_BUYER_RISK_MARKERS = (
    "watch out",
    "watchout",
    "beware",
    "red flag",
    "red flags",
    "caution",
    "what problems should",
    "problems should a buyer",
    "buyer watch",
)


def _has_rating_filter(filters: dict[str, Any]) -> bool:
    return bool(filters.get("review_rating"))


def _wants_symptom_issue_extraction(lower: str) -> bool:
    if "symptom" in lower:
        return True
    if "specific symptoms" in lower:
        return True
    if "list" in lower and any(t in lower for t in _SYMPTOM_SKIN_TERMS):
        return True
    if "if mentioned" in lower and any(t in lower for t in _SYMPTOM_SKIN_TERMS):
        return True
    return False


def _wants_buyer_risk_issues(lower: str) -> bool:
    if any(m in lower for m in _BUYER_RISK_MARKERS):
        return True
    if "buyer" in lower and "problem" in lower:
        return True
    if "before i buy" in lower or "before buying" in lower:
        return True
    return False


def _wants_value_complaint(lower: str) -> bool:
    return any(m in lower for m in _VALUE_INTENT_MARKERS)


def infer_query_family(request: RetrievalRequest) -> str:
    """Best-effort family label for policy (matches eval taxonomy when parser/trace set it)."""
    if request.query_family:
        return str(request.query_family)
    if _has_rating_filter(request.filters):
        return "rating_scoped_summary"
    q = (request.query_text or "").lower()

    # Symptom / skin-reaction extraction lists (before generic issue keywords catch "rash").
    if _wants_symptom_issue_extraction(q):
        return "symptom_issue_extraction"

    # Explicit price / worth / value questions (before broad complaint heuristics).
    if _wants_value_complaint(q):
        return "value_complaint"

    # General buyer risks — not value framing unless value markers matched above.
    if _wants_buyer_risk_issues(q):
        return "buyer_risk_issues"

    if any(p in q for p in _SUMMARY_PHRASES):
        return "abstract_complaint_summary"
    if any(k in q for k in _ISSUE_KEYWORDS):
        return "exact_issue_lookup"
    # Generic complaint / issue task: thematic complaints, not automatic value_complaint.
    if request.task_type == "complaint_summary":
        return "abstract_complaint_summary"
    return "unknown"


def apply_selective_rerank_policy(
    request: RetrievalRequest,
    *,
    selective_enabled: bool | None = None,
    query_family_override: str | None = None,
) -> None:
    """
    When selective_enabled (default: config.RERANK_SELECTIVE), set request.use_rerank
    and request.rerank_reason from rules if request.use_rerank is None.

    Explicit use_rerank True/False is never changed here (caller should set rerank_reason).
    """
    if request.use_rerank is not None:
        if not request.rerank_reason:
            request.rerank_reason = "explicit_request"
        return

    sel = config.RERANK_SELECTIVE if selective_enabled is None else bool(selective_enabled)
    if not sel:
        return

    if query_family_override:
        request.query_family = query_family_override

    request.rerank_skipped_due_to_query_family = False
    request.rerank_skipped_due_to_metadata_filters = False

    if _has_rating_filter(request.filters):
        request.use_rerank = False
        request.rerank_reason = "selective:metadata_rating_filter"
        request.rerank_skipped_due_to_metadata_filters = True
        return

    fam = infer_query_family(request)

    if fam in RERANK_OFF_QUERY_FAMILIES:
        request.use_rerank = False
        request.rerank_reason = f"selective:family_off:{fam}"
        request.rerank_skipped_due_to_query_family = True
        return

    if fam in RERANK_ON_QUERY_FAMILIES:
        request.use_rerank = True
        request.rerank_reason = f"selective:family_on:{fam}"
        return

    # Heuristics when family unknown / unmatched
    reason = request.strategy_reason or ""
    if reason == "abstract_summary_vector":
        request.use_rerank = True
        request.rerank_reason = "selective:heuristic:abstract_summary_vector"
        return
    if reason == "issue_keywords":
        request.use_rerank = False
        request.rerank_reason = "selective:heuristic:issue_keywords"
        request.rerank_skipped_due_to_query_family = True
        return
    if reason == "complaint_task_light_hybrid" and request.task_type == "complaint_summary":
        request.use_rerank = True
        request.rerank_reason = "selective:heuristic:complaint_task_light_hybrid"
        return

    # Conservative default: do not rerank unknown shapes
    request.use_rerank = False
    request.rerank_reason = f"selective:default_off:family={fam}"


def should_skip_rerank_for_confidence(df_pre_rerank: pd.DataFrame) -> bool:
    """If top fused score is already very high, skip second-stage rerank (hybrid paths only)."""
    thr = getattr(config, "RERANK_SKIP_IF_TOP_SCORE_AT_LEAST", None)
    if thr is None:
        return False
    if df_pre_rerank.empty:
        return False
    if not bool(getattr(config, "RERANK_CONFIDENCE_REQUIRES_HYBRID_SCORES", True)):
        sc = float(df_pre_rerank.iloc[0].get("score", 0.0))
        return sc >= float(thr)
    # Hybrid produces semantic_score + keyword_score columns on the frame
    if "semantic_score" not in df_pre_rerank.columns:
        return False
    sc = float(df_pre_rerank.iloc[0].get("score", 0.0))
    return sc >= float(thr)


def build_rerank_trace_decision(
    request: RetrievalRequest,
    *,
    rerank_requested_before_confidence: bool,
    rerank_applied: bool,
    rerank_skipped_due_to_confidence: bool,
    rerank_top_n_effective: int,
) -> dict[str, Any]:
    """Fields merged into retrieval trace JSON."""
    return {
        "use_rerank_requested": bool(rerank_requested_before_confidence),
        "use_rerank_effective": bool(rerank_applied),
        "rerank_reason": getattr(request, "rerank_reason", "") or "",
        "rerank_skipped_due_to_confidence": bool(rerank_skipped_due_to_confidence),
        "rerank_skipped_due_to_query_family": bool(
            getattr(request, "rerank_skipped_due_to_query_family", False)
        ),
        "rerank_skipped_due_to_metadata_filters": bool(
            getattr(request, "rerank_skipped_due_to_metadata_filters", False)
        ),
        "rerank_top_n_effective": int(rerank_top_n_effective),
        "query_family_effective": infer_query_family(request),
    }
