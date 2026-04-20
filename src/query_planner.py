"""
Optional LLM-assisted *retrieval planning* (Phase 5.x).

Produces a small JSON plan: optional retrieval_query_text, optional review_rating filter,
and optional query_family hint. Output is validated and merged conservatively.

This does not modify retriever/reranker internals; it only adjusts RetrievalRequest fields
before retrieval runs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.llm import BaseLLM
from src.query_parser import QueryParser
from src.rerank_policy import infer_query_family
from src.retrieval_request import RetrievalRequest

_ALLOWED_QUERY_FAMILIES = frozenset(
    {
        "abstract_complaint_summary",
        "value_complaint",
        "exact_issue_lookup",
        "rating_scoped_summary",
        "buyer_risk_issues",
        "symptom_issue_extraction",
    }
)


@dataclass(frozen=True)
class QueryPlanResult:
    applied: bool
    source: str  # "none" | "llm" | "llm_failed"
    retrieval_query_text: str | None
    filters_patch: dict[str, Any]
    query_family: str | None
    raw_model_text: str | None
    notes: str | None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    s = text.strip()
    # tolerate ```json fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.S | re.I)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
    except Exception:
        # try first {...} slice
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


def _coerce_rating_filter(val: Any) -> Any | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        ri = int(val)
        if 1 <= ri <= 5:
            return ri
        return None
    if isinstance(val, dict):
        out: dict[str, int] = {}
        if "min" in val and val["min"] is not None:
            try:
                lo = int(val["min"])
                if 1 <= lo <= 5:
                    out["min"] = lo
            except (TypeError, ValueError):
                pass
        if "max" in val and val["max"] is not None:
            try:
                hi = int(val["max"])
                if 1 <= hi <= 5:
                    out["max"] = hi
            except (TypeError, ValueError):
                pass
        if not out:
            return None
        if "min" in out and "max" in out and out["min"] > out["max"]:
            return None
        return out
    return None


def _validate_plan_obj(obj: dict[str, Any]) -> dict[str, Any] | None:
    """
    Returns a cleaned dict subset or None if unusable.
    Allowed keys: retrieval_query_text, review_rating, query_family, needs_low_rating_evidence
    """
    out: dict[str, Any] = {}
    rq = obj.get("retrieval_query_text")
    if isinstance(rq, str) and rq.strip():
        t = re.sub(r"\s+", " ", rq.strip())
        if len(t) > 800:
            t = t[:800]
        out["retrieval_query_text"] = t

    rr = _coerce_rating_filter(obj.get("review_rating"))
    if rr is not None:
        out["review_rating"] = rr

    if obj.get("needs_low_rating_evidence") is True and "review_rating" not in out:
        out["review_rating"] = {"max": 3}

    fam = obj.get("query_family")
    if isinstance(fam, str) and fam.strip() in _ALLOWED_QUERY_FAMILIES:
        out["query_family"] = fam.strip()

    return out or None


def _build_planner_prompt(user_query: str, baseline_filters: dict[str, Any]) -> str:
    bf = json.dumps(baseline_filters or {}, sort_keys=True)
    fams = ", ".join(sorted(_ALLOWED_QUERY_FAMILIES))
    return (
        "You are a retrieval planner for a product-review RAG system.\n\n"
        f"User query:\n{user_query}\n\n"
        "Baseline metadata filters already extracted by rules (may be empty):\n"
        f"{bf}\n\n"
        "Return ONLY a single JSON object (no markdown, no commentary) with any of these keys:\n"
        "- retrieval_query_text (string, optional): a concise paraphrase optimized for vector retrieval / keyword overlap. "
        "Do NOT invent facts. Do not include chunk ids.\n"
        "- review_rating (optional): either an integer 1-5 for equality filter, OR an object {\"min\": int, \"max\": int} with bounds in 1-5.\n"
        f"- query_family (optional): one of [{fams}]\n"
        "- needs_low_rating_evidence (optional boolean): true when the user is clearly asking for negative/low satisfaction/worst experiences.\n\n"
        "Rules:\n"
        "- If the user asks for worst/negative/low satisfaction but baseline filters are empty, set needs_low_rating_evidence=true "
        "unless you set review_rating explicitly.\n"
        "- Prefer review_rating constraints over long retrieval_query_text when the intent is clearly about low ratings.\n"
        "- If uncertain whether metadata exists, prefer conservative retrieval_query_text changes over inventing strict filters.\n\n"
        "JSON:"
    )


def apply_llm_query_plan(
    *,
    user_query: str,
    request: RetrievalRequest,
    llm: BaseLLM,
) -> QueryPlanResult:
    """
    Mutates ``request`` in-place when a valid plan is produced.
    """
    prompt = _build_planner_prompt(user_query, dict(request.filters or {}))
    try:
        raw = llm.generate(prompt)
    except Exception:
        return QueryPlanResult(
            applied=False,
            source="llm_failed",
            retrieval_query_text=None,
            filters_patch={},
            query_family=None,
            raw_model_text=None,
            notes="planner_llm_error",
        )

    obj = _extract_json_object(raw)
    if not obj:
        return QueryPlanResult(
            applied=False,
            source="llm_failed",
            retrieval_query_text=None,
            filters_patch={},
            query_family=None,
            raw_model_text=raw,
            notes="planner_invalid_json",
        )

    cleaned = _validate_plan_obj(obj)
    if not cleaned:
        return QueryPlanResult(
            applied=False,
            source="llm_failed",
            retrieval_query_text=None,
            filters_patch={},
            query_family=None,
            raw_model_text=raw,
            notes="planner_empty_after_validation",
        )

    changed = False
    if "retrieval_query_text" in cleaned and cleaned["retrieval_query_text"] != request.query_text:
        request.query_text = cleaned["retrieval_query_text"]
        changed = True

    if "review_rating" in cleaned:
        request.filters = dict(request.filters or {})
        request.filters["review_rating"] = cleaned["review_rating"]
        changed = True

    if "query_family" in cleaned:
        request.query_family = cleaned["query_family"]
        changed = True

    # If family was not explicitly set, re-infer after text/filter changes.
    if changed and not request.query_family:
        request.query_family = infer_query_family(request)

    return QueryPlanResult(
        applied=bool(changed),
        source="llm",
        retrieval_query_text=cleaned.get("retrieval_query_text"),
        filters_patch={
            k: v for k, v in cleaned.items() if k in ("review_rating", "query_family")
        },
        query_family=request.query_family,
        raw_model_text=raw,
        notes=None,
    )


def maybe_apply_query_planner(
    *,
    enabled: bool,
    user_query: str,
    request: RetrievalRequest,
    llm: BaseLLM,
    parser: QueryParser,
    skip_if_followup_filters: bool,
) -> QueryPlanResult:
    """
    High-level entry used by RAGPipeline.

    ``skip_if_followup_filters`` avoids fighting Phase 5.4 resolver merges when filters
    were explicitly supplied via overrides.
    """
    if not enabled:
        return QueryPlanResult(
            applied=False,
            source="none",
            retrieval_query_text=None,
            filters_patch={},
            query_family=None,
            raw_model_text=None,
            notes=None,
        )

    merged_has_rating = bool((request.filters or {}).get("review_rating"))
    if skip_if_followup_filters and merged_has_rating:
        return QueryPlanResult(
            applied=False,
            source="none",
            retrieval_query_text=None,
            filters_patch={},
            query_family=None,
            raw_model_text=None,
            notes="skipped_followup_filters_present",
        )

    # If rules already extracted rating filters for this utterance, don't spend an LLM call.
    probe = parser.parse(user_query, top_k=request.top_k)
    if probe.filters.get("review_rating"):
        return QueryPlanResult(
            applied=False,
            source="none",
            retrieval_query_text=None,
            filters_patch={},
            query_family=None,
            raw_model_text=None,
            notes="skipped_rules_already_have_rating_filter",
        )

    return apply_llm_query_plan(user_query=user_query, request=request, llm=llm)


__all__ = [
    "QueryPlanResult",
    "apply_llm_query_plan",
    "maybe_apply_query_planner",
]
