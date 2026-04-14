"""
Phase 5.1 — Structured explainability from existing pipeline state (no LLM).

Builds ``evidence[]``, ``reasoning_summary{}``, and ``confidence{}`` for optional
``explain=True`` answers. The **authoritative answer string** lives on the
pipeline response at top level ``answer`` — not duplicated here (Phase 5.2 API
envelope).

**Evidence naming:** ``evidence`` is the list of chunks **provided to the model**
in the prompt context (same ordering as the LLM saw). It is **not** a claim
about which excerpts the model relied on most (no attribution / attention).
Template-based rationale only — not chain-of-thought.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src import config
from src.rerank_policy import infer_query_family
from src.retrieval_request import RetrievalRequest


def _effective_hybrid(request: RetrievalRequest) -> bool:
    if request.use_hybrid is not None:
        return bool(request.use_hybrid)
    return bool(config.HYBRID_RETRIEVAL)


def _evidence_rows(retrieved: pd.DataFrame) -> list[dict[str, Any]]:
    """One row per chunk in the final context window passed to the LLM."""
    out: list[dict[str, Any]] = []
    if retrieved is None or retrieved.empty:
        return out
    for pos, (_, row) in enumerate(retrieved.iterrows(), start=1):
        cid = str(row.get("chunk_id", "") or "")
        text = str(row.get("text", "") or "")
        src = row.get("asin")
        if src is None or (isinstance(src, float) and pd.isna(src)):
            src = row.get("product_id")
        ev: dict[str, Any] = {
            "chunk_id": cid,
            "source_id": None if src is None or pd.isna(src) else str(src),
            "chunk_text": text,
            "rank_position": pos,
        }
        for col in (
            "score",
            "retrieval_score",
            "semantic_score",
            "keyword_score",
            "rerank_score",
        ):
            if col in row.index and pd.notna(row.get(col)):
                try:
                    ev[col] = float(row[col])
                except (TypeError, ValueError):
                    ev[col] = row[col]
        out.append(ev)
    return out


def _rating_scope_text(filters: dict[str, Any]) -> str | None:
    if not filters:
        return None
    rr = filters.get("review_rating")
    if rr is None:
        return None
    if isinstance(rr, dict):
        lo, hi = rr.get("min"), rr.get("max")
        if lo is not None and hi is not None:
            return f"review_rating between {lo} and {hi} (inclusive)"
        if hi is not None:
            return f"review_rating ≤ {hi}"
        if lo is not None:
            return f"review_rating ≥ {lo}"
        return None
    return f"review_rating == {rr}"


def _reasoning_summary(
    request: RetrievalRequest,
    retrieved: pd.DataFrame,
    prompt_template_id: str,
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    diag = diagnostics or {}
    rd = diag.get("rerank_decision") or {}
    hybrid = _effective_hybrid(request) or (
        retrieved is not None
        and not retrieved.empty
        and "semantic_score" in retrieved.columns
    )
    retrieval_mode = "hybrid" if hybrid else "vector"
    rerank_applied = bool(rd.get("use_rerank_effective"))
    qfam = getattr(request, "query_family", None) or infer_query_family(request)
    rating_scope = _rating_scope_text(dict(request.filters or {}))

    parts: list[str] = []
    if rerank_applied:
        parts.append(
            "Cross-encoder reranking was applied because the selective policy enabled it for this query shape."
        )
    else:
        if rd.get("rerank_skipped_due_to_metadata_filters"):
            parts.append(
                "Reranking was skipped because a metadata rating filter narrowed the candidate set (policy)."
            )
        elif rd.get("rerank_skipped_due_to_query_family"):
            parts.append(
                "Reranking was skipped because this query was classified as exact-issue or extraction-style retrieval (family policy)."
            )
        elif rd.get("rerank_skipped_due_to_confidence"):
            parts.append(
                "Reranking was skipped due to the confidence guard (already-strong hybrid top score)."
            )
        elif not bool(rd.get("use_rerank_requested")):
            parts.append(
                "Reranking was not requested for this request (config or selective policy off / default)."
            )
        else:
            parts.append("Reranking was not applied to the returned rows.")

    if hybrid:
        parts.append("Hybrid fusion (semantic + keyword) was used when scoring candidates.")
    else:
        parts.append("Vector similarity ordering was used (no hybrid fusion on this run).")

    if request.filters:
        parts.append(
            f"Metadata filters were applied: {request.filters!r}. "
            f"Excerpts provided to the model are only those passing filters."
        )

    summary_line = " ".join(parts)

    return {
        "retrieval_mode": retrieval_mode,
        "rerank_applied": rerank_applied,
        "rerank_requested": bool(rd.get("use_rerank_requested")),
        "query_family": qfam,
        "strategy_reason": request.strategy_reason or "",
        "filters_applied": dict(request.filters or {}),
        "rating_scope": rating_scope,
        "prompt_template_id": prompt_template_id,
        "summary_line": summary_line,
    }


def _confidence(
    request: RetrievalRequest,
    retrieved: pd.DataFrame,
    answer: str,
    diagnostics: dict[str, Any] | None,
    chunk_ids_used: list[str],
) -> dict[str, Any]:
    """
    Heuristic only. Does **not** parse chunk text for factual agreement or
    contradiction (no cross-chunk conflict signal yet).
    """
    diag = diagnostics or {}
    reasons: list[str] = []
    n = 0 if retrieved is None else len(retrieved)
    n_used = len(chunk_ids_used or [])
    under = bool(diag.get("underfilled_after_filtering"))
    rd = diag.get("rerank_decision") or {}
    rerank_on = bool(rd.get("use_rerank_effective"))
    a_low = (answer or "").lower().strip()
    insuff = a_low.startswith("insufficient evidence")

    score = 0.55  # base
    if n == 0:
        reasons.append("No excerpts in the answer context.")
        return {
            "confidence_label": "low",
            "confidence_score": 0.15,
            "confidence_reasons": reasons,
        }
    if insuff:
        reasons.append("Answer explicitly states insufficient evidence in excerpts.")
        score -= 0.25
    if under:
        reasons.append("Post-filter candidate count was below requested top_k (under-filled).")
        score -= 0.15
    if n_used <= 1 and n > 1:
        reasons.append("Answer context has multiple chunks but synthesis may lean on fewer; check chunk cites.")
        score -= 0.05
    if n == 1:
        reasons.append("Only one excerpt in context.")
        score -= 0.1
    if n >= 3 and not under and not insuff:
        reasons.append("Several excerpts passed filters and were available to the model.")
        score += 0.12
    if rerank_on and n >= 2:
        reasons.append("Cross-encoder rerank re-ordered multiple candidates.")
        score += 0.1
    if not rerank_on:
        reasons.append(
            "Selective policy or config left rerank off for this query (see reasoning_summary.rerank_applied)."
        )

    score = max(0.0, min(1.0, score))
    if score >= 0.72:
        label = "high"
    elif score >= 0.45:
        label = "medium"
    else:
        label = "low"
    reasons.append(f"Heuristic score={score:.2f} (not a calibrated probability).")
    return {
        "confidence_label": label,
        "confidence_score": round(score, 3),
        "confidence_reasons": reasons,
    }


def build_explanation_payload(
    *,
    request: RetrievalRequest,
    retrieved: pd.DataFrame,
    answer: str,
    chunk_ids_used: list[str],
    prompt_template_id: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Structured explainability object for API / CLI (no LLM call).

    Canonical keys under ``explanation`` (align with Phase 5.2): ``evidence``,
    ``reasoning_summary``, ``confidence``. Callers attach the final answer at
    response top level ``answer``.
    """
    diag = diagnostics or {}
    return {
        "evidence": _evidence_rows(retrieved),
        "reasoning_summary": _reasoning_summary(request, retrieved, prompt_template_id, diag),
        "confidence": _confidence(
            request, retrieved, answer, diag, chunk_ids_used
        ),
    }
