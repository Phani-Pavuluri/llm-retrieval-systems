"""
Phase 5.4 — Rule-based follow-up detection and query resolution.

No LLM, no DB, no retriever changes. Caller passes ConversationContext per request.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from src.conversation_state import ConversationContext, MAX_TURNS

FollowupType = Literal["none", "scope", "output", "explain", "aspect"]

_FOLLOWUP_CUES = (
    "what about",
    "only ",
    " only",
    "just ",
    "more briefly",
    "shorter",
    "shorter.",
    " why",
    "why?",
    "why ",
    "based on that",
    "which chunk",
    "which chunks",
    "support that",
    " that",
    " that?",
    " those",
    " it ",
    " it?",
)
_SCOPE_CUES = (
    "one-star",
    "1-star",
    "one star",
    "1 star",
    "only negative",
    "negative only",
    "low-rated",
    "low rated",
)
_OUTPUT_CUES = ("shorter", "more briefly", "brief", "bullet", "bullets", "top 3", "top three", "concise")
_EXPLAIN_CUES = ("why", "which chunk", "which chunks", "support that", "how confident")
_ASPECT_BUYER = ("buyer risk", "buyer risks", "red flag", "watch out")
_ASPECT_VALUE = ("value", "worth", "price", "overpriced")
_ASPECT_SYMPTOM = ("symptom", "rash", "reaction", "irritation")


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _jaccard(a: str, b: str) -> float:
    sa, sb = _tokens(a), _tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _has_followup_cues(q: str) -> bool:
    ql = q.lower()
    return any(c in ql for c in _FOLLOWUP_CUES)


def _is_probably_new_topic(current: str, last_resolved: str) -> bool:
    """Avoid sticky merge when the user pivots to a long unrelated question."""
    cur = current.strip()
    if len(cur) > 200 and not _has_followup_cues(cur):
        return True
    if len(cur.split()) > 22 and not _has_followup_cues(cur):
        return True
    if last_resolved and len(cur) > 50:
        if _jaccard(cur, last_resolved) < 0.06 and not _has_followup_cues(cur):
            return True
    # Medium-length new questions often share no tokens with the prior resolved query.
    if last_resolved and len(cur.split()) >= 6 and not _has_followup_cues(cur):
        if _jaccard(cur, last_resolved) < 0.12:
            return True
    return False


def detect_followup(current_query: str, state: ConversationContext | None) -> tuple[bool, FollowupType]:
    """
    Heuristic: short / cue-based follow-ups when prior turn exists.
    Returns (is_followup, coarse type for resolver branch ordering).
    """
    if not state or not state.turns:
        return False, "none"
    last = state.last_turn()
    if not last or not (last.resolved_query or "").strip():
        return False, "none"
    q = (current_query or "").strip()
    if not q:
        return False, "none"
    ql = q.lower()
    if ql.startswith("new question") or ql.startswith("start over") or ql.startswith("ignore that"):
        return False, "none"
    if _is_probably_new_topic(q, last.resolved_query):
        return False, "none"
    if len(q) > 140 and not _has_followup_cues(q):
        return False, "none"

    # Must look like a follow-up (cue or very short continuation)
    if len(q) > 100 and not _has_followup_cues(q):
        return False, "none"

    # Type classification — order: scope > explain > output > aspect (PRODUCT 5.4 precedence)
    if any(c in ql for c in _SCOPE_CUES):
        return True, "scope"
    if re.search(r"\bwhy\b", ql) or any(c in ql for c in ("which chunk", "which chunks", "support that")):
        return True, "explain"
    if any(c in ql for c in _OUTPUT_CUES):
        return True, "output"
    if any(c in ql for c in _ASPECT_BUYER) or (
        "what about" in ql and any(c in ql for c in _ASPECT_VALUE)
    ):
        return True, "aspect"
    if any(c in ql for c in _ASPECT_SYMPTOM) and ("what about" in ql or "any" in ql):
        return True, "aspect"
    if "what about" in ql or _has_followup_cues(q):
        return True, "aspect"
    # Very short continuations only (avoid treating unrelated 30–39 char questions as follow-ups).
    if len(q) <= 18:
        return True, "aspect"
    return False, "none"


@dataclass
class ResolutionResult:
    resolved_query: str
    filter_overrides: dict[str, Any] | None = None
    query_family_override: str | None = None
    explain_force: bool = False
    output_style_hints: dict[str, Any] | None = None
    reset_filters: bool = False
    is_followup: bool = False
    followup_type: FollowupType = "none"
    resolver_metadata: dict[str, Any] = field(default_factory=dict)


def resolve_followup(current_query: str, state: ConversationContext | None) -> ResolutionResult:
    """
    Merge current user message with last turn when detected as follow-up.
    Precedence: scope cues > explain > aspect > output (see PRODUCT_ROADMAP Phase 5.4).
    """
    q = (current_query or "").strip()
    is_fb, ftype = detect_followup(q, state)
    if not is_fb or not state or not state.turns:
        return ResolutionResult(
            resolved_query=q,
            is_followup=False,
            followup_type="none",
            resolver_metadata={"reused_fields": []},
        )

    last = state.last_turn()
    assert last is not None
    base = (last.resolved_query or last.user_query_raw or "").strip()
    meta: dict[str, Any] = {"reused_fields": ["resolved_query"], "prior_resolved_query": base}
    ql = q.lower()

    # Scope refinement
    if ftype == "scope":
        fo: dict[str, Any] = {}
        if any(c in ql for c in ("one-star", "1-star", "one star", "1 star")):
            fo["review_rating"] = 1
        elif any(c in ql for c in ("only negative", "negative only", "low-rated", "low rated")):
            fo["review_rating"] = {"max": 3}
        resolved = f"{base}\n\nFollow-up scope: {q.strip()}"
        meta["reused_fields"] = ["resolved_query", "filters"]
        meta["filter_merge"] = fo
        return ResolutionResult(
            resolved_query=resolved,
            filter_overrides=fo or None,
            is_followup=True,
            followup_type="scope",
            resolver_metadata=meta,
        )

    # Explanation follow-up
    if ftype == "explain":
        meta["reused_fields"] = ["resolved_query", "explain_forced"]
        return ResolutionResult(
            resolved_query=base,
            explain_force=True,
            is_followup=True,
            followup_type="explain",
            resolver_metadata=meta,
        )

    # Aspect shift — new angle on the thread; do not keep prior metadata filters (e.g. star scope).
    if ftype == "aspect":
        fam: str | None = None
        if any(c in ql for c in _ASPECT_BUYER):
            fam = "buyer_risk_issues"
        elif any(c in ql for c in _ASPECT_VALUE):
            fam = "value_complaint"
        elif any(c in ql for c in _ASPECT_SYMPTOM):
            fam = "symptom_issue_extraction"
        resolved = f"{base}\n\nFollow-up aspect: {q.strip()}"
        meta["reused_fields"] = ["resolved_query", "query_family", "filters_reset"]
        meta["query_family_override"] = fam
        meta["filters_reset"] = True
        return ResolutionResult(
            resolved_query=resolved,
            query_family_override=fam,
            reset_filters=True,
            is_followup=True,
            followup_type="aspect",
            resolver_metadata=meta,
        )

    # Output refinement
    hints: dict[str, Any] = {}
    if "bullet" in ql:
        hints["format"] = "bullets"
    if "shorter" in ql or "more briefly" in ql or "brief" in ql or "concise" in ql:
        hints["brevity"] = "short"
    if "top 3" in ql or "top three" in ql:
        hints["max_issues"] = 3
    meta["reused_fields"] = ["resolved_query", "output_style_hints"]
    meta["hints"] = hints
    return ResolutionResult(
        resolved_query=base,
        output_style_hints=hints if hints else None,
        is_followup=True,
        followup_type="output",
        resolver_metadata=meta,
    )


__all__ = [
    "detect_followup",
    "resolve_followup",
    "ResolutionResult",
    "FollowupType",
]
