"""
Post-retrieval excerpt shaping (Phase 5.x).

Drop rows whose text is primarily absence / negation (e.g. “no rash”) so they
are not passed into the LLM context — unless the user query asks for reassurance.

This does not modify retriever/reranker internals; it filters the DataFrame
after retrieval, before prompt assembly.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

_REASSURANCE_PHRASES = (
    "is it safe",
    "safe to use",
    "safe for",
    "will i get",
    "could i get",
    "side effect",
    "side effects",
    "allergic",
    "allergy",
    "worry",
    "worried",
    "reassure",
    "should i be concerned",
    "any risk",
    "risks",
)

_HEALTH_QUERY_TERMS = (
    "health",
    "rash",
    "irritat",
    "allerg",
    "symptom",
    "reaction",
    "burn",
    "hurt",
    "pain",
    "swell",
    "itch",
    "medical",
    "dermat",
    "skin",
)

_NEGATION_START = re.compile(
    r"(?im)^\s*(no|not|never|none|without|didn['’]?t|don['’]?t|doesn['’]?t|haven['’]?t|hasn['’]?t|wasn['’]?t|weren['’]?t)\b"
)

_STRONG_NEGATION = re.compile(
    r"\b(no|not|never|none|without|didn['’]?t|don['’]?t|doesn['’]?t|haven['’]?t|hasn['’]?t|wasn['’]?t|weren['’]?t)\s+"
    r"(experience|get|have|see|notice|develop|suffer|feel)\b",
    re.I,
)

_ADVERSE_HINT = re.compile(
    r"\b(rash|hives|irritation|irritated|burning|burned|swollen|swelling|blister|bleed|blood|"
    r"infection|allergic|allergy|reaction|pain|ache|sore|raw|broken|skin peeled|peeling|scar|hospital|"
    r"doctor|urgent|severe|worst)\b",
    re.I,
)

_NEGATOR_WORD = re.compile(
    r"\b(no|not|never|none|without|didn['’]?t|don['’]?t|doesn['’]?t|haven['’]?t|hasn['’]?t|wasn['’]?t|weren['’]?t)\b",
    re.I,
)


def user_seeks_reassurance(query: str) -> bool:
    q = (query or "").lower()
    return any(p in q for p in _REASSURANCE_PHRASES)


def health_intent_query(query: str) -> bool:
    """Only apply absence filtering when the question is health-adjacent."""
    q = (query or "").lower()
    return any(t in q for t in _HEALTH_QUERY_TERMS)


def _row_text(row: pd.Series) -> str:
    t = row.get("text", "")
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return ""
    return str(t)


def is_primarily_absence_excerpt(text: str) -> bool:
    """
    Heuristic: many negations / absence framing with little explicit adverse detail.
    """
    s = (text or "").strip()
    if len(s) < 24:
        return False

    neg_hits = len(_STRONG_NEGATION.findall(s))
    if _NEGATION_START.search(s):
        neg_hits += 1
    low = s.lower()
    padded = f" {low} "
    for tok in (
        " no ",
        " not ",
        " never ",
        " none ",
        " without ",
        " didn't ",
        " didnt ",
        " don't ",
        " dont ",
        " doesn't ",
        " doesnt ",
        " haven't ",
        " havent ",
        " hasn't ",
        " hasnt ",
        " wasn't ",
        " wasnt ",
        " weren't ",
        " werent ",
    ):
        neg_hits += padded.count(tok)
    if low.startswith(
        ("no ", "not ", "never ", "none ", "without ", "didn't ", "dont ", "don't ")
    ):
        neg_hits += 1

    adverse_hits = _affirmative_adverse_hits(s)
    if adverse_hits >= 2:
        return False
    if adverse_hits == 1 and neg_hits <= 1:
        return False

    absence_phrases = (
        "no evidence",
        "did not experience",
        "didn't experience",
        "no issues",
        "no problems",
        "no side effects",
        "no reaction",
        "no irritation",
        "no rash",
        "never had",
        "never experienced",
        "not allergic",
        "without any",
    )
    phrase_hits = sum(1 for p in absence_phrases if p in low)

    if phrase_hits >= 1 and adverse_hits == 0 and neg_hits >= 1:
        return True
    if neg_hits >= 2 and adverse_hits == 0:
        return True
    if phrase_hits >= 2 and adverse_hits == 0:
        return True
    return False


def _affirmative_adverse_hits(text: str) -> int:
    """
    Count adverse-symptom mentions that are not immediately negated in the same clause-ish window.

    This avoids treating "no rash" as an affirmative adverse hit.
    """
    s = text or ""
    if not s.strip():
        return 0
    hits = 0
    for m in _ADVERSE_HINT.finditer(s):
        start = max(0, m.start() - 28)
        prefix = s[start : m.start()]
        if _NEGATOR_WORD.search(prefix):
            continue
        hits += 1
    return hits


def filter_absence_focused_excerpts(
    retrieved: pd.DataFrame,
    *,
    user_question: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Returns (filtered_df, stats). If nothing to do, returns input unchanged.
    """
    if retrieved is None or retrieved.empty or "text" not in retrieved.columns:
        return retrieved, {"skipped": True, "reason": "empty_or_no_text_column"}

    if user_seeks_reassurance(user_question):
        return retrieved, {"skipped": True, "reason": "reassurance_query"}

    if not health_intent_query(user_question):
        return retrieved, {"skipped": True, "reason": "not_health_adjacent_query"}

    drop_idx: list[Any] = []
    for idx, row in retrieved.iterrows():
        if is_primarily_absence_excerpt(_row_text(row)):
            drop_idx.append(idx)

    if not drop_idx:
        return retrieved, {"skipped": True, "reason": "no_rows_matched"}

    filtered = retrieved.drop(index=drop_idx, errors="ignore").reset_index(drop=True)
    stats: dict[str, Any] = {
        "skipped": False,
        "dropped_row_index_count": len(drop_idx),
        "rows_in": int(len(retrieved)),
        "rows_out": int(len(filtered)),
    }
    return filtered, stats


__all__ = [
    "user_seeks_reassurance",
    "health_intent_query",
    "is_primarily_absence_excerpt",
    "filter_absence_focused_excerpts",
]
