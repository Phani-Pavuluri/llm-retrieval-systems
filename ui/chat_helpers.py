"""
Pure helpers for the Streamlit chat UI (Phase 5.3).

Builds POST /query JSON and formats API payloads for display — testable without Streamlit.
"""
from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_API_BASE = os.environ.get("CHAT_API_BASE", "http://127.0.0.1:8000").rstrip("/")


def build_query_json(
    *,
    query: str,
    explain: bool,
    llm_backend: str | None = None,
    llm_model: str | None = None,
    k: int | None = None,
    selective_rerank: bool | None = None,
    rerank_model: str | None = None,
    rerank_top_n: int | None = None,
    conversation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Body for POST /query; omits optional keys when None so the API uses server defaults."""
    body: dict[str, Any] = {"query": query.strip(), "explain": bool(explain)}
    if llm_backend:
        body["llm_backend"] = llm_backend.strip().lower()
    if llm_model:
        body["llm_model"] = llm_model.strip()
    if k is not None:
        body["k"] = int(k)
    if selective_rerank is not None:
        body["selective_rerank"] = bool(selective_rerank)
    if rerank_model:
        body["rerank_model"] = rerank_model.strip()
    if rerank_top_n is not None:
        body["rerank_top_n"] = int(rerank_top_n)
    if conversation_context is not None:
        body["conversation_context"] = conversation_context
    return body


def conversation_context_from_turn_history(
    turns: list[dict[str, Any]],
    *,
    max_turns: int = 5,
) -> dict[str, Any] | None:
    """
    Build Phase 5.4 ``conversation_context`` from prior successful UI turns (client-side only).

    ``turns`` entries are ``{ "query": str, "ok": bool, "data": API JSON }`` as stored by the chat UI.
    """
    if not turns:
        return None
    records: list[dict[str, Any]] = []
    for t in turns[-max_turns:]:
        if not t.get("ok"):
            continue
        data = t.get("data") or {}
        meta = data.get("metadata") or {}
        ans = data.get("answer") or ""
        records.append(
            {
                "user_query_raw": (t.get("query") or "").strip(),
                "resolved_query": (meta.get("resolved_query") or t.get("query") or "").strip(),
                "query_family": meta.get("query_family"),
                "filters": dict(meta.get("filters_applied") or {}),
                "answer_summary": (ans[:400] + ("…" if len(ans) > 400 else "")),
                "chunk_ids": list(meta.get("chunk_ids_used") or [])[:80],
                "explain_used": bool(meta.get("explain_used"))
                or bool((data.get("explanation") or {})),
            }
        )
    if not records:
        return None
    return {"turns": records}


def check_api_health(base_url: str, timeout: float = 3.0) -> tuple[bool, str | None]:
    """
    GET /health on base_url.
    Returns (True, None) if OK, else (False, human-readable reason).
    """
    base = base_url.rstrip("/")
    url = f"{base}/health"
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        return False, f"Could not reach API at {base!r}: {e}"
    if r.status_code != 200:
        return False, f"API health check failed (HTTP {r.status_code}) at {url!r}"
    try:
        data = r.json()
    except ValueError:
        return False, "API health response was not valid JSON."
    if data.get("status") != "ok":
        return False, f"Unexpected health payload: {data!r}"
    return True, None


def evidence_score_parts(ev: dict[str, Any]) -> list[str]:
    """Human-readable score fragments for one evidence row (no fabricated keys)."""
    parts: list[str] = []
    for key, label in (
        ("rerank_score", "rerank_score"),
        ("retrieval_score", "retrieval_score"),
        ("score", "score"),
        ("semantic_score", "semantic_score"),
        ("keyword_score", "keyword_score"),
    ):
        v = ev.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
            parts.append(f"{label}={fv:.4f}")
        except (TypeError, ValueError):
            parts.append(f"{label}={v!r}")
    return parts


def reasoning_summary_lines(rs: dict[str, Any]) -> list[str]:
    """Ordered markdown-friendly lines for reasoning_summary display."""
    lines: list[str] = []
    order = [
        ("retrieval_mode", "retrieval_mode"),
        ("rerank_applied", "rerank_applied"),
        ("query_family", "query_family"),
        ("strategy_reason", "strategy_reason"),
        ("filters_applied", "filters_applied"),
        ("rating_scope", "rating_scope"),
        ("prompt_template_id", "prompt_template_id"),
    ]
    for key, label in order:
        if key not in rs:
            continue
        val = rs[key]
        if val is None or val == "":
            continue
        if key == "filters_applied" and isinstance(val, dict) and not val:
            continue
        lines.append(f"**{label}:** {val}")
    sl = rs.get("summary_line")
    if sl:
        lines.append(f"**summary_line:** {sl}")
    return lines


def confidence_markdown_lines(conf: dict[str, Any]) -> list[str]:
    """Lines for confidence block (label, score, reason bullets)."""
    lines: list[str] = []
    if conf.get("confidence_label") is not None:
        lines.append(f"**Label:** `{conf['confidence_label']}`")
    if conf.get("confidence_score") is not None:
        lines.append(f"**Score (heuristic):** `{conf['confidence_score']}`")
    reasons = conf.get("confidence_reasons") or []
    if reasons:
        lines.append("**Reasons:**")
        for r in reasons:
            lines.append(f"- {r}")
    return lines


_METADATA_ORDER = (
    "llm_backend",
    "llm_model",
    "prompt_template_id",
    "query_family",
    "user_query",
    "resolved_query",
    "is_followup",
    "followup_type",
    "reused_fields",
    "filters_applied",
    "chunk_ids_used",
    "selective_rerank_effective",
    "answer_trace_path",
)


def metadata_markdown_lines(meta: dict[str, Any]) -> list[str]:
    """Lightweight metadata lines in a stable order; skips keys not present."""
    lines: list[str] = []
    for k in _METADATA_ORDER:
        if k not in meta:
            continue
        v = meta[k]
        if v is None or v == "":
            continue
        if k == "reused_fields" and isinstance(v, list) and not v:
            continue
        if k == "filters_applied" and isinstance(v, dict) and not v:
            continue
        if k == "chunk_ids_used" and isinstance(v, list) and not v:
            continue
        if k in ("reused_fields", "filters_applied", "chunk_ids_used"):
            lines.append(f"- **{k}:** `{v!r}`")
        else:
            lines.append(f"- **{k}:** `{v}`")
    return lines


def format_api_error(payload: dict[str, Any]) -> tuple[str, str, Any | None]:
    """
    Parse error JSON body; returns (code, message, details).
    Falls back if shape is unexpected.
    """
    err = payload.get("error")
    if isinstance(err, dict):
        return (
            str(err.get("code", "error")),
            str(err.get("message", "")),
            err.get("details"),
        )
    return ("error", str(payload), None)
