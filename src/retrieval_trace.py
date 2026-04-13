"""
Append one JSON object per line under artifacts/retrieval_traces/ (configurable).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.retrieval_request import RetrievalRequest


def _trace_path() -> Path:
    base = getattr(config, "RETRIEVAL_TRACE_DIR", config.PROJECT_ROOT / "artifacts" / "retrieval_traces")
    base.mkdir(parents=True, exist_ok=True)
    return base / "traces.jsonl"


def append_retrieval_trace(record: dict[str, Any]) -> None:
    if not getattr(config, "RETRIEVAL_TRACE_ENABLED", False):
        return
    row = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    path = _trace_path()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_trace_forced(record: dict[str, Any], path: Path | None = None) -> None:
    """Always write (for eval scripts), regardless of RETRIEVAL_TRACE_ENABLED."""
    row = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    out = path or _trace_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _chunk_id_list(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "chunk_id" not in df.columns:
        return []
    return [str(x) for x in df["chunk_id"].tolist()]


def _gold_set(trace_extra: dict[str, Any] | None) -> set[str]:
    gold_list = (trace_extra or {}).get("gold_chunk_ids") or []
    return {str(x) for x in gold_list}


def _gold_in_first_k(pre_ids: list[str], gold: set[str], k: int) -> bool | None:
    if not gold:
        return None
    kk = max(0, int(k))
    head = set(pre_ids[:kk])
    return bool(gold & head)


def build_retrieval_trace_record(
    request: RetrievalRequest,
    trace_extra: dict[str, Any] | None,
    df_pre_filter: pd.DataFrame,
    df_post_filter: pd.DataFrame,
    df_pre_rerank: pd.DataFrame,
    df_final: pd.DataFrame,
    k_fetch_requested: int,
    *,
    use_rerank_effective: bool,
    rerank_top_n_effective: int,
    rerank_applied: bool,
    rerank_model: str | None,
    rerank_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one JSON-serializable trace row (caller decides append vs forced path)."""
    use_hybrid = (
        request.use_hybrid if request.use_hybrid is not None else bool(config.HYBRID_RETRIEVAL)
    )
    alpha_e = (
        request.hybrid_alpha if request.hybrid_alpha is not None else config.HYBRID_ALPHA
    )
    beta_e = (
        request.hybrid_beta if request.hybrid_beta is not None else config.HYBRID_BETA
    )

    candidate_pool_size = len(df_pre_filter)
    post_filter_count = len(df_post_filter)
    top_k = int(request.top_k)
    underfilled = post_filter_count < top_k
    missing = max(0, top_k - post_filter_count) if underfilled else 0
    final_returned_count = len(df_final)

    pool_mult = float(request.candidate_pool_multiplier or 1.0)

    pre_rerank_ids = _chunk_id_list(df_pre_rerank)
    post_rerank_ids = _chunk_id_list(df_final)

    hits: list[dict[str, Any]] = []
    if not df_final.empty and "chunk_id" in df_final.columns:
        for _, r in df_final.iterrows():
            rs = r.get("retrieval_score", r.get("score", 0.0))
            h: dict[str, Any] = {
                "chunk_id": str(r["chunk_id"]),
                "fused_score": float(r.get("score", 0.0)),
                "retrieval_score": float(rs) if rs is not None else float(r.get("score", 0.0)),
            }
            if "semantic_score" in df_final.columns:
                h["semantic_score"] = float(r["semantic_score"])
            if "keyword_score" in df_final.columns:
                h["keyword_score"] = float(r["keyword_score"])
            if "rerank_score" in df_final.columns and pd.notna(r.get("rerank_score")):
                h["rerank_score"] = float(r["rerank_score"])
            if "final_rank" in df_final.columns and pd.notna(r.get("final_rank")):
                h["final_rank"] = int(r["final_rank"])
            hits.append(h)

    record: dict[str, Any] = {
        "query": request.original_query or request.query_text,
        "query_text": request.query_text,
        "task_type": request.task_type,
        "filters": dict(request.filters),
        "requested_top_k": top_k,
        "strategy_reason": request.strategy_reason,
        "use_hybrid": request.use_hybrid,
        "use_hybrid_effective": use_hybrid,
        "hybrid_alpha_effective": alpha_e if use_hybrid else None,
        "hybrid_beta_effective": beta_e if use_hybrid else None,
        "candidate_pool_multiplier": pool_mult,
        "retrieval_oversample_multiplier": config.RETRIEVAL_OVERSAMPLE_MULTIPLIER,
        "retrieval_min_candidates": config.RETRIEVAL_MIN_CANDIDATES,
        "k_fetch_requested": k_fetch_requested,
        "candidate_pool_size": candidate_pool_size,
        "post_filter_count": post_filter_count,
        "underfilled_after_filtering": underfilled,
        "underfilled_missing_count": missing,
        "final_returned_count": final_returned_count,
        "final_top_chunk_ids": post_rerank_ids,
        "hits": hits,
        "use_rerank": bool(use_rerank_effective),
        "rerank_model": rerank_model,
        "rerank_candidate_count": len(df_pre_rerank) if rerank_applied else None,
        "rerank_applied": bool(rerank_applied),
        "rerank_top_n": int(rerank_top_n_effective) if use_rerank_effective else None,
        "pre_rerank_top_chunk_ids": pre_rerank_ids,
        "post_rerank_top_chunk_ids": post_rerank_ids,
    }

    if rerank_decision:
        record.update(rerank_decision)
        if "use_rerank_effective" in rerank_decision:
            record["use_rerank"] = bool(rerank_decision["use_rerank_effective"])

    gold = _gold_set(trace_extra)
    if gold:
        pool_ids = set(_chunk_id_list(df_pre_filter))
        post_ids = set(_chunk_id_list(df_post_filter))
        top_ids = set(post_rerank_ids)
        record["gold_in_candidate_pool"] = bool(gold & pool_ids)
        record["gold_in_post_filter_pool"] = bool(gold & post_ids)
        record["gold_in_top_k_results"] = bool(gold & top_ids)
        record["gold_in_pre_rerank_top_k"] = _gold_in_first_k(pre_rerank_ids, gold, top_k)
        record["gold_in_post_rerank_top_k"] = _gold_in_first_k(post_rerank_ids, gold, top_k)

    if trace_extra:
        qfam = trace_extra.get("query_family")
        if qfam is not None:
            record["query_family"] = qfam
        extra = {k: v for k, v in trace_extra.items() if k not in ("trace_out",)}
        record.update({k: v for k, v in extra.items() if k not in record})

    return record


def emit_retrieval_trace_record(record: dict[str, Any], trace_extra: dict[str, Any] | None) -> None:
    """Append or force-write based on trace_out in trace_extra."""
    out_path = (trace_extra or {}).get("trace_out")
    if out_path is not None:
        write_trace_forced(record, Path(out_path))
    elif config.RETRIEVAL_TRACE_ENABLED:
        append_retrieval_trace(record)
