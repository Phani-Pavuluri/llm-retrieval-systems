"""Optional rerank stage after Retriever (keeps reranking out of Retriever)."""
from __future__ import annotations

import pandas as pd

from src import config
from src.rerank_policy import (
    build_rerank_trace_decision,
    should_skip_rerank_for_confidence,
)
from src.reranker import (
    CrossEncoderReranker,
    apply_rerank_to_candidates,
    effective_rerank,
    effective_rerank_model,
    rerank_top_n_for_request,
)
from src.retrieval_request import RetrievalRequest
from src.retrieval_trace import build_retrieval_trace_record, emit_retrieval_trace_record
from src.retriever import Retriever


def retrieve_with_optional_rerank(
    retriever: Retriever,
    request: RetrievalRequest,
    trace_extra: dict | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> pd.DataFrame:
    """
    Retrieval + metadata filter, optional cross-encoder rerank, trace emit.
    Used by RAGPipeline and eval scripts.
    """
    requested = effective_rerank(request)
    df = retriever.retrieve(
        request,
        trace_extra=trace_extra,
        write_trace=not requested,
    )
    rtn = rerank_top_n_for_request(request)
    rm = effective_rerank_model(request)

    if not requested:
        return df

    ctx = retriever.last_trace_build_context or {}
    df_pre = ctx.get("df_pre_rerank", df)

    if should_skip_rerank_for_confidence(df_pre):
        df_final = df_pre.head(int(request.top_k)).reset_index(drop=True)
        rd = build_rerank_trace_decision(
            request,
            rerank_requested_before_confidence=True,
            rerank_applied=False,
            rerank_skipped_due_to_confidence=True,
            rerank_top_n_effective=rtn,
        )
        record = build_retrieval_trace_record(
            request,
            trace_extra,
            ctx.get("df_pre_filter", pd.DataFrame()),
            ctx.get("df_post_filter", pd.DataFrame()),
            df_pre,
            df_final,
            int(ctx.get("k_fetch_requested", 0)),
            use_rerank_effective=False,
            rerank_top_n_effective=rtn,
            rerank_applied=False,
            rerank_model=rm,
            rerank_decision=rd,
        )
        emit_retrieval_trace_record(record, trace_extra)
        retriever.last_retrieval_diagnostics = {
            **(retriever.last_retrieval_diagnostics or {}),
            "final_returned_count": len(df_final),
            "rerank_candidate_count": len(df_pre),
            "rerank_decision": rd,
        }
        return df_final

    if reranker is not None:
        rer = reranker
        model_for_trace = getattr(rer, "_model_name", rm)
    else:
        rer = CrossEncoderReranker(model_name=rm)
        model_for_trace = rm

    df_final = apply_rerank_to_candidates(
        rer,
        request.query_text,
        df_pre,
        final_top_k=int(request.top_k),
    )

    rd = build_rerank_trace_decision(
        request,
        rerank_requested_before_confidence=True,
        rerank_applied=True,
        rerank_skipped_due_to_confidence=False,
        rerank_top_n_effective=rtn,
    )
    record = build_retrieval_trace_record(
        request,
        trace_extra,
        ctx.get("df_pre_filter", pd.DataFrame()),
        ctx.get("df_post_filter", pd.DataFrame()),
        df_pre,
        df_final,
        int(ctx.get("k_fetch_requested", 0)),
        use_rerank_effective=True,
        rerank_top_n_effective=rtn,
        rerank_applied=True,
        rerank_model=model_for_trace,
        rerank_decision=rd,
    )
    emit_retrieval_trace_record(record, trace_extra)

    retriever.last_retrieval_diagnostics = {
        **(retriever.last_retrieval_diagnostics or {}),
        "final_returned_count": len(df_final),
        "rerank_candidate_count": len(df_pre),
        "rerank_decision": rd,
    }
    return df_final
