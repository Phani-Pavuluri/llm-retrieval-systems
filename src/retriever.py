from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.embeddings import build_embedder_from_config
from src.hybrid_scoring import apply_hybrid_scoring
from src.metadata_filters import apply_metadata_filters
from src.rerank_policy import build_rerank_trace_decision
from src.reranker import effective_rerank, effective_rerank_model, rerank_top_n_for_request
from src.retrieval_request import RetrievalRequest
from src.retrieval_trace import build_retrieval_trace_record, emit_retrieval_trace_record
from src.vector_store import FaissVectorStore, load_metadata


class Retriever:
    def __init__(self) -> None:
        self.embedder = build_embedder_from_config()
        self.store = FaissVectorStore.load(config.VECTOR_STORE_DIR / "faiss.index")
        self.metadata = load_metadata(config.VECTOR_STORE_DIR / "chunk_metadata.csv")
        self.last_retrieval_diagnostics: dict[str, Any] | None = None
        self.last_trace_build_context: dict[str, Any] | None = None

    def retrieve(
        self,
        request: RetrievalRequest,
        trace_extra: dict[str, Any] | None = None,
        write_trace: bool | None = None,
    ) -> pd.DataFrame:
        do_rerank = effective_rerank(request)
        if write_trace is None:
            write_trace = not do_rerank

        k_fetch = self._candidate_count(request)
        ntotal = int(self.store.index.ntotal)
        if ntotal == 0:
            result = pd.DataFrame()
            self._finish_retrieval(
                request,
                trace_extra,
                df_pre_filter=result,
                df_post_filter=result,
                df_pre_rerank=result,
                result=result,
                k_fetch_requested=0,
                write_trace=write_trace,
            )
            return result
        k_fetch = min(k_fetch, ntotal)

        query_embedding = self.embedder.embed_query(request.query_text)
        scores, indices = self.store.search(query_embedding, k=k_fetch)

        rows = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.metadata.iloc[int(idx)].to_dict()
            row["score"] = float(score)
            rows.append(row)

        df_pre = pd.DataFrame(rows)
        if df_pre.empty:
            self._finish_retrieval(
                request,
                trace_extra,
                df_pre_filter=df_pre,
                df_post_filter=df_pre,
                df_pre_rerank=df_pre,
                result=df_pre,
                k_fetch_requested=k_fetch,
                write_trace=write_trace,
            )
            return df_pre

        if self._use_hybrid(request):
            alpha = (
                request.hybrid_alpha
                if request.hybrid_alpha is not None
                else config.HYBRID_ALPHA
            )
            beta = (
                request.hybrid_beta
                if request.hybrid_beta is not None
                else config.HYBRID_BETA
            )
            df_pre = apply_hybrid_scoring(
                df_pre,
                request.query_text,
                alpha=alpha,
                beta=beta,
            )

        df_post = apply_metadata_filters(df_pre, request.filters)
        df_post_sorted = df_post.sort_values("score", ascending=False)

        take_n = request.top_k
        if do_rerank:
            take_n = min(len(df_post_sorted), max(request.top_k, rerank_top_n_for_request(request)))
        result = df_post_sorted.head(take_n).reset_index(drop=True)

        if do_rerank and not result.empty:
            result = result.copy()
            result["retrieval_score"] = result["score"]

        self._finish_retrieval(
            request,
            trace_extra,
            df_pre_filter=df_pre,
            df_post_filter=df_post,
            df_pre_rerank=result,
            result=result,
            k_fetch_requested=k_fetch,
            write_trace=write_trace,
        )
        return result

    def _finish_retrieval(
        self,
        request: RetrievalRequest,
        trace_extra: dict[str, Any] | None,
        df_pre_filter: pd.DataFrame,
        df_post_filter: pd.DataFrame,
        df_pre_rerank: pd.DataFrame,
        result: pd.DataFrame,
        k_fetch_requested: int,
        write_trace: bool,
    ) -> None:
        do_rerank = effective_rerank(request)
        rtn = rerank_top_n_for_request(request)

        candidate_pool_size = len(df_pre_filter)
        post_filter_count = len(df_post_filter)
        top_k = int(request.top_k)
        underfilled = post_filter_count < top_k
        final_returned_count = len(result)

        self.last_trace_build_context = {
            "df_pre_filter": df_pre_filter,
            "df_post_filter": df_post_filter,
            "df_pre_rerank": df_pre_rerank,
            "k_fetch_requested": k_fetch_requested,
        }

        rd = build_rerank_trace_decision(
            request,
            rerank_requested_before_confidence=bool(do_rerank),
            rerank_applied=False,
            rerank_skipped_due_to_confidence=False,
            rerank_top_n_effective=rtn,
        )
        self.last_retrieval_diagnostics = {
            "candidate_pool_size": candidate_pool_size,
            "post_filter_count": post_filter_count,
            "final_returned_count": final_returned_count,
            "underfilled_after_filtering": underfilled,
            "requested_top_k": top_k,
            "rerank_candidate_count": len(df_pre_rerank) if do_rerank else None,
            "rerank_decision": rd,
        }

        if not write_trace:
            return

        record = build_retrieval_trace_record(
            request,
            trace_extra,
            df_pre_filter,
            df_post_filter,
            df_pre_rerank,
            result,
            k_fetch_requested,
            use_rerank_effective=do_rerank,
            rerank_top_n_effective=rtn,
            rerank_applied=False,
            rerank_model=effective_rerank_model(request),
            rerank_decision=rd,
        )
        emit_retrieval_trace_record(record, trace_extra)

    def _use_hybrid(self, request: RetrievalRequest) -> bool:
        if request.use_hybrid is not None:
            return request.use_hybrid
        return bool(config.HYBRID_RETRIEVAL)

    def _pool_target_k(self, request: RetrievalRequest) -> int:
        if effective_rerank(request):
            return max(int(request.top_k), int(rerank_top_n_for_request(request)))
        return int(request.top_k)

    def _candidate_count(self, request: RetrievalRequest) -> int:
        m = float(request.candidate_pool_multiplier or 1.0)
        eff_k = self._pool_target_k(request)
        a = math.ceil(eff_k * config.RETRIEVAL_OVERSAMPLE_MULTIPLIER * m)
        b = math.ceil(config.RETRIEVAL_MIN_CANDIDATES * m)
        return max(a, b)
