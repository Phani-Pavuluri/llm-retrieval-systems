from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.answer_trace import append_answer_trace
from src.llm import get_llm
from src.prompt_builder import build_answer_prompt, describe_prompt_routing
from src.query_parser import QueryParser
from src.rerank_policy import apply_selective_rerank_policy
from src.reranker import CrossEncoderReranker, effective_rerank, effective_rerank_model
from src.retrieval_request import RetrievalRequest
from src.retrieval_strategy import apply_strategy_to_request
from src.explanation_builder import build_explanation_payload
from src.retrieval_with_rerank import retrieve_with_optional_rerank
from src.retriever import Retriever


class RAGPipeline:
    def __init__(
        self,
        llm_backend: str | None = None,
        llm_model: str | None = None,
        query_parser: QueryParser | None = None,
    ) -> None:
        self.retriever = Retriever()
        self.query_parser = query_parser if query_parser is not None else QueryParser()
        self._llm_backend = llm_backend if llm_backend is not None else config.LLM_BACKEND
        self._llm_model_name = llm_model
        self.llm = get_llm(self._llm_backend, model_name=llm_model)
        self._rerankers: dict[str, CrossEncoderReranker] = {}

    def _get_reranker(self, request: RetrievalRequest) -> CrossEncoderReranker:
        name = effective_rerank_model(request)
        if name not in self._rerankers:
            self._rerankers[name] = CrossEncoderReranker(model_name=name)
        return self._rerankers[name]

    def _retrieve_chunks(
        self,
        request: RetrievalRequest,
        trace_extra: dict | None = None,
    ) -> pd.DataFrame:
        return retrieve_with_optional_rerank(
            self.retriever,
            request,
            trace_extra=trace_extra,
            reranker=self._get_reranker(request) if effective_rerank(request) else None,
        )

    def answer(
        self,
        query: str,
        k: int = 5,
        use_parser: bool = True,
        use_hybrid: bool | None = None,
        use_retrieval_strategy: bool = True,
        use_rerank: bool | None = None,
        rerank_top_n: int | None = None,
        rerank_model: str | None = None,
        selective_rerank: bool | None = None,
        trace_extra: dict | None = None,
        *,
        explain: bool = False,
        llm_backend: str | None = None,
        llm_model: str | None = None,
        filter_overrides: dict[str, Any] | None = None,
        query_family_override: str | None = None,
        output_style_hints: dict[str, Any] | None = None,
        reset_filters: bool = False,
    ) -> dict:
        if use_parser:
            request = self.query_parser.parse(query, top_k=k)
        else:
            request = RetrievalRequest.from_raw(query, top_k=k)

        if reset_filters:
            request.filters = dict(filter_overrides or {})
        elif filter_overrides:
            merged = dict(request.filters or {})
            merged.update(filter_overrides)
            request.filters = merged
        if query_family_override and str(query_family_override).strip():
            request.query_family = str(query_family_override).strip()

        if use_hybrid is not None:
            request.use_hybrid = use_hybrid
            request.hybrid_alpha = None
            request.hybrid_beta = None
            request.strategy_reason = "cli_override"
            request.candidate_pool_multiplier = None
        elif use_retrieval_strategy:
            apply_strategy_to_request(request)

        sel = config.RERANK_SELECTIVE if selective_rerank is None else bool(selective_rerank)
        apply_selective_rerank_policy(request, selective_enabled=sel)

        if use_rerank is not None:
            request.use_rerank = use_rerank
            request.rerank_reason = "cli_override"
        if rerank_top_n is not None:
            request.rerank_top_n = rerank_top_n
        if rerank_model is not None:
            request.rerank_model = rerank_model

        retrieved = self._retrieve_chunks(request, trace_extra=trace_extra)

        original = request.original_query or query
        built = build_answer_prompt(
            request,
            original,
            retrieved,
            output_style_hints=output_style_hints,
        )

        if llm_backend is not None or llm_model is not None:
            eb = (
                llm_backend.strip().lower()
                if llm_backend is not None
                else str(self._llm_backend).strip().lower()
            )
            gen_llm = get_llm(eb, llm_model)
            eff_backend = eb
        else:
            gen_llm = self.llm
            eff_backend = str(self._llm_backend).strip().lower()

        answer = gen_llm.generate(built.prompt)

        llm_model = getattr(gen_llm, "model_name", None) or self._llm_model_name
        routing = describe_prompt_routing(request)
        trace_row: dict[str, Any] = {
            "query": original,
            "query_text": request.query_text,
            "prompt_template_id": built.template_id,
            "prompt_template_label": built.template_label,
            "prompt_routing": routing,
            "chunk_ids_used": built.chunk_ids,
            "llm_backend": eff_backend,
            "llm_model": llm_model,
            "answer": answer,
            "task_type": request.task_type,
            "strategy_reason": request.strategy_reason,
        }
        if trace_extra:
            _skip = frozenset({"trace_out", "answer_trace_out"})
            te = {k: v for k, v in trace_extra.items() if k not in _skip}
            trace_row["trace_extra"] = te

        answer_trace_path: Path | None = None
        explicit_out = (trace_extra or {}).get("answer_trace_out")
        if explicit_out is not None:
            answer_trace_path = append_answer_trace(trace_row, Path(explicit_out))
        elif getattr(config, "ANSWER_TRACE_ENABLED", False):
            answer_trace_path = append_answer_trace(trace_row)

        out: dict[str, Any] = {
            "query": original,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "request": request,
            "prompt_template_id": built.template_id,
            "prompt_template_label": built.template_label,
            "chunk_ids_used": built.chunk_ids,
            "llm_backend": eff_backend,
            "llm_model": llm_model,
        }
        if answer_trace_path is not None:
            out["answer_trace_path"] = str(answer_trace_path)
        if explain:
            diag = dict(self.retriever.last_retrieval_diagnostics or {})
            out["explanation"] = build_explanation_payload(
                request=request,
                retrieved=retrieved,
                answer=answer,
                chunk_ids_used=built.chunk_ids,
                prompt_template_id=built.template_id,
                diagnostics=diag,
            )
        return out
