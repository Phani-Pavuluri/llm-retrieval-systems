"""
Phase 5.2 — Thin HTTP API over RAGPipeline (no auth, persistence, or streaming).

Run locally: ``uvicorn src.api:app --host 127.0.0.1 --port 8000`` from project root
with ``PYTHONPATH=.`` (or ``python scripts/run_api.py``).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src import config
from src.conversation_state import parse_conversation_context
from src.followup_resolver import ResolutionResult, resolve_followup
from src.rag_pipeline import RAGPipeline
from src.rerank_policy import infer_query_family

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    explain: bool = False
    llm_backend: str | None = None
    llm_model: str | None = None
    k: int | None = Field(default=None, ge=1)
    selective_rerank: bool | None = None
    rerank_model: str | None = None
    rerank_top_n: int | None = Field(default=None, ge=1)
    conversation_context: dict[str, Any] | None = None
    query_planner: bool | None = None

    @field_validator("llm_backend", "llm_model", "rerank_model", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v.strip() if isinstance(v, str) else v


class QuerySuccessResponse(BaseModel):
    answer: str
    explanation: dict[str, Any] | None
    metadata: dict[str, Any]


def get_pipeline(request: Request) -> RAGPipeline:
    """Single shared pipeline per process; tests may assign ``app.state.pipeline`` before requests."""
    p = getattr(request.app.state, "pipeline", None)
    if p is not None:
        return p
    request.app.state.pipeline = RAGPipeline()
    return request.app.state.pipeline


def _selective_rerank_effective(body: QueryRequest) -> bool:
    return (
        bool(config.RERANK_SELECTIVE)
        if body.selective_rerank is None
        else bool(body.selective_rerank)
    )


def _map_to_api_response(
    raw: dict[str, Any],
    body: QueryRequest,
    *,
    resolution: ResolutionResult | None = None,
) -> QuerySuccessResponse:
    req = raw["request"]
    qf = req.query_family or infer_query_family(req)
    res = resolution or ResolutionResult(resolved_query=body.query.strip(), is_followup=False)
    metadata: dict[str, Any] = {
        "query_family": qf,
        "prompt_template_id": raw["prompt_template_id"],
        "llm_backend": raw["llm_backend"],
        "llm_model": raw["llm_model"],
        "selective_rerank_effective": _selective_rerank_effective(body),
        "user_query": body.query.strip(),
        "resolved_query": res.resolved_query,
        "retrieval_query_text": getattr(req, "query_text", None),
        "is_followup": bool(res.is_followup),
        "followup_type": res.followup_type,
        "reused_fields": list(res.resolver_metadata.get("reused_fields") or []),
        "filters_applied": dict(req.filters or {}),
        "chunk_ids_used": list(raw.get("chunk_ids_used") or []),
        "query_plan": raw.get("query_plan"),
    }
    atp = raw.get("answer_trace_path")
    if atp:
        metadata["answer_trace_path"] = atp

    explain_on = bool(body.explain) or bool(res.explain_force)
    metadata["explain_used"] = explain_on

    explanation: dict[str, Any] | None
    if explain_on:
        explanation = raw.get("explanation")
        if not isinstance(explanation, dict):
            explanation = {}
    else:
        explanation = None

    return QuerySuccessResponse(
        answer=raw["answer"],
        explanation=explanation,
        metadata=metadata,
    )


def _error_payload(code: str, message: str, details: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        err["details"] = details
    return {"error": err}


def _run_query(pipeline: RAGPipeline, body: QueryRequest) -> QuerySuccessResponse:
    ctx = parse_conversation_context(body.conversation_context)
    res = resolve_followup(body.query.strip(), ctx)
    effective_query = res.resolved_query
    effective_explain = bool(body.explain) or bool(res.explain_force)

    k = body.k if body.k is not None else config.TOP_K
    raw = pipeline.answer(
        effective_query,
        k=k,
        explain=effective_explain,
        selective_rerank=body.selective_rerank,
        rerank_top_n=body.rerank_top_n,
        rerank_model=body.rerank_model,
        llm_backend=body.llm_backend,
        llm_model=body.llm_model,
        filter_overrides=res.filter_overrides,
        query_family_override=res.query_family_override,
        output_style_hints=res.output_style_hints,
        reset_filters=bool(res.reset_filters),
        query_planner=body.query_planner,
    )

    if effective_explain:
        ex = raw.get("explanation")
        if isinstance(ex, dict):
            prior: dict[str, Any] | None = None
            if ctx and ctx.last_turn():
                lt = ctx.last_turn()
                prior = {
                    "user_query_raw": lt.user_query_raw,
                    "resolved_query": lt.resolved_query,
                    "query_family": lt.query_family,
                    "filters": dict(lt.filters or {}),
                    "answer_summary": (lt.answer_summary or "")[:800],
                    "chunk_ids": list(lt.chunk_ids or [])[:50],
                    "explain_used": bool(lt.explain_used),
                }
            ex = {
                **ex,
                "conversation_transparency": {
                    "original_query": body.query.strip(),
                    "resolved_query": effective_query,
                    "reused_fields": list(res.resolver_metadata.get("reused_fields") or []),
                    "prior_turn": prior,
                },
            }
            raw["explanation"] = ex

    return _map_to_api_response(raw, body, resolution=res)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Reserve ``app.state.pipeline``; actual ``RAGPipeline`` is lazy-created in ``get_pipeline``."""
    app.state.pipeline = None
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Retrieval API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                "invalid_request",
                "Request validation failed.",
                details=exc.errors(),
            ),
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QuerySuccessResponse)
    def post_query(body: QueryRequest, request: Request):
        # Resolve pipeline only after body validation (avoid loading Retriever on 400s).
        try:
            pipeline = get_pipeline(request)
            return _run_query(pipeline, body)
        except ValueError as e:
            msg = str(e)
            if "Unsupported LLM backend" in msg:
                return JSONResponse(
                    status_code=400,
                    content=_error_payload("invalid_request", msg),
                )
            if "OPENAI_API_KEY" in msg:
                return JSONResponse(
                    status_code=503,
                    content=_error_payload(
                        "backend_unavailable",
                        "OpenAI backend is not configured (missing API key).",
                    ),
                )
            return JSONResponse(
                status_code=400,
                content=_error_payload("invalid_request", msg),
            )
        except ConnectionError as e:
            return JSONResponse(
                status_code=503,
                content=_error_payload("backend_unavailable", str(e)),
            )
        except RuntimeError as e:
            return JSONResponse(
                status_code=503,
                content=_error_payload("backend_unavailable", str(e)),
            )
        except (OSError, FileNotFoundError):
            logger.exception("pipeline_io")
            return JSONResponse(
                status_code=500,
                content=_error_payload(
                    "pipeline_error",
                    "Retrieval subsystem failed to load resources.",
                ),
            )
        except Exception:
            logger.exception("pipeline_error")
            return JSONResponse(
                status_code=500,
                content=_error_payload(
                    "internal_error",
                    "An unexpected error occurred.",
                ),
            )

    return app


app = create_app()
