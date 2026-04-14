#!/usr/bin/env python3
"""
Phase 4.1: Run all queries in eval/answer_eval_labeled.json through RAG + answer tracing.

Uses normal selective rerank policy (do not pass use_rerank=True — that would override selective).
Rerank model: config RERANK_MODEL unless --rerank-model is set (your chosen best model).

Writes:
  - JSONL lines to --out-jsonl (default: artifacts/answer_traces/answer_eval_runs.jsonl)
  - A combined JSON array to --out-json for easy diff / labeling prep

Usage:
  cd <repo> && PYTHONPATH=. python scripts/run_answer_eval.py
  PYTHONPATH=. python scripts/run_answer_eval.py --rerank-model cross-encoder/ms-marco-MiniLM-L-12-v2
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _repo_path(path: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()


def main() -> None:
    from src import config
    from src.rag_pipeline import RAGPipeline

    ap = argparse.ArgumentParser(
        description="Run answer-eval JSON queries with answer tracing (selective rerank unchanged)."
    )
    ap.add_argument(
        "--eval-file",
        type=Path,
        default=_ROOT / "eval" / "answer_eval_labeled.json",
    )
    ap.add_argument(
        "--out-jsonl",
        type=Path,
        default=_ROOT / "artifacts" / "answer_traces" / "answer_eval_runs.jsonl",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=_ROOT / "artifacts" / "answer_traces" / "answer_eval_runs.json",
    )
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=("openai", "ollama"),
        help="LLM backend (default: config LLM_BACKEND)",
    )
    ap.add_argument("--model", type=str, default=None, help="LLM model id (backend default if omitted)")
    ap.add_argument(
        "--rerank-model",
        type=str,
        default=None,
        help="Cross-encoder for rerank modes (default: config RERANK_MODEL)",
    )
    ap.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help="Optional rerank candidate pool override (rerank modes only)",
    )
    ap.add_argument(
        "--no-selective-rerank",
        action="store_true",
        help="Set selective_rerank=False (otherwise uses config RERANK_SELECTIVE)",
    )
    args = ap.parse_args()
    args.eval_file = _repo_path(args.eval_file)
    args.out_jsonl = _repo_path(args.out_jsonl)
    args.out_json = _repo_path(args.out_json)

    raw = json.loads(args.eval_file.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = list(raw.get("items") or [])
    if not items:
        print("No items in eval file.", file=sys.stderr)
        raise SystemExit(1)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.out_jsonl.exists():
        args.out_jsonl.unlink()

    backend = args.backend if args.backend is not None else config.LLM_BACKEND
    pipeline = RAGPipeline(llm_backend=backend, llm_model=args.model)

    selective = False if args.no_selective_rerank else None
    rerank_model = (args.rerank_model or "").strip() or None

    manifest: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "eval_file": str(args.eval_file),
        "out_jsonl": str(args.out_jsonl),
        "k": args.k,
        "llm_backend": backend,
        "llm_model": args.model,
        "rerank_model_effective": rerank_model or config.RERANK_MODEL,
        "rerank_top_n": args.rerank_top_n,
        "selective_rerank": selective if selective is not None else bool(config.RERANK_SELECTIVE),
        "note": (
            "use_rerank is NOT forced from CLI so selective rerank policy can turn rerank on/off per query."
        ),
    }
    runs: list[dict[str, Any]] = []

    for it in items:
        qid = it.get("id", "")
        query = (it.get("query") or "").strip()
        if not query:
            continue
        qfam = it.get("query_family")
        trace_extra: dict[str, Any] = {
            "answer_trace_out": args.out_jsonl,
            "answer_eval_id": qid,
            "answer_eval_query_family": qfam,
        }
        try:
            result = pipeline.answer(
                query,
                k=args.k,
                use_parser=True,
                use_retrieval_strategy=True,
                use_rerank=None,
                rerank_top_n=args.rerank_top_n,
                rerank_model=rerank_model,
                selective_rerank=selective,
                trace_extra=trace_extra,
            )
        except Exception as e:
            row = {
                "answer_eval_id": qid,
                "query": query,
                "query_family": qfam,
                "error": f"{type(e).__name__}: {e}",
            }
            runs.append(row)
            with args.out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{qid}] ERROR: {e}", file=sys.stderr)
            continue

        # One JSONL line per query is appended by RAGPipeline (answer_trace_out); do not duplicate here.
        req = result["request"]
        row = {
            "answer_eval_id": qid,
            "query": result.get("query"),
            "query_family": getattr(req, "query_family", None) or qfam,
            "eval_file_query_family": qfam,
            "answer": result.get("answer"),
            "prompt_template_id": result.get("prompt_template_id"),
            "prompt_template_label": result.get("prompt_template_label"),
            "chunk_ids_used": result.get("chunk_ids_used"),
            "llm_backend": result.get("llm_backend"),
            "llm_model": result.get("llm_model"),
            "answer_trace_path": result.get("answer_trace_path"),
            "request_task_type": req.task_type,
            "request_use_rerank": req.use_rerank,
            "rerank_reason": req.rerank_reason,
            "strategy_reason": req.strategy_reason,
            "filters": dict(req.filters),
        }
        runs.append(row)
        print(f"[{qid}] ok  chunks={len(result.get('chunk_ids_used') or [])}")

    args.out_json.write_text(
        json.dumps({"manifest": manifest, "runs": runs}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    mf_path = args.out_json.parent / "answer_eval_manifest.json"
    mf_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out_jsonl} ({len(runs)} lines)")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {mf_path}")


if __name__ == "__main__":
    main()
