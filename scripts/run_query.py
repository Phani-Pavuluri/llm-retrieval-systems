from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG query with OpenAI or Ollama LLM backend."
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "ollama"],
        default=None,
        help="LLM backend (default: config LLM_BACKEND)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the backend (default: OPENAI_MODEL or OLLAMA_MODEL from config)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Question to ask",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--no-parser",
        action="store_true",
        help="Skip rule-based query parser; use raw query as embedding text only",
    )
    hy = parser.add_mutually_exclusive_group()
    hy.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid (semantic + keyword) ranking for this run",
    )
    hy.add_argument(
        "--vector-only",
        action="store_true",
        help="Force vector-only ranking (ignore config HYBRID_RETRIEVAL)",
    )
    parser.add_argument(
        "--no-strategy",
        action="store_true",
        help="Disable rule-based retrieval strategy (use config HYBRID_RETRIEVAL only)",
    )
    rr = parser.add_mutually_exclusive_group()
    rr.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking for this run (overrides config)",
    )
    rr.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking for this run (overrides config)",
    )
    sel_rr = parser.add_mutually_exclusive_group()
    sel_rr.add_argument(
        "--selective-rerank",
        action="store_true",
        help="Enable task-aware selective rerank policy (default: config RERANK_SELECTIVE)",
    )
    sel_rr.add_argument(
        "--no-selective-rerank",
        action="store_true",
        help="Disable selective rerank policy; use_rerank follows config only unless --rerank/--no-rerank",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help="Max filtered candidates to rerank (default: config RERANK_TOP_N)",
    )
    parser.add_argument(
        "--rerank-model",
        type=str,
        default=None,
        help="Cross-encoder model id for reranking (default: config RERANK_MODEL)",
    )
    parser.add_argument(
        "--answer-trace",
        action="store_true",
        help="Append one answer-generation JSONL record (see config ANSWER_TRACE_DIR)",
    )
    parser.add_argument(
        "--answer-trace-file",
        type=Path,
        default=None,
        help="With --answer-trace: JSONL path (default: artifacts/answer_traces/answers.jsonl)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Structured explanation: evidence (chunks in LLM context), reasoning_summary, confidence — not model attribution. No extra LLM call",
    )
    args = parser.parse_args()

    from src.rag_pipeline import RAGPipeline

    backend = args.backend if args.backend is not None else config.LLM_BACKEND
    pipeline = RAGPipeline(llm_backend=backend, llm_model=args.model)

    query = args.query or (
        "What kinds of counterfeit or defective product complaints appear in these reviews?"
    )
    use_hybrid: bool | None = None
    if args.hybrid:
        use_hybrid = True
    elif args.vector_only:
        use_hybrid = False

    use_rerank: bool | None = None
    if args.rerank:
        use_rerank = True
    elif args.no_rerank:
        use_rerank = False

    selective_rerank: bool | None = None
    if args.selective_rerank:
        selective_rerank = True
    elif args.no_selective_rerank:
        selective_rerank = False

    trace_extra: dict | None = None
    if args.answer_trace:
        from src.answer_trace import default_answer_trace_path

        p = args.answer_trace_file
        if p is not None:
            p = Path(p).expanduser()
            if not p.is_absolute():
                p = (_ROOT / p).resolve()
        trace_extra = {"answer_trace_out": p or default_answer_trace_path()}

    result = pipeline.answer(
        query,
        k=args.k,
        use_parser=not args.no_parser,
        use_hybrid=use_hybrid,
        use_retrieval_strategy=not args.no_strategy,
        use_rerank=use_rerank,
        rerank_top_n=args.rerank_top_n,
        rerank_model=args.rerank_model,
        selective_rerank=selective_rerank,
        trace_extra=trace_extra,
        explain=args.explain,
    )

    print("\nQUERY:")
    print(result["query"])

    _eff = (
        result["request"].use_hybrid
        if result["request"].use_hybrid is not None
        else config.HYBRID_RETRIEVAL
    )
    print("\nRETRIEVAL:", "hybrid" if _eff else "vector-only")
    if result["request"].strategy_reason:
        print("STRATEGY:", result["request"].strategy_reason)
    if _eff:
        ra = result["request"]
        a = ra.hybrid_alpha if ra.hybrid_alpha is not None else config.HYBRID_ALPHA
        b = ra.hybrid_beta if ra.hybrid_beta is not None else config.HYBRID_BETA
        print(f"HYBRID WEIGHTS: alpha={a} beta={b}")
    print("\nTASK TYPE:", result["request"].task_type)
    if result["request"].filters:
        print("FILTERS:", result["request"].filters)

    from src.rerank_policy import infer_query_family

    qfam = result["request"].query_family or infer_query_family(result["request"])
    print("QUERY_FAMILY:", qfam)
    if result["request"].rerank_reason:
        print("RERANK_REASON:", result["request"].rerank_reason)

    diag = pipeline.retriever.last_retrieval_diagnostics or {}
    rd = diag.get("rerank_decision") or {}
    if rd:
        print(
            "RERANK_TRACE: "
            f"use_rerank_requested={rd.get('use_rerank_requested')} "
            f"use_rerank_effective={rd.get('use_rerank_effective')} "
            f"rerank_top_n_effective={rd.get('rerank_top_n_effective')} "
            f"skip_family={rd.get('rerank_skipped_due_to_query_family')} "
            f"skip_meta={rd.get('rerank_skipped_due_to_metadata_filters')} "
            f"skip_conf={rd.get('rerank_skipped_due_to_confidence')}"
        )

    _ur = (
        result["request"].use_rerank
        if result["request"].use_rerank is not None
        else config.RERANK_ENABLED
    )
    if _ur:
        print("\nRERANK: enabled (request/config)")
        eff_rm = result["request"].rerank_model or config.RERANK_MODEL
        print("RERANK MODEL:", eff_rm)
        n_cand = diag.get("rerank_candidate_count")
        if n_cand is not None:
            print("RERANK CANDIDATE COUNT:", n_cand)
    else:
        print("\nRERANK: disabled (request/config)")

    print("\nPROMPT TEMPLATE:", result.get("prompt_template_id"), "-", result.get("prompt_template_label"))
    if result.get("answer_trace_path"):
        print("ANSWER TRACE:", result["answer_trace_path"])

    print("\nANSWER:")
    print(result["answer"])

    if args.explain and result.get("explanation"):
        ex = result["explanation"]
        print("\n--- EXPLAIN (Phase 5.1) ---")
        rs = ex.get("reasoning_summary") or {}
        print("\nWHY THIS ANSWER (system summary):")
        print(rs.get("summary_line", ""))
        print(
            f"  query_family={rs.get('query_family')}  retrieval_mode={rs.get('retrieval_mode')}  "
            f"rerank_applied={rs.get('rerank_applied')}  strategy={rs.get('strategy_reason')!r}"
        )
        if rs.get("filters_applied"):
            print(f"  filters={rs.get('filters_applied')}")
        if rs.get("rating_scope"):
            print(f"  rating_scope={rs.get('rating_scope')}")
        print(f"  prompt_template={rs.get('prompt_template_id')}")
        cf = ex.get("confidence") or {}
        print(
            f"\nCONFIDENCE: {cf.get('confidence_label')} (score={cf.get('confidence_score')}, heuristic)"
        )
        for r in cf.get("confidence_reasons") or []:
            print(f"  - {r}")
        print("\nEVIDENCE PROVIDED TO THE MODEL (context window; not which chunks the model relied on most):")
        for i, ev in enumerate(ex.get("evidence") or [], start=1):
            cid = ev.get("chunk_id", "")
            sid = ev.get("source_id") or "—"
            print(f"  {i}. {cid} (source={sid})")
            rs_ = []
            if ev.get("rerank_score") is not None:
                rs_.append(f"rerank_score={ev['rerank_score']:.4f}")
            if ev.get("retrieval_score") is not None:
                rs_.append(f"retrieval_score={ev['retrieval_score']:.4f}")
            elif ev.get("score") is not None:
                rs_.append(f"score={ev['score']:.4f}")
            if ev.get("semantic_score") is not None:
                rs_.append(f"semantic={ev['semantic_score']:.4f}")
            if ev.get("keyword_score") is not None:
                rs_.append(f"keyword={ev['keyword_score']:.4f}")
            if rs_:
                print("     " + "  ".join(rs_))
            txt = (ev.get("chunk_text") or "")[:220].replace("\n", " ")
            if len(ev.get("chunk_text") or "") > 220:
                txt += "…"
            print(f"     {txt}")
        print("\n(JSON)")
        print(json.dumps(ex, indent=2, ensure_ascii=False, default=str))

    print("\nRETRIEVED CHUNKS:")
    preferred = [
        "asin",
        "review_rating",
        "brand",
        "category",
        "score",
        "retrieval_score",
        "rerank_score",
        "final_rank",
        "text",
    ]
    df = result["retrieved_chunks"]
    cols = [c for c in preferred if c in df.columns]
    if cols:
        print(df[cols].to_string(index=False))
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
