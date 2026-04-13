from __future__ import annotations

import argparse
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
