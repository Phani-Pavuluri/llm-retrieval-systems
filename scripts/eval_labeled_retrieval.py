"""
Labeled retrieval eval: Precision@k, Recall@k, MRR for vector / hybrid / auto
and optional rerank variants.

Pool diagnostics (Phase 2): avg candidate_pool_size, avg post_filter_count,
pct underfilled after filtering — overall and by query_family.

Traces include gold_* flags for manual failure typing:
  recall: gold_in_candidate_pool false
  filtering: candidate true, post_filter false
  ranking: post_filter true, top_k false

Usage:
  PYTHONPATH=. python scripts/eval_labeled_retrieval.py
  PYTHONPATH=. python scripts/eval_labeled_retrieval.py --k 5 --labeled-file eval/retrieval_labeled_eval.json

After traces are written, summarize rerank impact:
  PYTHONPATH=. python scripts/analyze_rerank_impact.py --trace-file artifacts/retrieval_traces/labeled_eval_run.jsonl

Selective rerank measurement artifacts (comparison, policy stats, examples):
  PYTHONPATH=. python scripts/validate_selective_rerank.py --run-eval
  # or: python scripts/analyze_rerank_impact.py --trace-file ... --selective-validation-artifacts

Reranker model A/B (same eval harness; optional --rerank-model on eval):
  PYTHONPATH=. python scripts/compare_reranker_models.py

Full local validation (tests + eval + artifacts):
  PYTHONPATH=. python scripts/run_rerank_validation.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _repo_path(path: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()

from src import config
from src.query_parser import QueryParser
from src.rerank_policy import apply_selective_rerank_policy
from src.retrieval_with_rerank import retrieve_with_optional_rerank
from src.reranker import CrossEncoderReranker
from src.retrieval_metrics import (
    aggregate_mean,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from src.retrieval_request import RetrievalRequest
from src.retrieval_strategy import apply_strategy_to_request
from src.retriever import Retriever

MODES = ("vector", "hybrid", "auto", "auto_rerank", "auto_selective_rerank", "hybrid_rerank")


def _ranked_chunk_ids(df) -> list[str]:
    if df is None or df.empty or "chunk_id" not in df.columns:
        return []
    return [str(x) for x in df["chunk_id"].tolist()]


def _print_pool_block(title: str, diags: list[dict]) -> None:
    if not diags:
        return
    n = len(diags)
    avg_pool = sum(d["candidate_pool_size"] for d in diags) / n
    avg_post = sum(d["post_filter_count"] for d in diags) / n
    uf = sum(1 for d in diags if d["underfilled_after_filtering"])
    pct = 100.0 * uf / n
    print(
        f"  {title}: n={n} avg_candidate_pool={avg_pool:.1f} "
        f"avg_post_filter={avg_post:.1f} pct_underfilled={pct:.1f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Labeled retrieval metrics + traces.")
    ap.add_argument(
        "--labeled-file",
        type=Path,
        default=_ROOT / "eval" / "retrieval_labeled_eval.json",
    )
    ap.add_argument("--k", type=int, default=5, help="k for P@k / R@k")
    ap.add_argument(
        "--trace-file",
        type=Path,
        default=_ROOT / "artifacts" / "retrieval_traces" / "labeled_eval_run.jsonl",
        help="Append one JSON line per (query × mode) retrieval",
    )
    ap.add_argument(
        "--rerank-model",
        type=str,
        default=None,
        help=(
            "Cross-encoder model id for rerank modes (auto_rerank, hybrid_rerank, "
            "auto_selective_rerank). Default: config RERANK_MODEL."
        ),
    )
    ap.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help="Optional override for rerank candidate pool on rerank modes only (default: config)",
    )
    args = ap.parse_args()
    args.labeled_file = _repo_path(args.labeled_file)
    args.trace_file = _repo_path(args.trace_file)

    rows = json.loads(args.labeled_file.read_text(encoding="utf-8"))
    retriever = Retriever()
    parser = QueryParser()
    shared_reranker: CrossEncoderReranker | None = None

    metrics: dict[str, list[tuple[float | None, float | None, float]]] = {
        m: [] for m in MODES
    }
    by_family: dict[str, dict[str, list[tuple[float | None, float | None, float]]]] = (
        defaultdict(lambda: {m: [] for m in MODES})
    )

    pool_overall: dict[str, list[dict]] = {m: [] for m in MODES}
    pool_by_fam: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {m: [] for m in MODES}
    )

    args.trace_file.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_file.exists():
        args.trace_file.unlink()

    resolved_rerank_model = (args.rerank_model or "").strip() or str(config.RERANK_MODEL)

    for row in rows:
        qid = row["id"]
        qfam = row.get("query_family", "unknown")
        query = row["query"]
        gold = set(row.get("gold_chunk_ids") or [])
        base = parser.parse(query, top_k=args.k)

        for mode in MODES:
            fam_kw = dict(
                query_family=str(qfam),
                rerank_reason="",
                rerank_skipped_due_to_query_family=False,
                rerank_skipped_due_to_metadata_filters=False,
            )
            if mode == "vector":
                req = dataclasses.replace(
                    base,
                    use_hybrid=False,
                    hybrid_alpha=None,
                    hybrid_beta=None,
                    strategy_reason="eval_labeled_vector",
                    candidate_pool_multiplier=None,
                    use_rerank=False,
                    rerank_top_n=None,
                    **fam_kw,
                )
            elif mode == "hybrid":
                req = dataclasses.replace(
                    base,
                    use_hybrid=True,
                    hybrid_alpha=None,
                    hybrid_beta=None,
                    strategy_reason="eval_labeled_hybrid",
                    candidate_pool_multiplier=None,
                    use_rerank=False,
                    rerank_top_n=None,
                    **fam_kw,
                )
            elif mode == "auto":
                req = dataclasses.replace(
                    base,
                    use_hybrid=None,
                    hybrid_alpha=None,
                    hybrid_beta=None,
                    strategy_reason="",
                    candidate_pool_multiplier=None,
                    use_rerank=False,
                    rerank_top_n=None,
                    **fam_kw,
                )
                apply_strategy_to_request(req)
            elif mode == "auto_rerank":
                req = dataclasses.replace(
                    base,
                    use_hybrid=None,
                    hybrid_alpha=None,
                    hybrid_beta=None,
                    strategy_reason="",
                    candidate_pool_multiplier=None,
                    use_rerank=True,
                    rerank_top_n=None,
                    **fam_kw,
                )
                apply_strategy_to_request(req)
            elif mode == "auto_selective_rerank":
                req = dataclasses.replace(
                    base,
                    use_hybrid=None,
                    hybrid_alpha=None,
                    hybrid_beta=None,
                    strategy_reason="",
                    candidate_pool_multiplier=None,
                    use_rerank=None,
                    rerank_top_n=None,
                    **fam_kw,
                )
                apply_strategy_to_request(req)
                apply_selective_rerank_policy(
                    req,
                    selective_enabled=True,
                    query_family_override=str(qfam),
                )
            else:
                req = dataclasses.replace(
                    base,
                    use_hybrid=True,
                    hybrid_alpha=None,
                    hybrid_beta=None,
                    strategy_reason="eval_labeled_hybrid_rerank",
                    candidate_pool_multiplier=None,
                    use_rerank=True,
                    rerank_top_n=None,
                    **fam_kw,
                )

            if mode in ("auto_rerank", "hybrid_rerank", "auto_selective_rerank"):
                if shared_reranker is None:
                    shared_reranker = CrossEncoderReranker(model_name=resolved_rerank_model)
                rer = shared_reranker
            else:
                rer = None

            req = dataclasses.replace(req, rerank_model=resolved_rerank_model)
            if args.rerank_top_n is not None and mode in (
                "auto_rerank",
                "hybrid_rerank",
                "auto_selective_rerank",
            ):
                req = dataclasses.replace(req, rerank_top_n=int(args.rerank_top_n))

            trace_extra = {
                "eval_id": qid,
                "query_family": qfam,
                "eval_mode": mode,
                "gold_chunk_ids": list(gold),
                "trace_out": args.trace_file,
            }
            df = retrieve_with_optional_rerank(
                retriever,
                req,
                trace_extra=trace_extra,
                reranker=rer,
            )
            diag = retriever.last_retrieval_diagnostics
            if diag:
                pool_overall[mode].append(diag)
                pool_by_fam[qfam][mode].append(diag)

            ranked = _ranked_chunk_ids(df)

            p = precision_at_k(ranked, gold, args.k) if gold else None
            r = recall_at_k(ranked, gold, args.k) if gold else None
            mrr = mean_reciprocal_rank(ranked, gold) if gold else 0.0

            metrics[mode].append((p, r, mrr))
            by_family[qfam][mode].append((p, r, mrr))

    k = args.k
    print(f"Labeled file: {args.labeled_file}")
    print(f"Traces: {args.trace_file}")
    print(f"Rerank model (rerank modes): {resolved_rerank_model}")
    if args.rerank_top_n is not None:
        print(f"Rerank top-n override (rerank modes): {args.rerank_top_n}")
    print(f"P@{k} / R@{k} / MRR (queries without gold: P/R omitted from means)\n")

    for mode in MODES:
        ps = [t[0] for t in metrics[mode]]
        rs = [t[1] for t in metrics[mode]]
        mrrs = [t[2] for t in metrics[mode]]
        print(
            f"OVERALL {mode}: "
            f"P@{k}={aggregate_mean(ps):.3f} "
            f"R@{k}={aggregate_mean(rs):.3f} "
            f"MRR={sum(mrrs)/len(mrrs):.3f}"
        )
        _print_pool_block(f"pool [{mode}]", pool_overall[mode])

    print("\nBY query_family:")
    for fam in sorted(by_family.keys()):
        print(f"  [{fam}]")
        for mode in MODES:
            ps = [t[0] for t in by_family[fam][mode]]
            rs = [t[1] for t in by_family[fam][mode]]
            mrrs = [t[2] for t in by_family[fam][mode]]
            print(
                f"    {mode}: P@{k}={aggregate_mean(ps):.3f} "
                f"R@{k}={aggregate_mean(rs):.3f} "
                f"MRR={sum(mrrs)/len(mrrs):.3f}"
            )
            _print_pool_block(f"      pool", pool_by_fam[fam][mode])

    print("\n--- Failure typing (use trace JSONL gold_* fields) ---")
    print("  recall:     gold_in_candidate_pool == false")
    print("  filtering:  gold_in_candidate_pool true, gold_in_post_filter_pool false")
    print("  ranking:    gold_in_post_filter_pool true, gold_in_top_k_results false")


if __name__ == "__main__":
    main()
