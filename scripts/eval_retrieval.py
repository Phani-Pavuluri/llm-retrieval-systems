"""
Compare vector-only vs fixed hybrid vs rule-based AUTO strategy on the eval set.

Uses QueryParser so task_type/filters match production; AUTO applies
select_retrieval_strategy (issue keywords → hybrid 0.6/0.4, summary phrases →
vector, complaint_summary → hybrid 0.85/0.15, else vector).

Usage:
  PYTHONPATH=. python scripts/eval_retrieval.py
  PYTHONPATH=. python scripts/eval_retrieval.py --summary-only
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

from src.query_parser import QueryParser
from src.retrieval_request import RetrievalRequest
from src.retrieval_strategy import apply_strategy_to_request
from src.retriever import Retriever


def _hint_coverage(text: str, hints: list[str]) -> float:
    if not hints:
        return 0.0
    low = text.lower()
    hit = sum(1 for h in hints if h.lower() in low)
    return hit / len(hints)


def _preview(text: str, n: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Eval retrieval: vector vs hybrid vs auto strategy (by query_family)."
    )
    ap.add_argument(
        "--eval-file",
        type=Path,
        default=_ROOT / "eval" / "retrieval_eval_set.json",
        help="JSON list with id, query, query_family?, hints, ...",
    )
    ap.add_argument("--k", type=int, default=5, help="Final top-k after filters")
    ap.add_argument("--preview", type=int, default=160, help="Chunk preview length")
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only per-family and overall aggregates (no per-query blocks)",
    )
    args = ap.parse_args()

    data = json.loads(args.eval_file.read_text(encoding="utf-8"))
    retriever = Retriever()
    qparser = QueryParser()

    print(f"Eval file: {args.eval_file}")
    print(f"top_k={args.k}, n_queries={len(data)}\n")

    sum_cov_v = sum_cov_h = sum_cov_a = 0.0
    counted = 0

    # family -> lists of coverage (only rows with hints)
    by_family: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"v": [], "h": [], "a": []}
    )

    for row in data:
        qid = row.get("id", "?")
        family = row.get("query_family", "unknown")
        query = row["query"]
        note = row.get("good_retrieval_note", "")
        hints = row.get("relevant_lexical_hints") or []

        base = qparser.parse(query, top_k=args.k)

        req_v = dataclasses.replace(
            base,
            use_hybrid=False,
            hybrid_alpha=None,
            hybrid_beta=None,
            strategy_reason="eval_vector",
        )
        req_h = dataclasses.replace(
            base,
            use_hybrid=True,
            hybrid_alpha=None,
            hybrid_beta=None,
            strategy_reason="eval_hybrid_config",
        )
        req_a = dataclasses.replace(
            base,
            use_hybrid=None,
            hybrid_alpha=None,
            hybrid_beta=None,
            strategy_reason="",
        )
        apply_strategy_to_request(req_a)

        df_v = retriever.retrieve(req_v)
        df_h = retriever.retrieve(req_h)
        df_a = retriever.retrieve(req_a)

        blob_v = " ".join(df_v["text"].astype(str).tolist()) if not df_v.empty else ""
        blob_h = " ".join(df_h["text"].astype(str).tolist()) if not df_h.empty else ""
        blob_a = " ".join(df_a["text"].astype(str).tolist()) if not df_a.empty else ""

        if hints:
            cv = _hint_coverage(blob_v, hints)
            ch = _hint_coverage(blob_h, hints)
            ca = _hint_coverage(blob_a, hints)
            sum_cov_v += cv
            sum_cov_h += ch
            sum_cov_a += ca
            counted += 1
            by_family[family]["v"].append(cv)
            by_family[family]["h"].append(ch)
            by_family[family]["a"].append(ca)

        if not args.summary_only:
            print("=" * 72)
            print(f"[{qid}] family={family} QUERY: {query}")
            print(f"NOTE: {note}")
            if hints:
                print(f"HINTS: {hints}")
            print(
                f"AUTO: hybrid={req_a.use_hybrid} reason={req_a.strategy_reason} "
                f"α={req_a.hybrid_alpha} β={req_a.hybrid_beta}"
            )
            print("-" * 72)
            print("--- VECTOR-ONLY ---")
            for i, (_, r) in enumerate(df_v.head(3).iterrows(), 1):
                print(
                    f"  {i}. score={float(r.get('score', 0)):.4f} | "
                    f"{_preview(r.get('text', ''), args.preview)}"
                )
            print("--- HYBRID (config α/β) ---")
            for i, (_, r) in enumerate(df_h.head(3).iterrows(), 1):
                extra = ""
                if "semantic_score" in df_h.columns:
                    extra = (
                        f" sem={float(r.get('semantic_score', 0)):.3f} "
                        f"kw={float(r.get('keyword_score', 0)):.3f}"
                    )
                print(
                    f"  {i}. score={float(r.get('score', 0)):.4f}{extra} | "
                    f"{_preview(r.get('text', ''), args.preview)}"
                )
            print("--- AUTO (strategy) ---")
            for i, (_, r) in enumerate(df_a.head(3).iterrows(), 1):
                print(
                    f"  {i}. score={float(r.get('score', 0)):.4f} | "
                    f"{_preview(r.get('text', ''), args.preview)}"
                )
            if hints:
                print(
                    f"hint coverage (top-{args.k}): "
                    f"vector={_hint_coverage(blob_v, hints):.2f} "
                    f"hybrid={_hint_coverage(blob_h, hints):.2f} "
                    f"auto={_hint_coverage(blob_a, hints):.2f}"
                )
            print()

    print("=" * 72)
    if counted:
        print(
            "OVERALL mean hint coverage (queries with hints): "
            f"vector={sum_cov_v / counted:.3f} "
            f"hybrid={sum_cov_h / counted:.3f} "
            f"auto={sum_cov_a / counted:.3f}"
        )
        print()
        print("BY query_family (mean hint coverage, hints only):")
        for fam in sorted(by_family.keys()):
            bv = by_family[fam]["v"]
            bh = by_family[fam]["h"]
            ba = by_family[fam]["a"]
            if not bv:
                continue
            print(
                f"  {fam}: n={len(bv)} "
                f"vector={sum(bv)/len(bv):.3f} hybrid={sum(bh)/len(bh):.3f} "
                f"auto={sum(ba)/len(ba):.3f}"
            )
    print("(Heuristic hints only—read chunk text for real judgments.)")


if __name__ == "__main__":
    main()
