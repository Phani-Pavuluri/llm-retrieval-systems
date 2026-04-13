"""
Classify labeled-eval retrieval failures from JSONL traces.

Failure = no gold chunk in final top-k (gold_in_top_k_results is false).
Then:
  recall     — gold never in FAISS candidate pool
  filtering  — gold in pool but removed by metadata filters
  ranking    — gold survived filters but not in top-k

When reranking ran, traces may also include gold_in_pre_rerank_top_k /
gold_in_post_rerank_top_k (gold in retrieval top-k before / after rerank).

Usage:
  PYTHONPATH=. python scripts/inspect_labeled_failures.py
  PYTHONPATH=. python scripts/inspect_labeled_failures.py --trace-file artifacts/retrieval_traces/labeled_eval_run.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]


def classify(row: dict[str, Any]) -> str | None:
    if not row.get("gold_chunk_ids"):
        return None
    for key in (
        "gold_in_candidate_pool",
        "gold_in_post_filter_pool",
        "gold_in_top_k_results",
    ):
        if key not in row:
            return "unknown"

    if row["gold_in_top_k_results"]:
        return "success"

    if not row["gold_in_candidate_pool"]:
        return "recall"
    if not row["gold_in_post_filter_pool"]:
        return "filtering"
    return "ranking"


def main() -> None:
    ap = argparse.ArgumentParser(description="Count recall vs filtering vs ranking failures.")
    ap.add_argument(
        "--trace-file",
        type=Path,
        default=_ROOT / "artifacts" / "retrieval_traces" / "labeled_eval_run.jsonl",
    )
    args = ap.parse_args()

    if not args.trace_file.is_file():
        print(f"No trace file: {args.trace_file}")
        print("Run: PYTHONPATH=. python scripts/eval_labeled_retrieval.py")
        return

    rows = []
    with args.trace_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # overall failure counts (only rows with gold)
    failure_counts: dict[str, int] = defaultdict(int)
    by_mode: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    n_gold = 0
    n_failed = 0

    for row in rows:
        label = classify(row)
        if label is None:
            continue
        n_gold += 1
        mode = row.get("eval_mode", "unknown")
        if label == "success":
            by_mode[mode]["success"] += 1
            continue
        n_failed += 1
        failure_counts[label] += 1
        by_mode[mode][label] += 1

    print(f"Trace file: {args.trace_file}")
    n_success = sum(1 for row in rows if classify(row) == "success")

    print(f"Rows with gold labels: {n_gold}")
    print(f"Successes (≥1 gold in top-k): {n_success}")
    print(f"Failures (no gold in top-k): {n_failed}")
    print()

    if n_failed == 0 and n_gold > 0:
        print("All gold-labeled rows have at least one gold chunk in top-k.")
    elif n_failed > 0:
        print("Failure breakdown (among failed queries only):")
        for k in ("recall", "filtering", "ranking", "unknown"):
            c = failure_counts.get(k, 0)
            pct = 100.0 * c / n_failed if n_failed else 0.0
            print(f"  {k}: {c} ({pct:.1f}% of failures)")

    print()
    print("By eval_mode (success + failure types):")
    for mode in sorted(by_mode.keys()):
        d = by_mode[mode]
        parts = [f"{k}={d[k]}" for k in sorted(d.keys())]
        print(f"  {mode}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
