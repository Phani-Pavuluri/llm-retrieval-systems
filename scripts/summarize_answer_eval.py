#!/usr/bin/env python3
"""
Phase 4.2–4.3: Summarize manual scores in eval/answer_eval_labeled.json (optional failure_bucket).

Reads scores.grounded / correct / complete (1–3 or null). Writes a short text + JSON summary.

Usage:
  PYTHONPATH=. python scripts/summarize_answer_eval.py
  PYTHONPATH=. python scripts/summarize_answer_eval.py --labeled eval/answer_eval_labeled.json --out artifacts/answer_traces/answer_eval_score_summary.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.answer_eval_constants import ANSWER_EVAL_FAILURE_BUCKET_SET


def _repo_path(path: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()


def _is_fully_scored(scores: dict[str, Any] | None) -> bool:
    if not scores:
        return False
    for k in ("grounded", "correct", "complete"):
        v = scores.get(k)
        if v is None:
            return False
        try:
            int(v)
        except (TypeError, ValueError):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate answer-eval manual scores by query_family.")
    ap.add_argument(
        "--labeled",
        type=Path,
        default=_ROOT / "eval" / "answer_eval_labeled.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "artifacts" / "answer_traces" / "answer_eval_score_summary.txt",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=_ROOT / "artifacts" / "answer_traces" / "answer_eval_score_summary.json",
    )
    args = ap.parse_args()
    args.labeled = _repo_path(args.labeled)
    args.out = _repo_path(args.out)
    args.out_json = _repo_path(args.out_json)

    doc = json.loads(args.labeled.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = list(doc.get("items") or [])

    file_buckets = doc.get("failure_buckets")
    if isinstance(file_buckets, list) and file_buckets:
        allowed: frozenset[str] = frozenset(str(b).strip() for b in file_buckets if str(b).strip())
    else:
        allowed = ANSWER_EVAL_FAILURE_BUCKET_SET
    unknown_buckets: dict[str, int] = defaultdict(int)

    labeled: list[dict[str, Any]] = []
    pending: list[str] = []
    for it in items:
        iid = str(it.get("id", ""))
        sc = it.get("scores") or {}
        if _is_fully_scored(sc):
            labeled.append(it)
        elif iid:
            pending.append(iid)

    lines: list[str] = []
    lines.append("Answer eval score summary")
    lines.append("=" * 50)
    lines.append(f"Labeled file: {args.labeled}")
    lines.append(f"Total items: {len(items)}")
    lines.append(f"Fully scored: {len(labeled)}")
    lines.append(f"Pending (null scores): {len(pending)}")
    if pending:
        lines.append(f"  ids: {', '.join(pending)}")
    lines.append("")

    by_fam: dict[str, list[dict[str, int]]] = defaultdict(list)
    buckets: dict[str, int] = defaultdict(int)
    for it in labeled:
        fam = str(it.get("query_family", "unknown"))
        sc = it.get("scores") or {}
        g, c, o = int(sc["grounded"]), int(sc["correct"]), int(sc["complete"])
        by_fam[fam].append({"grounded": g, "correct": c, "complete": o})
        fb = it.get("failure_bucket")
        if fb is not None and str(fb).strip():
            key = str(fb).strip()
            if key in allowed:
                buckets[key] += 1
            else:
                unknown_buckets[key] += 1

    def mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    lines.append("Means by query_family (1=poor, 3=good)")
    lines.append("-" * 50)
    summary_json: dict[str, Any] = {
        "labeled_count": len(labeled),
        "pending_ids": pending,
        "by_query_family": {},
        "allowed_failure_buckets": sorted(allowed),
        "failure_bucket_counts": dict(buckets) if buckets else {},
        "unknown_failure_bucket_counts": dict(unknown_buckets) if unknown_buckets else {},
    }

    for fam in sorted(by_fam.keys()):
        rows = by_fam[fam]
        gs = [r["grounded"] for r in rows]
        cs = [r["correct"] for r in rows]
        os_ = [r["complete"] for r in rows]
        summary_json["by_query_family"][fam] = {
            "n": len(rows),
            "mean_grounded": round(mean(gs), 3),
            "mean_correct": round(mean(cs), 3),
            "mean_complete": round(mean(os_), 3),
        }
        lines.append(
            f"  [{fam}] n={len(rows)}  "
            f"grounded={mean(gs):.2f}  correct={mean(cs):.2f}  complete={mean(os_):.2f}"
        )

    lines.append("")
    lines.append("Canonical failure_bucket ids (use null when no primary failure):")
    lines.append("  " + ", ".join(sorted(allowed)))

    if buckets:
        lines.append("")
        lines.append("failure_bucket counts (labeled rows only)")
        for k in sorted(buckets.keys()):
            lines.append(f"  {k}: {buckets[k]}")

    if unknown_buckets:
        lines.append("")
        lines.append("WARNING: unknown failure_bucket values (fix typos or extend allowed list):")
        for k in sorted(unknown_buckets.keys()):
            lines.append(f"  {k}: {unknown_buckets[k]}")

    if not labeled:
        lines.append("")
        lines.append("No fully labeled rows yet — fill scores in the eval JSON, then re-run.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    args.out_json.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(text)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
