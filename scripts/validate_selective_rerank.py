#!/usr/bin/env python3
"""
Measurement-only pipeline: run labeled eval (optional), then write selective rerank artifacts.

Does not modify retrieval, reranker, hybrid, filters, or parser — delegates to eval + analyze.

Usage:
  cd <repo> && PYTHONPATH=. python scripts/validate_selective_rerank.py --run-eval
  PYTHONPATH=. python scripts/validate_selective_rerank.py --skip-eval   # trace must exist
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _repo_path(path: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()


def _load_analyze_module():  # type: ignore[no-untyped-def]
    path = _ROOT / "scripts" / "analyze_rerank_impact.py"
    spec = importlib.util.spec_from_file_location("analyze_rerank_impact", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run labeled eval (optional) and write selective rerank validation artifacts."
    )
    ap.add_argument(
        "--run-eval",
        action="store_true",
        help="Run eval_labeled_retrieval.py before writing artifacts",
    )
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not run eval (only read --trace-file and write artifacts)",
    )
    ap.add_argument(
        "--labeled-file",
        type=Path,
        default=_ROOT / "eval" / "retrieval_labeled_eval.json",
    )
    ap.add_argument(
        "--trace-file",
        type=Path,
        default=_ROOT / "artifacts" / "retrieval_traces" / "labeled_eval_run.jsonl",
    )
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--rerank-model",
        type=str,
        default=None,
        help="Forwarded to eval_labeled_retrieval.py when --run-eval is used",
    )
    ap.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help="Forwarded to eval_labeled_retrieval.py when --run-eval is used",
    )
    ap.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_ROOT / "artifacts",
    )
    args = ap.parse_args()
    args.labeled_file = _repo_path(args.labeled_file)
    args.trace_file = _repo_path(args.trace_file)
    args.artifacts_dir = _repo_path(args.artifacts_dir)

    if args.run_eval and not args.skip_eval:
        cmd = [
            sys.executable,
            str(_ROOT / "scripts" / "eval_labeled_retrieval.py"),
            "--trace-file",
            str(args.trace_file),
            "--labeled-file",
            str(args.labeled_file),
            "--k",
            str(args.k),
        ]
        if args.rerank_model:
            cmd.extend(["--rerank-model", args.rerank_model])
        if args.rerank_top_n is not None:
            cmd.extend(["--rerank-top-n", str(args.rerank_top_n)])
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(_ROOT))
    elif args.run_eval and args.skip_eval:
        print("Ignoring --run-eval because --skip-eval was set.", file=sys.stderr)

    ar = _load_analyze_module()
    if not args.trace_file.is_file():
        print(f"Trace not found: {args.trace_file}", file=sys.stderr)
        print("Run with --run-eval or pass --trace-file", file=sys.stderr)
        raise SystemExit(1)
    rows = ar._load_jsonl(args.trace_file)
    if not rows:
        print(f"Trace is empty: {args.trace_file}", file=sys.stderr)
        raise SystemExit(1)

    ar.write_selective_validation_artifacts(
        rows,
        args.k,
        args.artifacts_dir,
        trace_path=args.trace_file,
        labeled_file=args.labeled_file,
    )
    print(f"Wrote selective validation artifacts under: {args.artifacts_dir}")
    for name in (
        "eval_selective_rerank.txt",
        "eval_selective_rerank.json",
        "selective_rerank_comparison.txt",
        "selective_rerank_policy_stats.txt",
        "selective_vs_full_rerank_impact.txt",
        "selective_rerank_by_query_family.txt",
        "selective_rerank_examples.txt",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
