#!/usr/bin/env python3
"""
Run local verification for reranking: unit tests (tee to disk) + labeled eval + trace analysis.

Does not change retrieval or reranking logic — orchestration and logging only.

From repo root:
  PYTHONPATH=. python scripts/run_rerank_validation.py

Artifacts (after full run):
  artifacts/test_results.txt           — unittest discover -v + rerank-focused modules
  artifacts/eval_labeled_results.txt   — stdout from eval_labeled_retrieval.py
  artifacts/eval_labeled_results.json  — P@k / R@k / MRR recomputed from traces
  artifacts/rerank_impact_summary.txt  — gold movement + improvement/degradation rates
  artifacts/rerank_examples.txt        — short qualitative examples
  artifacts/rerank_impact_report.md    — full markdown report
  artifacts/eval_selective_rerank.txt / .json — metrics (same as analyze --selective-validation-artifacts)
  artifacts/selective_rerank_comparison.txt, selective_rerank_policy_stats.txt, …

Requires: vector store + deps for eval (same as eval_labeled_retrieval.py).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(path: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()


def _child_env() -> dict[str, str]:
    """Ensure repo root is on PYTHONPATH (subprocess does not inherit shell exports reliably)."""
    env = os.environ.copy()
    root = str(_ROOT)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"
    return env


def _run_cmd_tee(cmd: list[str], log_path: Path, cwd: Path, append: bool = False) -> int:
    """Run command; copy stdout+stderr to console and to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as logf:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            logf.write(line)
        p.wait()
        return int(p.returncode or 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run tests + labeled eval + rerank artifact analysis.")
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval + analyze (tests only)",
    )
    ap.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit tests (eval + analyze only)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for labeled eval and analysis",
    )
    ap.add_argument(
        "--trace-file",
        type=Path,
        default=_ROOT / "artifacts" / "retrieval_traces" / "labeled_eval_run.jsonl",
    )
    ap.add_argument(
        "--labeled-file",
        type=Path,
        default=_ROOT / "eval" / "retrieval_labeled_eval.json",
    )
    args = ap.parse_args()
    args.trace_file = _repo_path(args.trace_file)
    args.labeled_file = _repo_path(args.labeled_file)

    artifacts = _ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    if not args.skip_tests:
        test_log = artifacts / "test_results.txt"
        print(f"\n>>> Writing tests to {test_log}\n")
        rc = _run_cmd_tee(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ],
            test_log,
            _ROOT,
        )
        if rc != 0:
            print(f"\nunittest discover exited with {rc}", file=sys.stderr)
            sys.exit(rc)

        with test_log.open("a", encoding="utf-8") as logf:
            logf.write("\n\n=== Rerank-focused (subset) ===\n\n")
        print(f"\n>>> Appending rerank subset to {test_log}\n")
        # Use test file paths + -v before paths (package import tests.* can fail without PYTHONPATH).
        rc2 = _run_cmd_tee(
            [
                sys.executable,
                "-m",
                "unittest",
                "-v",
                "tests/test_reranker.py",
                "tests/test_retriever_pool.py",
                "tests/test_analyze_rerank_impact.py",
                "tests/test_compare_reranker_models.py",
                "tests/test_prompt_builder.py",
                "tests/test_answer_eval_scripts.py",
            ],
            test_log,
            _ROOT,
            append=True,
        )
        if rc2 != 0:
            print(f"\nrerank subset tests exited with {rc2}", file=sys.stderr)
            sys.exit(rc2)

    if args.skip_eval:
        print("\n>>> Skipped eval (--skip-eval)\n")
        sys.exit(0)

    eval_log = artifacts / "eval_labeled_results.txt"
    print(f"\n>>> Running labeled eval; logging to {eval_log}\n")
    rc3 = _run_cmd_tee(
        [
            sys.executable,
            str(_ROOT / "scripts" / "eval_labeled_retrieval.py"),
            "--k",
            str(args.k),
            "--trace-file",
            str(args.trace_file),
            "--labeled-file",
            str(args.labeled_file),
        ],
        eval_log,
        _ROOT,
    )
    if rc3 != 0:
        print(f"\neval_labeled_retrieval exited with {rc3}", file=sys.stderr)
        sys.exit(rc3)

    print(f"\n>>> Writing validation artifacts from traces\n")
    rc4 = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "analyze_rerank_impact.py"),
            "--trace-file",
            str(args.trace_file),
            "--k",
            str(args.k),
            "--validation-artifacts",
            "--selective-validation-artifacts",
            "--labeled-file",
            str(args.labeled_file),
        ],
        cwd=str(_ROOT),
    )
    if rc4.returncode != 0:
        sys.exit(rc4.returncode)

    print(
        "\nDone. See artifacts/*.txt, *.json, rerank_impact_report.md, "
        "selective_rerank_*.txt\n"
    )


if __name__ == "__main__":
    main()
