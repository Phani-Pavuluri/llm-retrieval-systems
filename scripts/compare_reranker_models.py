#!/usr/bin/env python3
"""
Run labeled eval per cross-encoder model and compare metrics (same selective policy).

Does not change parser, strategy, hybrid, filters, or selective policy — only rerank model id.

Usage:
  cd <repo> && PYTHONPATH=. python scripts/compare_reranker_models.py
  PYTHONPATH=. python scripts/compare_reranker_models.py \\
    --models cross-encoder/ms-marco-MiniLM-L-6-v2,cross-encoder/ms-marco-MiniLM-L-12-v2

Optional second phase (same best model, two rerank_top_n values; reported separately):
  PYTHONPATH=. python scripts/compare_reranker_models.py --top-n-sweep 12,20
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config
from src.reranker import try_load_cross_encoder

FAMILIES_RERANK_INTENT = frozenset(
    {"value_complaint", "abstract_complaint_summary", "buyer_risk_issues"}
)


def _repo_path(path: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()


def trace_slug(model: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", model.strip())
    return (s[:100].strip("_") or "default").lower()


def _load_analyze():
    path = _ROOT / "scripts" / "analyze_rerank_impact.py"
    spec = importlib.util.spec_from_file_location("analyze_rerank_impact", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dedupe_preserve(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _selective_gold_stats(ar: Any, rows: list[dict[str, Any]]) -> dict[str, int]:
    sel = [
        r
        for r in rows
        if r.get("eval_mode") == "auto_selective_rerank" and r.get("gold_chunk_ids")
    ]
    c = {"helped": 0, "unchanged": 0, "hurt": 0, "unknown": 0}
    for r in sel:
        mv = ar._rerank_gold_movement(r)
        if mv is None:
            c["unknown"] += 1
        else:
            c[mv] += 1
    return c


def _mrr_selective_on_families(
    ar: Any, rows: list[dict[str, Any]], default_k: int, families: frozenset[str]
) -> float | None:
    sub = [
        r
        for r in rows
        if r.get("eval_mode") == "auto_selective_rerank"
        and str(r.get("query_family", "")) in families
    ]
    if not sub:
        return None
    mrrs: list[float] = []
    for r in sub:
        _, _, m = ar._metrics_for_row(r, default_k)
        mrrs.append(m)
    return sum(mrrs) / len(mrrs) if mrrs else None


def _run_eval(
    *,
    rerank_model: str,
    trace_file: Path,
    labeled_file: Path,
    k: int,
    rerank_top_n: int | None,
) -> int:
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "eval_labeled_retrieval.py"),
        "--rerank-model",
        rerank_model,
        "--trace-file",
        str(trace_file),
        "--labeled-file",
        str(labeled_file),
        "--k",
        str(k),
    ]
    if rerank_top_n is not None:
        cmd.extend(["--rerank-top-n", str(rerank_top_n)])
    print("Running:", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(_ROOT))
    return int(p.returncode or 0)


def _summarize_one_model(
    ar: Any,
    model: str,
    trace_file: Path,
    rows: list[dict[str, Any]],
    k: int,
    error: str | None,
) -> dict[str, Any]:
    if error:
        return {
            "rerank_model": model,
            "trace_file": str(trace_file),
            "error": error,
        }
    payload = ar.build_eval_results_json(rows, k, None, trace_file)
    gold = _selective_gold_stats(ar, rows)
    mrr_intent = _mrr_selective_on_families(ar, rows, k, FAMILIES_RERANK_INTENT)
    _, _, mrr_sel, n_sel = ar._mode_stats(rows, "auto_selective_rerank", k)
    return {
        "rerank_model": model,
        "trace_file": str(trace_file),
        "error": None,
        "n_rows_selective": n_sel,
        "overall_metrics": payload["overall"],
        "by_query_family": payload["by_query_family"],
        "auto_selective_rerank_gold_movement": gold,
        "auto_selective_mrr_on_rerank_intent_families": mrr_intent,
        "auto_selective_mrr_overall": mrr_sel,
    }


def _build_comparison_text(results: list[dict[str, Any]], top_n_extra: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("Cross-encoder reranker model comparison")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Modes: auto, auto_rerank, auto_selective_rerank (same selective policy).")
    lines.append("")

    def fmt_triplet(d: dict[str, Any], mode: str) -> str:
        o = d.get("overall_metrics", {}).get(mode) or {}
        p, r, m, n = o.get("precision_at_k"), o.get("recall_at_k"), o.get("mrr"), o.get("n_rows")
        if not n:
            return "n=0"
        p_s = f"{p:.3f}" if p is not None else "—"
        r_s = f"{r:.3f}" if r is not None else "—"
        return f"P@k={p_s} R@k={r_s} MRR={m:.3f} (n={n})"

    for r in results:
        lines.append(f"## Model: {r.get('rerank_model')}")
        if r.get("error"):
            lines.append(f"  ERROR: {r['error']}")
            lines.append("")
            continue
        lines.append(f"  trace: {r.get('trace_file')}")
        for mode in ("auto", "auto_rerank", "auto_selective_rerank"):
            lines.append(f"  {mode:26s} {fmt_triplet(r, mode)}")
        g = r.get("auto_selective_rerank_gold_movement") or {}
        lines.append(
            f"  selective gold movement: helped={g.get('helped', 0)} "
            f"unchanged={g.get('unchanged', 0)} hurt={g.get('hurt', 0)} unknown={g.get('unknown', 0)}"
        )
        lines.append(
            f"  selective MRR (intent families): {r.get('auto_selective_mrr_on_rerank_intent_families')}"
        )
        lines.append("")

    lines.append("## By query_family — auto_selective_rerank MRR (delta vs first successful model)")
    lines.append("")
    baseline = next((x for x in results if not x.get("error")), None)
    base_by = (baseline or {}).get("by_query_family") or {}
    for r in results:
        if r.get("error"):
            continue
        lines.append(f"### {r['rerank_model']}")
        by = r.get("by_query_family") or {}
        for fam in sorted(by.keys()):
            sel = (by.get(fam) or {}).get("auto_selective_rerank") or {}
            m = sel.get("mrr")
            b = (base_by.get(fam) or {}).get("auto_selective_rerank") or {}
            bm = b.get("mrr")
            if m is None and bm is None:
                continue
            dm = (m - bm) if (m is not None and bm is not None) else None
            dm_s = f"{dm:+.4f}" if dm is not None else "n/a"
            lines.append(f"  [{fam}] MRR={m}  Δvs_baseline={dm_s}")
        lines.append("")

    if top_n_extra:
        lines.append("## Optional: rerank_top_n sweep (best model only; separate from model table)")
        lines.append("")
        for e in top_n_extra:
            lines.append(json.dumps(e, indent=2))
            lines.append("")

    lines.append("## Decision summary")
    lines.append("")
    ok = [r for r in results if not r.get("error") and r.get("n_rows_selective", 0) > 0]
    if not ok:
        lines.append("No successful model runs; fix errors above and re-run.")
        return "\n".join(lines)

    best_mrr = max(ok, key=lambda x: float(x.get("auto_selective_mrr_overall") or 0.0))
    lines.append(
        f"- Best overall MRR under auto_selective_rerank: **{best_mrr['rerank_model']}** "
        f"(MRR={best_mrr.get('auto_selective_mrr_overall'):.4f})"
    )

    best_intent = max(
        ok,
        key=lambda x: float(x.get("auto_selective_mrr_on_rerank_intent_families") or -1.0),
    )
    if best_intent.get("auto_selective_mrr_on_rerank_intent_families") is not None:
        lines.append(
            f"- Best on intent families {sorted(FAMILIES_RERANK_INTENT)}: **{best_intent['rerank_model']}** "
            f"(mean MRR={best_intent.get('auto_selective_mrr_on_rerank_intent_families'):.4f})"
        )

    base = ok[0]
    base_h = (base.get("auto_selective_rerank_gold_movement") or {}).get("hurt", 0)
    lines.append(
        f"- Baseline run (first listed successful model) selective `hurt` count: {base_h}"
    )
    for r in ok[1:]:
        h = (r.get("auto_selective_rerank_gold_movement") or {}).get("hurt", 0)
        tag = "no extra hurt vs baseline order" if h <= base_h else "more hurt than baseline (review)"
        lines.append(
            f"  - {r['rerank_model']}: hurt={h} ({tag})"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare reranker models via labeled eval + traces.")
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated cross-encoder model ids. "
            "Default: config RERANK_MODEL_CANDIDATES (deduped)."
        ),
    )
    ap.add_argument(
        "--labeled-file",
        type=Path,
        default=_ROOT / "eval" / "retrieval_labeled_eval.json",
    )
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_ROOT / "artifacts",
    )
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only read existing trace files (labeled_eval_<slug>.jsonl); skip subprocess eval",
    )
    ap.add_argument(
        "--probe-only",
        action="store_true",
        help="Try sentence_transformers.CrossEncoder(model) for each --models; exit 0 after report",
    )
    ap.add_argument(
        "--top-n-sweep",
        type=str,
        default=None,
        help=(
            "After model comparison, run best selective-MRR model at these rerank_top_n values "
            "(comma-separated, e.g. 12,20). Writes extra traces; reported separately in JSON."
        ),
    )
    args = ap.parse_args()
    args.labeled_file = _repo_path(args.labeled_file)
    args.artifacts_dir = _repo_path(args.artifacts_dir)
    traces_dir = args.artifacts_dir / "retrieval_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        models = _dedupe_preserve(args.models.split(","))
    else:
        models = _dedupe_preserve(list(config.RERANK_MODEL_CANDIDATES))

    if args.probe_only:
        lines = ["Cross-encoder load probe", "=" * 40]
        for m in models:
            ok, err = try_load_cross_encoder(m)
            lines.append(f"{m}: {'OK' if ok else 'FAIL ' + (err or '')}")
        txt = "\n".join(lines) + "\n"
        print(txt)
        (args.artifacts_dir / "reranker_model_probe.txt").write_text(txt, encoding="utf-8")
        return

    ar = _load_analyze()
    results: list[dict[str, Any]] = []
    for model in models:
        slug = trace_slug(model)
        trace_path = traces_dir / f"labeled_eval_{slug}.jsonl"
        err: str | None = None
        rows: list[dict[str, Any]] = []
        if not args.skip_eval:
            rc = _run_eval(
                rerank_model=model,
                trace_file=trace_path,
                labeled_file=args.labeled_file,
                k=args.k,
                rerank_top_n=None,
            )
            if rc != 0:
                err = f"eval_labeled_retrieval.py exited with code {rc}"
        if err is None:
            if not trace_path.is_file():
                err = f"missing trace file after eval: {trace_path}"
            else:
                rows = ar._load_jsonl(trace_path)
                if not rows:
                    err = "trace file empty"
        results.append(_summarize_one_model(ar, model, trace_path, rows, args.k, err))

    top_n_extra: list[dict[str, Any]] = []
    if args.top_n_sweep and not args.probe_only:
        ok = [r for r in results if not r.get("error")]
        if ok:
            best = max(ok, key=lambda x: float(x.get("auto_selective_mrr_overall") or 0.0))
            best_model = str(best["rerank_model"])
            slug = trace_slug(best_model)
            for part in args.top_n_sweep.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    tn = int(part)
                except ValueError:
                    continue
                tpath = traces_dir / f"labeled_eval_{slug}_topn{tn}.jsonl"
                if not args.skip_eval:
                    _run_eval(
                        rerank_model=best_model,
                        trace_file=tpath,
                        labeled_file=args.labeled_file,
                        k=args.k,
                        rerank_top_n=tn,
                    )
                er: str | None = None
                rws: list[dict[str, Any]] = []
                if tpath.is_file():
                    rws = ar._load_jsonl(tpath)
                if not rws:
                    er = "missing or empty trace"
                summ = _summarize_one_model(ar, best_model, tpath, rws, args.k, er)
                summ["rerank_top_n"] = tn
                summ["phase"] = "top_n_sweep"
                top_n_extra.append(summ)

    out_txt = args.artifacts_dir / "reranker_model_comparison.txt"
    out_json = args.artifacts_dir / "reranker_model_comparison.json"
    doc = {
        "k": args.k,
        "labeled_file": str(args.labeled_file),
        "models_requested": models,
        "results": results,
        "top_n_sweep": top_n_extra or None,
    }
    out_json.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    out_txt.write_text(_build_comparison_text(results, top_n_extra), encoding="utf-8")
    print(f"Wrote {out_txt}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
