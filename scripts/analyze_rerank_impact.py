"""
Phase 3: analyze labeled-eval JSONL traces for reranking impact.

Reads traces produced by scripts/eval_labeled_retrieval.py (do not re-run retrieval).
Optionally runs the eval first to refresh traces.

Usage:
  PYTHONPATH=. python scripts/analyze_rerank_impact.py
  PYTHONPATH=. python scripts/analyze_rerank_impact.py --trace-file artifacts/retrieval_traces/labeled_eval_run.jsonl
  PYTHONPATH=. python scripts/analyze_rerank_impact.py --run-eval
  PYTHONPATH=. python scripts/analyze_rerank_impact.py --validation-artifacts
  PYTHONPATH=. python scripts/analyze_rerank_impact.py --selective-validation-artifacts
  PYTHONPATH=. python scripts/validate_selective_rerank.py --run-eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _repo_path(path: Path) -> Path:
    """Resolve paths relative to repo root (so running from notebooks/ still works)."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()

from src.retrieval_metrics import (
    aggregate_mean,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

MODES = ("vector", "hybrid", "auto", "auto_rerank", "auto_selective_rerank", "hybrid_rerank")
RERANK_MODES = frozenset({"auto_rerank", "auto_selective_rerank", "hybrid_rerank"})
# If a hybrid+selective mode is added to eval, include it in eval summaries (optional).
OPTIONAL_EVAL_MODES = ("hybrid_selective_rerank",)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _gold_set(row: dict[str, Any]) -> set[str]:
    return {str(x) for x in (row.get("gold_chunk_ids") or [])}


def _ranked_ids(row: dict[str, Any]) -> list[str]:
    ids = row.get("final_top_chunk_ids")
    if ids is not None:
        return [str(x) for x in ids]
    hits = row.get("hits") or []
    return [str(h["chunk_id"]) for h in hits if "chunk_id" in h]


def _row_k(row: dict[str, Any], default_k: int) -> int:
    return int(row.get("requested_top_k") or default_k)


def _metrics_for_row(row: dict[str, Any], default_k: int) -> tuple[float | None, float | None, float]:
    gold = _gold_set(row)
    ranked = _ranked_ids(row)
    k = _row_k(row, default_k)
    if not gold:
        return None, None, 0.0
    p = precision_at_k(ranked, gold, k)
    r = recall_at_k(ranked, gold, k)
    m = mean_reciprocal_rank(ranked, gold)
    return p, r, m


def _pct_diff(new: float | None, old: float | None) -> str:
    if new is None or old is None:
        return "n/a"
    if abs(old) < 1e-12:
        if abs(new or 0) < 1e-12:
            return "0%"
        return "n/a (base≈0)"
    return f"{100.0 * (new - old) / old:+.1f}%"


def _abs_diff(new: float | None, old: float | None) -> str:
    if new is None or old is None:
        return "n/a"
    return f"{new - old:+.4f}"


def _aggregate_mode(rows: list[dict[str, Any]], mode: str, default_k: int) -> tuple[list, list, list]:
    ps: list[float | None] = []
    rs: list[float | None] = []
    mrrs: list[float] = []
    for row in rows:
        if row.get("eval_mode") != mode:
            continue
        p, r, m = _metrics_for_row(row, default_k)
        ps.append(p)
        rs.append(r)
        mrrs.append(m)
    return ps, rs, mrrs


def _mean_triplet(ps: list[float | None], rs: list[float | None], mrrs: list[float]) -> tuple[float | None, float | None, float]:
    if not mrrs:
        return None, None, 0.0
    return aggregate_mean(ps), aggregate_mean(rs), sum(mrrs) / len(mrrs)


def _mode_stats(
    rows: list[dict[str, Any]], mode: str, default_k: int
) -> tuple[float | None, float | None, float, int]:
    """(mean P, mean R, mean MRR, n_rows). n_rows==0 means no trace lines for this mode."""
    ps, rs, ms = _aggregate_mode(rows, mode, default_k)
    n = len(ms)
    if n == 0:
        return None, None, 0.0, 0
    p, r, m = _mean_triplet(ps, rs, ms)
    return p, r, m, n


def _compare_block(
    title: str,
    base_p: float | None,
    base_r: float | None,
    base_m: float,
    rr_p: float | None,
    rr_r: float | None,
    rr_m: float,
) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append("| Metric | Base | Rerank | Δ abs | Δ % |")
    lines.append("|--------|------|--------|-------|-----|")
    for name, b, n in (
        ("P@k", base_p, rr_p),
        ("R@k", base_r, rr_r),
        ("MRR", base_m, rr_m),
    ):
        b_fmt = f"{b:.4f}" if b is not None else "—"
        n_fmt = f"{n:.4f}" if n is not None else "—"
        lines.append(
            f"| {name} | {b_fmt} | {n_fmt} | {_abs_diff(n, b)} | {_pct_diff(n, b)} |"
        )
    lines.append("")
    return lines


def _rerank_pre_post_category(row: dict[str, Any]) -> str | None:
    """
    Fine-grained gold-in-top-k movement for rerank trace rows.
    Returns: 'fixed' | 'both_true' | 'both_false' | 'hurt' | None
    """
    if row.get("eval_mode") not in RERANK_MODES:
        return None
    if not row.get("gold_chunk_ids"):
        return None
    pre = row.get("gold_in_pre_rerank_top_k")
    post = row.get("gold_in_post_rerank_top_k")
    if pre is None or post is None:
        return None
    if (not pre) and post:
        return "fixed"
    if pre and post:
        return "both_true"
    if (not pre) and (not post):
        return "both_false"
    if pre and (not post):
        return "hurt"
    return None


def _rerank_gold_movement(row: dict[str, Any]) -> str | None:
    """'helped' | 'unchanged' | 'hurt' | None if not applicable."""
    if row.get("eval_mode") not in RERANK_MODES:
        return None
    if not row.get("gold_chunk_ids"):
        return None
    pre = row.get("gold_in_pre_rerank_top_k")
    post = row.get("gold_in_post_rerank_top_k")
    if pre is None or post is None:
        return None
    if (not pre) and post:
        return "helped"
    if pre and (not post):
        return "hurt"
    return "unchanged"


def _topk_prefix(ids: list[Any], k: int) -> list[str]:
    out = [str(x) for x in (ids or [])][:k]
    return out


def _example_note(row: dict[str, Any], k: int) -> str:
    pre = _topk_prefix(row.get("pre_rerank_top_chunk_ids"), k)
    post = _topk_prefix(row.get("post_rerank_top_chunk_ids"), k)
    if pre == post:
        return "Top-k chunk IDs and order unchanged after rerank."
    moved = [x for x in post if x in pre and post.index(x) != pre.index(x)]
    if moved:
        return f"Order changed within top-k (e.g. {moved[0]!r} moved relative to retrieval ranking)."
    new_ids = [x for x in post if x not in pre[:k]]
    if new_ids:
        return f"Chunks entered top-k after rerank: {new_ids[:3]!r}."
    return "Top-k membership or ordering changed."


def build_report(
    rows: list[dict[str, Any]],
    default_k: int,
    trace_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Rerank impact report (Phase 3)")
    lines.append("")
    lines.append(f"**Source traces:** `{trace_path}`")
    lines.append(f"**Rows loaded:** {len(rows)}")
    rerank_rows = [r for r in rows if r.get("rerank_applied")]
    lines.append(f"**Rows with `rerank_applied`:** {len(rerank_rows)}")
    lines.append("")

    # --- Overall metrics ---
    lines.append("## 1. Metrics by mode (overall)")
    lines.append("")
    lines.append("| Mode | P@k | R@k | MRR | n |")
    lines.append("|------|-----|-----|-----|---|")
    overall: dict[str, tuple[float | None, float | None, float, int]] = {}
    for mode in MODES:
        p, r, m, n = _mode_stats(rows, mode, default_k)
        overall[mode] = (p, r, m, n)
        if n == 0:
            lines.append(f"| {mode} | — | — | — | 0 |")
            continue
        p_s = f"{p:.3f}" if p is not None else "—"
        r_s = f"{r:.3f}" if r is not None else "—"
        lines.append(f"| {mode} | {p_s} | {r_s} | {m:.3f} | {n} |")
    lines.append("")

    # --- By query_family ---
    families = sorted({str(r.get("query_family", "unknown")) for r in rows})
    lines.append("## 2. Metrics by `query_family`")
    lines.append("")
    for fam in families:
        sub = [r for r in rows if str(r.get("query_family", "unknown")) == fam]
        lines.append(f"### Family: `{fam}`")
        lines.append("")
        lines.append("| Mode | P@k | R@k | MRR |")
        lines.append("|------|-----|-----|-----|")
        for mode in MODES:
            p, r, m, n = _mode_stats(sub, mode, default_k)
            if n == 0:
                lines.append(f"| {mode} | — | — | — |")
                continue
            p_s = f"{p:.3f}" if p is not None else "—"
            r_s = f"{r:.3f}" if r is not None else "—"
            lines.append(f"| {mode} | {p_s} | {r_s} | {m:.3f} |")
        lines.append("")

    # --- Deltas ---
    lines.append("## 3. Rerank vs baseline (aggregate deltas)")
    lines.append("")
    auto = overall["auto"]
    auto_rr = overall["auto_rerank"]
    hyb = overall["hybrid"]
    hyb_rr = overall["hybrid_rerank"]
    if auto[3] == 0 or auto_rr[3] == 0:
        lines.append("### auto_rerank vs auto")
        lines.append("")
        lines.append(
            "_No numeric comparison: missing trace rows for `auto` and/or `auto_rerank`. "
            "Re-run `scripts/eval_labeled_retrieval.py` to refresh all eval modes (incl. `auto_selective_rerank`)._"
        )
        lines.append("")
    else:
        lines.extend(
            _compare_block(
                "auto_rerank vs auto",
                auto[0],
                auto[1],
                auto[2],
                auto_rr[0],
                auto_rr[1],
                auto_rr[2],
            )
        )
    if hyb[3] == 0 or hyb_rr[3] == 0:
        lines.append("### hybrid_rerank vs hybrid")
        lines.append("")
        lines.append(
            "_No numeric comparison: missing trace rows for `hybrid` and/or `hybrid_rerank`._"
        )
        lines.append("")
    else:
        lines.extend(
            _compare_block(
                "hybrid_rerank vs hybrid",
                hyb[0],
                hyb[1],
                hyb[2],
                hyb_rr[0],
                hyb_rr[1],
                hyb_rr[2],
            )
        )

    auto_sel = overall["auto_selective_rerank"]
    if auto_sel[3] > 0 and auto_rr[3] > 0:
        lines.extend(
            _compare_block(
                "auto_selective_rerank vs auto_rerank",
                auto_rr[0],
                auto_rr[1],
                auto_rr[2],
                auto_sel[0],
                auto_sel[1],
                auto_sel[2],
            )
        )
    elif auto_sel[3] == 0:
        lines.append("### auto_selective_rerank vs auto_rerank")
        lines.append("")
        lines.append("_No `auto_selective_rerank` rows in trace._")
        lines.append("")
    if auto_sel[3] > 0 and auto[3] > 0:
        lines.extend(
            _compare_block(
                "auto_selective_rerank vs auto",
                auto[0],
                auto[1],
                auto[2],
                auto_sel[0],
                auto_sel[1],
                auto_sel[2],
            )
        )

    # --- Gold movement (rerank modes only) ---
    lines.append("## 4. Rerank effectiveness (gold in top-k, pre vs post)")
    lines.append("")
    lines.append(
        "Counts use trace flags `gold_in_pre_rerank_top_k` and `gold_in_post_rerank_top_k` "
        "on **auto_rerank**, **auto_selective_rerank**, and **hybrid_rerank** rows with non-empty gold labels."
    )
    lines.append("")

    def count_movements(subset: list[dict[str, Any]]) -> dict[str, int]:
        c = {"helped": 0, "unchanged": 0, "hurt": 0}
        for row in subset:
            mv = _rerank_gold_movement(row)
            if mv is None:
                continue
            c[mv] += 1
        return c

    rr_subset = [r for r in rows if r.get("eval_mode") in RERANK_MODES]
    tot = count_movements(rr_subset)  # helped / unchanged / hurt (coarse)
    lines.append("### Overall (all rerank eval rows)")
    lines.append("")
    lines.append(f"- **Helped** (pre=False, post=True): **{tot['helped']}**")
    lines.append(f"- **Unchanged** (same gold-in-top-k flag): **{tot['unchanged']}**")
    lines.append(f"- **Hurt** (pre=True, post=False): **{tot['hurt']}**")
    lines.append("")

    sel_only = [r for r in rows if r.get("eval_mode") == "auto_selective_rerank"]
    if sel_only:
        lines.append("### `auto_selective_rerank`: rerank applied vs skipped (by `query_family`)")
        lines.append("")
        lines.append(
            "| query_family | rows | rerank_applied | skipped_family | "
            "skipped_metadata | skipped_confidence |"
        )
        lines.append(
            "|----------------|------|----------------|----------------|------------------|--------------------|"
        )
        sel_fams = sorted({str(r.get("query_family", "unknown")) for r in sel_only})
        for fam in sel_fams:
            sub = [r for r in sel_only if str(r.get("query_family", "unknown")) == fam]
            lines.append(
                f"| {fam} | {len(sub)} | "
                f"{sum(1 for r in sub if r.get('rerank_applied'))} | "
                f"{sum(1 for r in sub if r.get('rerank_skipped_due_to_query_family'))} | "
                f"{sum(1 for r in sub if r.get('rerank_skipped_due_to_metadata_filters'))} | "
                f"{sum(1 for r in sub if r.get('rerank_skipped_due_to_confidence'))} |"
            )
        lines.append("")

    lines.append("### By `query_family`")
    lines.append("")
    lines.append("| Family | helped | unchanged | hurt |")
    lines.append("|--------|--------|-----------|------|")
    for fam in families:
        sub = [r for r in rr_subset if str(r.get("query_family", "unknown")) == fam]
        c = count_movements(sub)
        if sum(c.values()) == 0:
            continue
        lines.append(
            f"| {fam} | {c['helped']} | {c['unchanged']} | {c['hurt']} |"
        )
    lines.append("")

    # --- Qualitative examples ---
    lines.append("## 5. Representative examples")
    lines.append("")

    helped, neutral, hurt = _pick_example_rows(rows, default_k)

    def emit_example_block(label: str, ex_rows: list[dict[str, Any]]) -> None:
        lines.append(f"### {label}")
        lines.append("")
        if not ex_rows:
            lines.append("_No qualifying rows in this trace._")
            lines.append("")
            return
        for i, row in enumerate(ex_rows, 1):
            k = _row_k(row, default_k)
            lines.append(f"#### Example {i}: `{row.get('eval_id', '?')}`")
            lines.append("")
            lines.append(f"- **Query:** {row.get('query', '')}")
            lines.append(f"- **query_family:** `{row.get('query_family', '')}`")
            lines.append(f"- **eval_mode:** `{row.get('eval_mode', '')}`")
            lines.append(f"- **pre-rerank top-{k} chunk IDs:** `{_topk_prefix(row.get('pre_rerank_top_chunk_ids'), k)}`")
            lines.append(f"- **post-rerank top-{k} chunk IDs:** `{_topk_prefix(row.get('post_rerank_top_chunk_ids'), k)}`")
            pre_g = row.get("gold_in_pre_rerank_top_k")
            post_g = row.get("gold_in_post_rerank_top_k")
            lines.append(
                f"- **Gold moved into top-k:** {bool((not pre_g) and post_g)} "
                f"(pre={pre_g}, post={post_g})"
            )
            lines.append(f"- **Note:** {_example_note(row, k)}")
            lines.append("")

    emit_example_block("Reranking clearly helped (2)", helped)
    emit_example_block("Reranking did nothing observable (2)", neutral)
    emit_example_block("Reranking hurt (up to 1)", hurt)

    lines.append("---")
    lines.append("")
    lines.append(
        "_Generated by `scripts/analyze_rerank_impact.py`. "
        "Re-run `eval_labeled_retrieval.py` then this script to refresh._"
    )
    lines.append("")
    return "\n".join(lines)


def _pick_example_rows(
    rows: list[dict[str, Any]], default_k: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rr_subset = [r for r in rows if r.get("eval_mode") in RERANK_MODES]

    def pick(kind: str, limit: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[Any] = set()
        for row in rr_subset:
            if _rerank_gold_movement(row) != kind:
                continue
            eid = row.get("eval_id")
            if eid in seen:
                continue
            seen.add(eid)
            out.append(row)
            if len(out) >= limit:
                break
        return out

    helped = pick("helped", 2)

    def pick_neutral(limit: int) -> list[dict[str, Any]]:
        pool = [
            r
            for r in rr_subset
            if _rerank_gold_movement(r) == "unchanged" and r.get("gold_chunk_ids")
        ]
        prefer = [
            r
            for r in pool
            if _topk_prefix(r.get("pre_rerank_top_chunk_ids"), _row_k(r, default_k))
            == _topk_prefix(r.get("post_rerank_top_chunk_ids"), _row_k(r, default_k))
        ]
        out: list[dict[str, Any]] = []
        seen: set[Any] = set()
        for src in (prefer, pool):
            for row in src:
                eid = row.get("eval_id")
                if eid in seen:
                    continue
                seen.add(eid)
                out.append(row)
                if len(out) >= limit:
                    return out
        return out

    neutral = pick_neutral(2)
    hurt = pick("hurt", 1)
    return helped, neutral, hurt


def build_eval_results_json(
    rows: list[dict[str, Any]],
    default_k: int,
    labeled_file: Path | None,
    trace_file: Path | None,
) -> dict[str, Any]:
    """Structured P@k / R@k / MRR from traces (matches eval_labeled_retrieval when traces are complete)."""
    families = sorted({str(r.get("query_family", "unknown")) for r in rows})
    overall: dict[str, Any] = {}
    by_family: dict[str, Any] = {}
    for mode in MODES:
        p, r, m, n = _mode_stats(rows, mode, default_k)
        overall[mode] = {
            "precision_at_k": p,
            "recall_at_k": r,
            "mrr": m,
            "n_rows": n,
        }
    for fam in families:
        sub = [r for r in rows if str(r.get("query_family", "unknown")) == fam]
        by_family[fam] = {}
        for mode in MODES:
            p, r, m, n = _mode_stats(sub, mode, default_k)
            by_family[fam][mode] = {
                "precision_at_k": p,
                "recall_at_k": r,
                "mrr": m,
                "n_rows": n,
            }
    k = default_k
    if rows:
        k = _row_k(rows[0], default_k)
    return {
        "k": k,
        "labeled_file": str(labeled_file) if labeled_file else None,
        "trace_file": str(trace_file) if trace_file else None,
        "overall": overall,
        "by_query_family": by_family,
    }


def build_rerank_impact_summary_text(rows: list[dict[str, Any]], default_k: int) -> str:
    lines: list[str] = []
    lines.append("Rerank impact summary (from labeled eval traces)")
    lines.append("=" * 60)
    lines.append("")

    with_gold = [r for r in rows if r.get("gold_chunk_ids")]
    lines.append("All trace rows with gold labels (every query × mode row)")
    lines.append(f"  row_count: {len(with_gold)}")
    if with_gold:
        ok = sum(1 for r in with_gold if r.get("gold_in_top_k_results") is True)
        bad = sum(1 for r in with_gold if r.get("gold_in_top_k_results") is False)
        lines.append(f"  gold_in_top_k_results True (successes): {ok}")
        lines.append(f"  gold_in_top_k_results False (failures): {bad}")
    lines.append("")

    eids = {r.get("eval_id") for r in rows if r.get("eval_id") is not None}
    lines.append(f"Unique eval_id in trace: {len(eids)}")
    lines.append("")

    rr = [r for r in rows if r.get("eval_mode") in RERANK_MODES and r.get("gold_chunk_ids")]
    lines.append("Rerank modes (auto_rerank, auto_selective_rerank, hybrid_rerank), rows with gold")
    lines.append(f"  row_count: {len(rr)}")
    lines.append("")

    sel_rows = [r for r in rows if r.get("eval_mode") == "auto_selective_rerank"]
    if sel_rows:
        lines.append("Selective rerank diagnostics (eval_mode=auto_selective_rerank)")
        lines.append(f"  rows (all): {len(sel_rows)}")
        lines.append(
            f"  rerank_applied True: {sum(1 for r in sel_rows if r.get('rerank_applied'))}"
        )
        lines.append(
            f"  use_rerank_requested True: {sum(1 for r in sel_rows if r.get('use_rerank_requested'))}"
        )
        lines.append(
            "  rerank_skipped_due_to_query_family: "
            f"{sum(1 for r in sel_rows if r.get('rerank_skipped_due_to_query_family'))}"
        )
        lines.append(
            "  rerank_skipped_due_to_metadata_filters: "
            f"{sum(1 for r in sel_rows if r.get('rerank_skipped_due_to_metadata_filters'))}"
        )
        lines.append(
            "  rerank_skipped_due_to_confidence: "
            f"{sum(1 for r in sel_rows if r.get('rerank_skipped_due_to_confidence'))}"
        )
        lines.append("  By query_family (all selective rows, skip counts):")
        for fam in sorted({str(r.get("query_family", "unknown")) for r in sel_rows}):
            sub = [r for r in sel_rows if str(r.get("query_family", "unknown")) == fam]
            lines.append(
                f"    [{fam}] n={len(sub)}  applied={sum(1 for r in sub if r.get('rerank_applied'))}  "
                f"skip_family={sum(1 for r in sub if r.get('rerank_skipped_due_to_query_family'))}  "
                f"skip_meta={sum(1 for r in sub if r.get('rerank_skipped_due_to_metadata_filters'))}  "
                f"skip_conf={sum(1 for r in sub if r.get('rerank_skipped_due_to_confidence'))}"
            )
        lines.append("")

    cats = {"fixed": 0, "both_true": 0, "both_false": 0, "hurt": 0, "unknown": 0}
    for row in rr:
        c = _rerank_pre_post_category(row)
        if c is None:
            cats["unknown"] += 1
        else:
            cats[c] += 1

    lines.append("Gold in top-k: pre_rerank vs post_rerank (trace flags)")
    lines.append(
        f"  fixed_ranking (pre=False, post=True): {cats['fixed']}  "
        "← rerank moved gold into top-k"
    )
    lines.append(
        f"  both_true (pre=True, post=True): {cats['both_true']}  "
        "← gold stayed in top-k"
    )
    lines.append(
        f"  both_false (pre=False, post=False): {cats['both_false']}  "
        "← still no gold in top-k"
    )
    lines.append(
        f"  hurt (pre=True, post=False): {cats['hurt']}  "
        "← rerank dropped gold from top-k"
    )
    lines.append(f"  missing_pre_or_post_flags: {cats['unknown']}")
    lines.append("")

    denom = cats["fixed"] + cats["both_true"] + cats["both_false"] + cats["hurt"]
    if denom > 0:
        imp = 100.0 * cats["fixed"] / denom
        deg = 100.0 * cats["hurt"] / denom
        lines.append("Rates (denominator = fixed + both_true + both_false + hurt)")
        lines.append(f"  improvement_rate (fixed / denom): {imp:.1f}%")
        lines.append(f"  degradation_rate (hurt / denom): {deg:.1f}%")
    else:
        lines.append("Rates: n/a (no rerank rows with complete pre/post gold flags)")
        lines.append("  Re-run scripts/eval_labeled_retrieval.py to populate rerank modes.")
    lines.append("")

    lines.append("By query_family (rerank rows with gold, category counts)")
    lines.append("-" * 40)
    fams = sorted({str(r.get("query_family", "unknown")) for r in rr})
    for fam in fams:
        sub = [r for r in rr if str(r.get("query_family", "unknown")) == fam]
        fc = {"fixed": 0, "both_true": 0, "both_false": 0, "hurt": 0, "unknown": 0}
        for row in sub:
            c = _rerank_pre_post_category(row)
            if c is None:
                fc["unknown"] += 1
            else:
                fc[c] += 1
        lines.append(f"  [{fam}] n={len(sub)}  fixed={fc['fixed']}  both_true={fc['both_true']}  ")
        lines.append(
            f"        both_false={fc['both_false']}  hurt={fc['hurt']}  unknown={fc['unknown']}"
        )
    lines.append("")
    return "\n".join(lines)


def build_rerank_examples_text(rows: list[dict[str, Any]], default_k: int) -> str:
    helped, neutral, hurt = _pick_example_rows(rows, default_k)
    lines: list[str] = []
    lines.append("Rerank qualitative examples (from traces)")
    lines.append("=" * 60)
    lines.append("")

    def block(title: str, ex_rows: list[dict[str, Any]]) -> None:
        lines.append(title)
        lines.append("-" * len(title))
        if not ex_rows:
            lines.append("  (no examples in this trace)")
            lines.append("")
            return
        for i, row in enumerate(ex_rows, 1):
            k = _row_k(row, default_k)
            pre = row.get("gold_in_pre_rerank_top_k")
            post = row.get("gold_in_post_rerank_top_k")
            lines.append(f"  Example {i}  eval_id={row.get('eval_id')!r}  mode={row.get('eval_mode')!r}")
            lines.append(f"    query: {row.get('query', '')}")
            lines.append(f"    query_family: {row.get('query_family', '')}")
            lines.append(f"    pre-rerank top-{k} chunk_ids: {_topk_prefix(row.get('pre_rerank_top_chunk_ids'), k)}")
            lines.append(f"    post-rerank top-{k} chunk_ids: {_topk_prefix(row.get('post_rerank_top_chunk_ids'), k)}")
            lines.append(f"    gold_in_pre_rerank_top_k: {pre}")
            lines.append(f"    gold_in_post_rerank_top_k: {post}")
            lines.append(f"    note: {_example_note(row, k)}")
            lines.append("")

    block("Where reranking helped (up to 2)", helped)
    block("Where reranking did not change gold status (up to 2)", neutral)
    block("Where reranking hurt (up to 1)", hurt)
    return "\n".join(lines)


def _modes_present_in_trace(rows: list[dict[str, Any]]) -> list[str]:
    have = {str(r.get("eval_mode")) for r in rows if r.get("eval_mode")}
    out = [m for m in MODES if m in have]
    for m in OPTIONAL_EVAL_MODES:
        if m in have and m not in out:
            out.append(m)
    return out


def build_eval_selective_rerank_text(
    rows: list[dict[str, Any]], default_k: int
) -> str:
    """Human-readable P@k / R@k / MRR overall and by query_family (trace-derived)."""
    k = _row_k(rows[0], default_k) if rows else default_k
    lines: list[str] = []
    lines.append("Labeled retrieval metrics (from JSONL traces)")
    lines.append(f"k={k}  (P@k / R@k / MRR; queries without gold omit P/R from means)")
    lines.append("")
    modes = _modes_present_in_trace(rows)
    if not modes:
        lines.append("(no eval_mode rows in trace)")
        return "\n".join(lines)

    lines.append("OVERALL")
    lines.append("-" * 60)
    for mode in modes:
        p, r, m, n = _mode_stats(rows, mode, default_k)
        if n == 0:
            lines.append(f"  {mode}: (no rows)")
            continue
        p_s = f"{p:.3f}" if p is not None else "—"
        r_s = f"{r:.3f}" if r is not None else "—"
        lines.append(f"  {mode}: P@{k}={p_s}  R@{k}={r_s}  MRR={m:.3f}  (n={n})")
    lines.append("")

    families = sorted({str(r.get("query_family", "unknown")) for r in rows})
    lines.append("BY query_family")
    lines.append("-" * 60)
    for fam in families:
        sub = [r for r in rows if str(r.get("query_family", "unknown")) == fam]
        lines.append(f"  [{fam}]")
        for mode in modes:
            p, r, m, n = _mode_stats(sub, mode, default_k)
            if n == 0:
                lines.append(f"    {mode}: (no rows)")
                continue
            p_s = f"{p:.3f}" if p is not None else "—"
            r_s = f"{r:.3f}" if r is not None else "—"
            lines.append(f"    {mode}: P@{k}={p_s}  R@{k}={r_s}  MRR={m:.3f}  (n={n})")
        lines.append("")
    return "\n".join(lines)


def _fmt_metric_comparison_block(
    title: str,
    base_label: str,
    sel_label: str,
    base: tuple[float | None, float | None, float, int],
    sel: tuple[float | None, float | None, float, int],
) -> list[str]:
    """Compare selective (sel) vs baseline (base); deltas = sel - base."""
    bp, br, bm, bn = base
    sp, sr, sm, sn = sel
    lines = [title, "=" * len(title), ""]
    lines.append(f"Rows (base): n={bn}   Rows (selective): n={sn}")
    lines.append("")
    if bn == 0 or sn == 0:
        lines.append(
            "Skipping numeric deltas: need trace rows for both modes. "
            "Re-run scripts/eval_labeled_retrieval.py (includes auto_selective_rerank)."
        )
        lines.append("")
        return lines
    lines.append(
        f"| Metric | {base_label} | {sel_label} | Δ abs (sel−base) | Δ % (rel. to base) |"
    )
    lines.append("|--------|------|--------|------------------|---------------------|")
    for name, b, s in (
        ("P@k", bp, sp),
        ("R@k", br, sr),
        ("MRR", bm, sm),
    ):
        b_fmt = f"{b:.4f}" if b is not None else "—"
        s_fmt = f"{s:.4f}" if s is not None else "—"
        lines.append(
            f"| {name} | {b_fmt} | {s_fmt} | {_abs_diff(s, b)} | {_pct_diff(s, b)} |"
        )
    lines.append("")
    lines.append(
        "Interpretation: positive Δ means selective mode scored higher than the baseline "
        "on that metric."
    )
    lines.append("")
    return lines


def build_selective_rerank_comparison_text(
    rows: list[dict[str, Any]], default_k: int
) -> str:
    lines: list[str] = []
    lines.append("Selective rerank vs baselines (trace-derived aggregates)")
    lines.append("")
    lines.append(
        "Deltas are (auto_selective_rerank − baseline). "
        "Percent change is relative to the baseline metric."
    )
    lines.append("")

    auto_rr = _mode_stats(rows, "auto_rerank", default_k)
    auto_sel = _mode_stats(rows, "auto_selective_rerank", default_k)
    auto_base = _mode_stats(rows, "auto", default_k)

    lines.extend(
        _fmt_metric_comparison_block(
            "1) auto_selective_rerank vs auto_rerank (always-on rerank)",
            "auto_rerank",
            "auto_selective_rerank",
            auto_rr,
            auto_sel,
        )
    )
    lines.extend(
        _fmt_metric_comparison_block(
            "2) auto_selective_rerank vs auto (strategy, no rerank)",
            "auto",
            "auto_selective_rerank",
            auto_base,
            auto_sel,
        )
    )
    return "\n".join(lines)


def build_selective_rerank_policy_stats_text(rows: list[dict[str, Any]]) -> str:
    sel = [r for r in rows if r.get("eval_mode") == "auto_selective_rerank"]
    lines: list[str] = []
    lines.append("Selective rerank policy statistics (eval_mode=auto_selective_rerank)")
    lines.append("=" * 60)
    lines.append("")
    if not sel:
        lines.append("No auto_selective_rerank rows in trace.")
        return "\n".join(lines)

    total = len(sel)
    applied = sum(1 for r in sel if r.get("rerank_applied"))
    skipped = total - applied
    n_sf = sum(1 for r in sel if r.get("rerank_skipped_due_to_query_family"))
    n_sm = sum(1 for r in sel if r.get("rerank_skipped_due_to_metadata_filters"))
    n_sc = sum(1 for r in sel if r.get("rerank_skipped_due_to_confidence"))

    lines.append(f"Total queries (rows): {total}")
    lines.append(f"Rerank applied: {applied}")
    lines.append(f"Rerank skipped (no cross-encoder stage): {skipped}")
    lines.append("")
    lines.append("Skip flags (row counts; a row may match multiple categories):")
    lines.append(f"  rerank_skipped_due_to_query_family: {n_sf}")
    lines.append(f"  rerank_skipped_due_to_metadata_filters: {n_sm}")
    lines.append(f"  rerank_skipped_due_to_confidence: {n_sc}")
    lines.append("")

    def pct_share(x: int) -> float:
        return (100.0 * x / total) if total else 0.0

    lines.append("Percentages (denominator = total selective rows):")
    lines.append(f"  Queries reranked: {pct_share(applied):.1f}%")
    lines.append(f"  Rows with skip_family flag: {pct_share(n_sf):.1f}%")
    lines.append(f"  Rows with skip_metadata flag: {pct_share(n_sm):.1f}%")
    lines.append(f"  Rows with skip_confidence flag: {pct_share(n_sc):.1f}%")
    lines.append("")
    return "\n".join(lines)


def _gold_movement_counts(subset: list[dict[str, Any]]) -> dict[str, int]:
    c = {"helped": 0, "unchanged": 0, "hurt": 0, "unknown": 0}
    for row in subset:
        mv = _rerank_gold_movement(row)
        if mv is None:
            c["unknown"] += 1
        else:
            c[mv] += 1
    return c


def build_selective_vs_full_rerank_impact_text(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("Gold in top-k: pre vs post rerank (labeled gold only)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        "Uses trace flags gold_in_pre_rerank_top_k and gold_in_post_rerank_top_k."
    )
    lines.append("")

    for mode, label in (
        ("auto_rerank", "Always-on rerank (auto_rerank)"),
        ("auto_selective_rerank", "Selective rerank (auto_selective_rerank)"),
    ):
        sub = [r for r in rows if r.get("eval_mode") == mode and r.get("gold_chunk_ids")]
        c = _gold_movement_counts(sub)
        lines.append(f"## {label}")
        lines.append(f"Rows with gold: {len(sub)}")
        lines.append(f"  helped   (pre=False, post=True):  {c['helped']}")
        lines.append(f"  unchanged:                        {c['unchanged']}")
        lines.append(f"  hurt     (pre=True, post=False):  {c['hurt']}")
        lines.append(f"  unknown (missing flags):          {c['unknown']}")
        lines.append("")

    lines.append("## Side-by-side summary")
    lines.append("")
    lines.append("| Category | auto_rerank | auto_selective_rerank |")
    lines.append("|----------|-------------|------------------------|")
    for key, lab in (
        ("helped", "helped (False→True)"),
        ("unchanged", "unchanged"),
        ("hurt", "hurt (True→False)"),
        ("unknown", "unknown / missing flags"),
    ):
        s1 = [
            r
            for r in rows
            if r.get("eval_mode") == "auto_rerank" and r.get("gold_chunk_ids")
        ]
        s2 = [
            r
            for r in rows
            if r.get("eval_mode") == "auto_selective_rerank" and r.get("gold_chunk_ids")
        ]
        c1 = _gold_movement_counts(s1)
        c2 = _gold_movement_counts(s2)
        lines.append(f"| {lab} | {c1[key]} | {c2[key]} |")
    lines.append("")
    return "\n".join(lines)


def build_selective_rerank_by_query_family_text(rows: list[dict[str, Any]]) -> str:
    sel = [r for r in rows if r.get("eval_mode") == "auto_selective_rerank"]
    lines: list[str] = []
    lines.append("auto_selective_rerank — breakdown by query_family")
    lines.append("=" * 60)
    lines.append("")
    if not sel:
        lines.append("No selective rows.")
        return "\n".join(lines)

    lines.append(
        "| query_family | count | rerank_applied | rerank_skipped | "
        "helped | unchanged | hurt | unknown |"
    )
    lines.append(
        "|--------------|-------|----------------|----------------|"
        "--------|-----------|------|---------|"
    )
    for fam in sorted({str(r.get("query_family", "unknown")) for r in sel}):
        sub = [r for r in sel if str(r.get("query_family", "unknown")) == fam]
        with_gold = [r for r in sub if r.get("gold_chunk_ids")]
        c = _gold_movement_counts(with_gold)
        applied = sum(1 for r in sub if r.get("rerank_applied"))
        skipped = len(sub) - applied
        lines.append(
            f"| {fam} | {len(sub)} | {applied} | {skipped} | "
            f"{c['helped']} | {c['unchanged']} | {c['hurt']} | {c['unknown']} |"
        )
    lines.append("")
    lines.append(
        "Policy intent check: expect rerank mostly on value_complaint / abstract_complaint_summary; "
        "mostly skipped on rating_scoped_summary / exact_issue_lookup."
    )
    lines.append("")
    return "\n".join(lines)


def _selective_example_explanation(
    row: dict[str, Any],
    mv: str | None,
    full_row: dict[str, Any] | None,
) -> str:
    """One-line narrative for selective_rerank_examples.txt."""
    if mv == "helped":
        return "Gold was outside top-k before rerank and inside after selective rerank."
    if mv == "hurt":
        return "Gold was in top-k before rerank but dropped after selective rerank."
    if not row.get("rerank_applied"):
        fr_mv = _rerank_gold_movement(full_row) if full_row else None
        if fr_mv == "hurt":
            return (
                "Rerank skipped by policy/confidence; same query under always-on "
                "auto_rerank hurt gold-in-top-k — likely avoided degradation."
            )
        if fr_mv == "helped":
            return (
                "Rerank skipped; always-on rerank would have helped — worth reviewing policy."
            )
        return "Rerank skipped; compare to auto_rerank trace for same eval_id if needed."
    if mv == "unchanged":
        return "Gold presence in top-k unchanged pre/post selective rerank."
    return ""


def _selective_skip_reason(row: dict[str, Any]) -> str:
    if row.get("rerank_applied"):
        return "rerank_applied"
    parts: list[str] = []
    if row.get("rerank_skipped_due_to_metadata_filters"):
        parts.append("metadata")
    if row.get("rerank_skipped_due_to_query_family"):
        parts.append("query_family")
    if row.get("rerank_skipped_due_to_confidence"):
        parts.append("confidence")
    if parts:
        return "+".join(parts)
    return row.get("rerank_reason") or "unknown"


def _example_block_selective(
    lines: list[str],
    title: str,
    ex_rows: list[dict[str, Any]],
    default_k: int,
    full_by_id: dict[Any, dict[str, Any]],
) -> None:
    lines.append(title)
    lines.append("-" * len(title))
    if not ex_rows:
        lines.append("  (none in this trace)")
        lines.append("")
        return
    for i, row in enumerate(ex_rows, 1):
        k = _row_k(row, default_k)
        lines.append(f"  Example {i}  eval_id={row.get('eval_id')!r}")
        lines.append(f"    query: {row.get('query', '')}")
        lines.append(f"    query_family: {row.get('query_family', '')}")
        lines.append(
            f"    rerank_applied: {row.get('rerank_applied')}  "
            f"reason_hint: {_selective_skip_reason(row)}  "
            f"trace rerank_reason: {row.get('rerank_reason', '')!r}"
        )
        lines.append(
            f"    pre top-{k}:  {_topk_prefix(row.get('pre_rerank_top_chunk_ids'), k)}"
        )
        lines.append(
            f"    post top-{k}: {_topk_prefix(row.get('post_rerank_top_chunk_ids'), k)}"
        )
        lines.append(
            f"    gold_in_pre_rerank_top_k: {row.get('gold_in_pre_rerank_top_k')}  "
            f"gold_in_post_rerank_top_k: {row.get('gold_in_post_rerank_top_k')}"
        )
        mv = _rerank_gold_movement(row)
        lines.append(f"    gold movement (selective): {mv}")
        lines.append(f"    note: {_example_note(row, k)}")
        expl = _selective_example_explanation(
            row, mv, full_by_id.get(row.get("eval_id"))
        )
        if expl:
            lines.append(f"    explanation: {expl}")
        lines.append("")


def build_selective_rerank_examples_text(
    rows: list[dict[str, Any]], default_k: int
) -> str:
    """Curated examples: helped, skipped (prefer where full rerank hurt), selective hurt."""
    sel = [r for r in rows if r.get("eval_mode") == "auto_selective_rerank"]
    full_by_id: dict[Any, dict[str, Any]] = {
        r.get("eval_id"): r
        for r in rows
        if r.get("eval_mode") == "auto_rerank" and r.get("eval_id") is not None
    }

    helped = [
        r
        for r in sel
        if r.get("gold_chunk_ids") and _rerank_gold_movement(r) == "helped"
    ][:2]

    skip_candidates = [
        r
        for r in sel
        if not r.get("rerank_applied")
        and (
            r.get("rerank_skipped_due_to_query_family")
            or r.get("rerank_skipped_due_to_metadata_filters")
            or r.get("rerank_skipped_due_to_confidence")
        )
    ]

    def skip_sort_key(r: dict[str, Any]) -> tuple[int, str]:
        fr = full_by_id.get(r.get("eval_id"))
        if fr and _rerank_gold_movement(fr) == "hurt":
            return (0, str(r.get("query", "")))
        if fr and _rerank_gold_movement(fr) == "unchanged":
            return (1, str(r.get("query", "")))
        return (2, str(r.get("query", "")))

    skip_sorted = sorted(skip_candidates, key=skip_sort_key)
    skipped_best = skip_sorted[:2]

    hurt_sel = [
        r
        for r in sel
        if r.get("rerank_applied") and _rerank_gold_movement(r) == "hurt"
    ][:1]

    lines: list[str] = []
    lines.append("Selective rerank — representative examples (from traces)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        "Skipped examples prefer queries where always-on auto_rerank *hurt* gold-in-top-k "
        "(same eval_id), illustrating avoided degradation."
    )
    lines.append("")

    _example_block_selective(
        lines,
        "A) Up to 2 queries where selective rerank helped (gold False→True)",
        helped,
        default_k,
        full_by_id,
    )
    _example_block_selective(
        lines,
        "B) Up to 2 queries where rerank was skipped (policy / confidence)",
        skipped_best,
        default_k,
        full_by_id,
    )
    _example_block_selective(
        lines,
        "C) Up to 1 query where selective rerank still hurt (gold True→False), if any",
        hurt_sel,
        default_k,
        full_by_id,
    )
    return "\n".join(lines)


def write_selective_validation_artifacts(
    rows: list[dict[str, Any]],
    default_k: int,
    artifacts_dir: Path,
    trace_path: Path | None = None,
    labeled_file: Path | None = None,
) -> None:
    """Write selective rerank measurement artifacts under artifacts_dir (no retrieval changes)."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    (artifacts_dir / "eval_selective_rerank.txt").write_text(
        build_eval_selective_rerank_text(rows, default_k),
        encoding="utf-8",
    )

    payload = build_eval_results_json(rows, default_k, labeled_file, trace_path)
    payload["modes_present_in_trace"] = _modes_present_in_trace(rows)
    payload["optional_modes_checked"] = list(OPTIONAL_EVAL_MODES)
    payload["note"] = (
        "Metrics recomputed from JSONL traces; matches eval_labeled_retrieval when "
        "traces are from the same run."
    )
    (artifacts_dir / "eval_selective_rerank.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (artifacts_dir / "selective_rerank_comparison.txt").write_text(
        build_selective_rerank_comparison_text(rows, default_k),
        encoding="utf-8",
    )
    (artifacts_dir / "selective_rerank_policy_stats.txt").write_text(
        build_selective_rerank_policy_stats_text(rows),
        encoding="utf-8",
    )
    (artifacts_dir / "selective_vs_full_rerank_impact.txt").write_text(
        build_selective_vs_full_rerank_impact_text(rows),
        encoding="utf-8",
    )
    (artifacts_dir / "selective_rerank_by_query_family.txt").write_text(
        build_selective_rerank_by_query_family_text(rows),
        encoding="utf-8",
    )
    (artifacts_dir / "selective_rerank_examples.txt").write_text(
        build_selective_rerank_examples_text(rows, default_k),
        encoding="utf-8",
    )


def write_validation_artifacts(
    rows: list[dict[str, Any]],
    default_k: int,
    trace_path: Path,
    labeled_file: Path,
    artifacts_dir: Path,
    report_md_path: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    jpath = artifacts_dir / "eval_labeled_results.json"
    jpath.write_text(
        json.dumps(
            build_eval_results_json(rows, default_k, labeled_file, trace_path),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "rerank_impact_summary.txt").write_text(
        build_rerank_impact_summary_text(rows, default_k),
        encoding="utf-8",
    )
    (artifacts_dir / "rerank_examples.txt").write_text(
        build_rerank_examples_text(rows, default_k),
        encoding="utf-8",
    )
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(build_report(rows, default_k, trace_path.resolve()), encoding="utf-8")


def _print_console_summary(
    rows: list[dict[str, Any]],
    default_k: int,
) -> None:
    print("=== Metrics by mode (overall) ===\n")
    overall: dict[str, tuple[float | None, float | None, float, int]] = {}
    for mode in MODES:
        p, r, m, n = _mode_stats(rows, mode, default_k)
        overall[mode] = (p, r, m, n)
        if n == 0:
            print(f"  {mode:14s}  (no rows in trace)")
            continue
        p_s = f"{p:.3f}" if p is not None else "—"
        r_s = f"{r:.3f}" if r is not None else "—"
        print(f"  {mode:14s}  P@k={p_s}  R@k={r_s}  MRR={m:.3f}  (n={n})")

    print("\n=== Rerank vs baseline (Δ abs, Δ %) ===\n")

    def one_pair(bname: str, rname: str) -> None:
        b = overall[bname]
        n = overall[rname]
        if b[3] == 0 or n[3] == 0:
            print(f"  {rname} vs {bname}: skipped (n={b[3]} vs n={n[3]})")
            print()
            return
        print(f"  {rname} vs {bname}:")
        for label, x0, x1 in (
            ("P@k", b[0], n[0]),
            ("R@k", b[1], n[1]),
            ("MRR", b[2], n[2]),
        ):
            print(
                f"    {label}:  {_abs_diff(x1, x0)} abs   {_pct_diff(x1, x0)} rel"
            )
        print()

    if overall["auto_rerank"][3] == 0:
        print(
            "  Note: no `auto_rerank` rows — re-run `eval_labeled_retrieval.py` "
            "(writes vector, hybrid, auto, auto_rerank, auto_selective_rerank, hybrid_rerank).\n"
        )
    one_pair("auto", "auto_rerank")
    one_pair("hybrid", "hybrid_rerank")
    one_pair("auto_rerank", "auto_selective_rerank")

    sel_rows = [r for r in rows if r.get("eval_mode") == "auto_selective_rerank"]
    if sel_rows:
        print("=== Selective rerank diagnostics (auto_selective_rerank) ===\n")
        print(f"  rows: {len(sel_rows)}")
        print(
            f"  rerank_applied True: {sum(1 for r in sel_rows if r.get('rerank_applied'))}"
        )
        print(
            f"  use_rerank_requested True: {sum(1 for r in sel_rows if r.get('use_rerank_requested'))}"
        )
        print(
            "  skipped query_family: "
            f"{sum(1 for r in sel_rows if r.get('rerank_skipped_due_to_query_family'))}"
        )
        print(
            "  skipped metadata_filters: "
            f"{sum(1 for r in sel_rows if r.get('rerank_skipped_due_to_metadata_filters'))}"
        )
        print(
            "  skipped confidence: "
            f"{sum(1 for r in sel_rows if r.get('rerank_skipped_due_to_confidence'))}"
        )
        print()

    print("=== Gold movement (rerank modes, labeled gold only) ===\n")
    rr_subset = [r for r in rows if r.get("eval_mode") in RERANK_MODES]
    for label, key in (
        ("helped (pre=False, post=True)", "helped"),
        ("unchanged", "unchanged"),
        ("hurt (pre=True, post=False)", "hurt"),
    ):
        n = sum(1 for r in rr_subset if _rerank_gold_movement(r) == key)
        print(f"  {label}: {n}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze labeled-eval traces for rerank impact.")
    ap.add_argument(
        "--trace-file",
        type=Path,
        default=_ROOT / "artifacts" / "retrieval_traces" / "labeled_eval_run.jsonl",
        help="JSONL traces (relative paths are resolved from repo root, not cwd)",
    )
    ap.add_argument("--k", type=int, default=5, help="Fallback k if trace row omits requested_top_k")
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "artifacts" / "retrieval_traces" / "rerank_impact_report.md",
        help="Write markdown report here",
    )
    ap.add_argument(
        "--run-eval",
        action="store_true",
        help="Run eval_labeled_retrieval.py first (same --trace-file and --k)",
    )
    ap.add_argument(
        "--rerank-model",
        type=str,
        default=None,
        help="With --run-eval: pass through to eval_labeled_retrieval.py",
    )
    ap.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help="With --run-eval: pass through to eval_labeled_retrieval.py",
    )
    ap.add_argument(
        "--labeled-file",
        type=Path,
        default=_ROOT / "eval" / "retrieval_labeled_eval.json",
        help="Used with --run-eval and for JSON metadata in --validation-artifacts (relative → repo root)",
    )
    ap.add_argument(
        "--validation-artifacts",
        action="store_true",
        help=(
            "Write artifacts/eval_labeled_results.json, rerank_impact_summary.txt, "
            "rerank_examples.txt, and rerank_impact_report.md (does not run eval unless --run-eval)"
        ),
    )
    ap.add_argument(
        "--selective-validation-artifacts",
        action="store_true",
        help=(
            "Write selective rerank measurement files: eval_selective_rerank.txt/json, "
            "selective_rerank_comparison.txt, selective_rerank_policy_stats.txt, "
            "selective_vs_full_rerank_impact.txt, selective_rerank_by_query_family.txt, "
            "selective_rerank_examples.txt (under --artifacts-dir; does not run eval unless --run-eval)"
        ),
    )
    ap.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_ROOT / "artifacts",
        help="Directory for --validation-artifacts outputs",
    )
    args = ap.parse_args()

    args.trace_file = _repo_path(args.trace_file)
    args.labeled_file = _repo_path(args.labeled_file)
    args.artifacts_dir = _repo_path(args.artifacts_dir)
    if args.validation_artifacts:
        args.out = args.artifacts_dir / "rerank_impact_report.md"
    else:
        args.out = _repo_path(args.out)

    if args.run_eval:
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

    trace_abs = args.trace_file
    if not trace_abs.is_file():
        print("analyze_rerank_impact: trace file not found (exits with code 1).", file=sys.stderr)
        print(f"  Resolved path: {trace_abs}", file=sys.stderr)
        print(f"  Repo root: {_ROOT}", file=sys.stderr)
        print(f"  cwd: {Path.cwd()}", file=sys.stderr)
        print(
            "  Fix: run labeled eval from repo root, or pass --trace-file (relative paths use repo root).\n"
            "    cd <repo> && PYTHONPATH=. python scripts/eval_labeled_retrieval.py\n"
            "    PYTHONPATH=. python scripts/analyze_rerank_impact.py "
            "--trace-file artifacts/retrieval_traces/labeled_eval_run.jsonl",
            file=sys.stderr,
        )
        raise SystemExit(1)

    rows = _load_jsonl(trace_abs)
    if not rows:
        print(f"analyze_rerank_impact: trace file is empty: {trace_abs}", file=sys.stderr)
        raise SystemExit(1)

    _print_console_summary(rows, args.k)
    if args.validation_artifacts:
        write_validation_artifacts(
            rows,
            args.k,
            trace_abs,
            args.labeled_file,
            args.artifacts_dir,
            args.out,
        )
        print(f"Wrote validation artifacts under: {args.artifacts_dir}")
        print(f"  - eval_labeled_results.json")
        print(f"  - rerank_impact_summary.txt")
        print(f"  - rerank_examples.txt")
        print(f"  - rerank_impact_report.md -> {args.out}")
    if args.selective_validation_artifacts:
        write_selective_validation_artifacts(
            rows,
            args.k,
            args.artifacts_dir,
            trace_path=trace_abs,
            labeled_file=args.labeled_file,
        )
        print(f"Wrote selective validation artifacts under: {args.artifacts_dir}")
        print("  - eval_selective_rerank.txt / .json")
        print("  - selective_rerank_comparison.txt")
        print("  - selective_rerank_policy_stats.txt")
        print("  - selective_vs_full_rerank_impact.txt")
        print("  - selective_rerank_by_query_family.txt")
        print("  - selective_rerank_examples.txt")
    if not args.validation_artifacts and not args.selective_validation_artifacts:
        report = build_report(rows, args.k, trace_abs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote report: {args.out}")


if __name__ == "__main__":
    main()
