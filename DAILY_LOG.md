# Daily log (append-only)

Format per entry: **Date** → **Changes** → **Why** → **Impact** → **Notes**.

---

Date: 2026-04-06

Changes:

- Added repo documentation: **README.md** (shortened), **SYSTEM_OVERVIEW.md**, **SYSTEM_EVOLUTION.md**, **DAILY_LOG.md** (this file).
- Documented phases **1 → 4.5** with **problem → fix → impact**; embedded **12-query answer eval** metrics for **baseline vs Phase 4.4 vs Phase 4.5** (overall means + failure buckets).
- Recorded **eval hygiene**: `query_family` in `eval/answer_eval_labeled.json` must match parser; `run_answer_eval.py` row field uses runtime `request.query_family`.

Why:

- Preserve **decision trail** and interview-ready narrative without re-deriving from git history each time.

Impact:

- Readers can see **quantified** Phase 4.4 / 4.5 deltas (e.g. hallucination **1 → 0**, mean correct **2.42 → 2.75** baseline → 4.5) and where metrics are **not** comparable (by-family after taxonomy split).

Notes:

- Early phases (1–3.5) impacts are mostly **qualitative** unless a specific offline table is cited elsewhere.
- Future agents: after eval runs, append a new **Date** block here and extend **SYSTEM_EVOLUTION.md** (Phase / product sections) rather than deleting history.

---

Date: 2026-04-13

Changes:

- Implemented **Phase 4.5** routing improvements (`rerank_policy.infer_query_family`).
- Added query families: **`buyer_risk_issues`**, **`symptom_issue_extraction`**.
- Added **negation-aware** symptom extraction rules and **`family_buyer_risk_issues`** / **`family_symptom_issue_extraction`** prompt templates (`prompt_builder.py`).
- Aligned **selective rerank** with new families (buyer risk **ON**, symptom extraction **OFF**).
- **Runtime-aligned trace fields:** `scripts/run_answer_eval.py` run rows use **`request.query_family`** (plus **`eval_file_query_family`** for the label snapshot).
- **Re-ran** full answer evaluation (`scripts/run_answer_eval.py`); **relabeled** all 12 queries; synced **`query_family`** in `eval/answer_eval_labeled.json` to **runtime** parser output.
- Completed **Phase 4.5 evaluation** documentation — metrics and failure buckets in **SYSTEM_EVOLUTION.md** §10 (**Results** / **Failure** interpretations, **Example-level behavior**, **Key Insight**, **Decision**); expanded **§11** (ranking vs filtering vs recall, when rerank helps/hurts, task-aware prompting).

Why:

- Reduce **wrong_scope** (value vs buyer-risk vs complaints).
- Remove **symptom hallucination**-class failures via explicit attribution / negation rules.
- Raise **correctness** and **completeness** without touching retrieval / hybrid / candidate-pool math.
- **Validate** routing and negation-aware extraction; ensure evaluation reflects **actual** system behavior.
- **Measure** impact across phases (baseline / 4.4 / 4.5) without recomputing new metrics.

Impact:

- **grounded:** 2.92 → 3.00 (baseline → Phase 4.5, 12-item set).
- **correct:** 2.50 → 2.75 (Phase 4.4 → Phase 4.5).
- **complete:** 2.58 → 2.67 (Phase 4.4 → Phase 4.5).
- **hallucination:** 1 → 0 (Phase 4.4 → Phase 4.5).
- **wrong_scope:** 3 → 2 (Phase 4.4 → Phase 4.5).

Notes:

- **ae006:** hallucination class cleared; still **weak_synthesis** (single-chunk-heavy).
- **ae012 / ae009:** routing fixes **successful**.
- **ae001 / ae010:** remaining **scope** issues.
- System **stable**; move **beyond prompt tuning** per **SYSTEM_EVOLUTION.md** §10 **Decision**; next phase should emphasize **product / UX**.
