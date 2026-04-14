# System evolution (deep technical log)

## Purpose of this document

This document records the detailed evolution of the system, including problems observed, diagnostic reasoning, changes applied, and measured outcomes. It is intended to help with understanding how the system works, why specific design decisions were made, and how performance improved across phases.

**Measurement spine:** Unless stated otherwise, answer-quality numbers refer to the **12-query manual answer eval** (`eval/answer_eval_labeled.json`, `scripts/run_answer_eval.py`, `scripts/summarize_answer_eval.py`) and summarized artifacts under `artifacts/answer_traces/`. Retrieval and rerank behavior are additionally supported by optional JSONL traces when enabled (`RETRIEVAL_TRACE_*`, `ANSWER_TRACE_*` in `src/config.py`).

**Engineering goal (unchanged across phases):** Build a **transparent RAG** stack over product-review chunks—**correct metadata filtering**, **retrieval strategy** selection, **optional reranking**, and **grounded answers**—with enough **instrumentation** to separate failures into retrieval vs rerank vs generation vs routing.

---

## 2. Pre–Phase 1 baseline (reference stack)

**One-line purpose:** Document the stack **before** structured query parsing, so later phases have an explicit “from what” reference.

### Context / starting point

- **Architecture:** Chunked product reviews → embeddings (SentenceTransformers default) → **FAISS** vector store. Query → single embedding → top‑k retrieval → **one generic prompt** → LLM (Ollama or OpenAI via `src/llm.py`).
- **Behavior:** No `task_type` or structured `filters` from natural language; vector order only (no cross-encoder second stage); little structured observability tying failures to pool size vs filter vs generation.
- **Assumptions:** Answer quality was tightly coupled to whichever chunks surfaced; prompts did not encode strong **scope** (complaint vs praise, rating intent) or **synthesis** structure.

### Problem observed (why Phase 1 was necessary)

- Rating / brand constraints in user text did not reliably become **metadata filters**, so filtered pools and user intent diverged.
- Failures were hard to attribute: recall vs ranking vs “model went off” without traces.

### Diagnosis

No dedicated parse step; queries were opaque strings for embedding only—so filters and task shape could not be applied consistently downstream.

### Changes made

None in this document section (historical snapshot). The **first code changes** begin in Phase 1.

### Why these changes should work

N/A for baseline.

### Results

This slice is the implicit **“before”** for Phase 4.4 / 4.5 tables later (means **grounded 2.92**, **correct 2.42**, **complete 2.42** at the **Baseline** column in §10’s comparison tables).

### Tradeoffs / limitations

Vector-only ranking; no hybrid fusion; no selective rerank policy; weak scope control in a single generic prompt.

### Key takeaway

- Without **parse → request shape**, filters and prompts fight the wrong battle.  
- Without **traces**, “bad RAG” is not diagnosable.  
- Baseline **correct/complete** (~2.42) left headroom for routing + prompt work.

### Artifacts / evidence

- Chunk + index layout under `data/vector_store/` (see `src/config.py` paths).  
- Early single-template behavior described in `src/prompt_builder.py` history (task/family templates added in Phase 4).

---

## 3. Phase 1 — Query parsing

**One-line purpose:** Turn natural-language queries into a structured **`RetrievalRequest`** (`task_type`, optional `filters`, `query_text`, `top_k`) so metadata and prompts can align with user intent.

### Context / starting point

Baseline stack (§2): no structured extraction of rating or brand constraints; no complaint vs general **task** distinction for retrieval or prompting.

### Problem observed

Concrete pattern: phrases like “low-rated”, “1-star”, or brand-like spans did not reliably become **`filters`**, so `apply_metadata_filters` either did nothing or mis-scoped the pool relative to the question.

### Diagnosis

**Evidence:** Misaligned post-filter counts vs user-stated rating intent; manual inspection showed queries embedded constraints that never reached `RetrievalRequest.filters`. Root cause: **no parse layer**—only raw string → embedding.

### Changes made

- **`src/query_parser.py`:** `QueryParser.parse()` → `RetrievalRequest` with `task_type` (e.g. `complaint_summary` vs `general_qa`), optional **`filters`** (`review_rating`, brand heuristics), `query_text`, `top_k`.
- **`infer_query_family()`** (initial wiring) to connect parsing to downstream **policy** and later **prompt selection** (`src/rerank_policy.py`, Phase 4+).
- Rule-based keyword / phrase maps for **rating** and **brand**; complaint vs general **task** inference.

### Why these changes should work

Structured filters apply deterministically in `metadata_filters.apply_metadata_filters`; traces can later expose **post_filter_count** vs `top_k`, making filter bugs **visible** instead of folded into “bad retrieval.”

### Results

**Qualitative (pre–full-eval):** Filters apply in the retriever path; strategy and traces (Phase 2+) can show when the pool is empty or narrow because of parse output.

### Tradeoffs / limitations

Heuristic parsing can miss phrasing variants; **wrong parser branch** later showed up as **`ignored_filter`** in eval (addressed in Phase 4.4, e.g. **ae002** one-star vs `max: 3`).

### Key takeaway

- **Parser alignment** is a first-class dependency for fair **metadata-narrowed** retrieval.  
- **`query_family` / `task_type`** become the spine for rerank policy and prompts in later phases.

### Artifacts / evidence

- `src/query_parser.py`, `src/retrieval_request.py`, `tests/test_query_parser.py`.  
- Trace fields referencing `filters` once `retrieval_trace` enabled (Phase 2).

---

## 4. Phase 2 — Retrieval diagnostics

**One-line purpose:** Make retrieval failures **legible** (pool size, filters, final chunk ids) via structured JSONL traces.

### Context / starting point

Phase 1 could set filters, but operators still could not tell whether a bad answer came from **empty pool**, **wrong filter**, **low recall**, or generation—no durable artifact per retrieve.

### Problem observed

Failures were attributed to “bad RAG” without evidence for **k_fetch**, **metadata filter**, or **embedding recall** as the driver of empty or skewed evidence.

### Diagnosis

**Evidence gap:** No structured log of oversample multiplier, **k_fetch**, filters, and **final chunk list**. Root cause: observability was not a first-class output of `Retriever.retrieve()`.

### Changes made

- **`src/retrieval_trace.py`** + **`RETRIEVAL_TRACE_ENABLED`**, **`RETRIEVAL_TRACE_DIR`** in `src/config.py`: JSONL records per retrieve (strategy, oversample, candidate counts, **`underfilled_after_filtering`**, chunk ids).
- **`src/retrieval_metrics.py`:** helpers for offline analysis of traces vs labels.

### Why these changes should work

When traces are enabled, each retrieve emits a reproducible record; cross-checking **`ignored_filter`** in answer eval vs actual `filters` in trace becomes possible.

### Results

**Qualitative:** Reproducible debugging; supports later comparison scripts (e.g. rerank impact analyses under `scripts/`).

### Tradeoffs / limitations

Trace volume and I/O when enabled; traces do not by themselves fix ranking— they **instrument** it.

### Key takeaway

- **Observability** is prerequisite for honest **failure bucket** attribution.  
- **underfilled_after_filtering** supports confidence / explanation heuristics later (Phase 5.1).

### Artifacts / evidence

- `artifacts/retrieval_traces/` (when `RETRIEVAL_TRACE_ENABLED` is true).  
- `src/retrieval_trace.py`, `src/retriever.py` hooks.

---

## 5. Phase 3 — Hybrid retrieval + cross-encoder reranking

**One-line purpose:** Add a **second relevance signal** (lexical fusion + optional query–passage cross-encoder scores) on top of dense retrieval.

### Context / starting point

Dense-only FAISS ordering; single similarity signal. Issue-heavy queries with exact token cues could rank oddly vs user phrasing.

### Problem observed

Concrete pattern: exact tokens (e.g. “counterfeit”) sometimes underrepresented in pure embedding order; vector-only order not always best for user phrasing on issue-heavy pulls.

### Diagnosis

**Evidence:** Qualitative inspection + later selective-rerank experiments: single-signal ranking is weak when lexical overlap matters. Root cause: **no (query, passage) relevance model** and no fusion of lexical with semantic scores on the candidate pool.

### Changes made

- **`src/hybrid_scoring.py`:** optional **lexical + semantic** fusion on candidates (`HYBRID_ALPHA` / `HYBRID_BETA` in `src/config.py`).
- **`src/reranker.py`**, **`src/retrieval_with_rerank.py`:** cross-encoder rerank (sentence-transformers; e.g. MiniLM / Electra-style models configurable via `RERANK_MODEL`) over top‑N candidates; wiring through `Retriever` / pipeline.

### Why these changes should work

Hybrid adds a complementary signal where keywords align; cross-encoder directly scores query–passage fit on a **truncated pool**, improving head relevance when the right rows are already retrieved.

### Results

**Qualitative / eval-dependent:** Better head ranking on issue-heavy queries **when rerank runs**; additional **latency and CPU** when rerank + hybrid are on.

### Tradeoffs / limitations

Does not fix **absence** of correct chunks from the pool; does not fix **parser/filter** bugs; global rerank-on-everything later proved wasteful → Phase 3.5.

### Key takeaway

- **Rerank helps reordering**, not recall creation.  
- **Cost** and **query-shape fit** must be governed—leading to selective policy (§6).

### Artifacts / evidence

- `src/hybrid_scoring.py`, `src/reranker.py`, `src/retrieval_with_rerank.py`.  
- `scripts/analyze_rerank_impact.py` and related comparison outputs (where used in experiments).

---

## 6. Phase 3.5 — Selective reranking

**One-line purpose:** Gate rerank **on/off** from **query_family**, metadata narrowness, and policy flags so compute and reordering behavior match query shape.

### Context / starting point

Phase 3 made rerank available globally; blind rerank on every query wasted compute and sometimes **reordered** already-narrow filtered sets in ways that did not fix scope/synthesis failures.

### Problem observed

Concrete pattern: rerank on **rating-narrowed** or **extraction-shaped** pools added cost without fixing **wrong_scope** / attribution issues driven by prompts.

### Diagnosis

**Evidence:** Policy experiments + eval intuition: rerank is a **reordering** tool; on narrow pools it does not repair **template** or **filter** mistakes. Root cause: rerank treated as a **global toggle**, unrelated to **query_family** or **filter narrowness**.

### Changes made

- **`src/rerank_policy.py`:** `apply_selective_rerank_policy()` sets `request.use_rerank` when `config.RERANK_SELECTIVE` and request does not override.
- **Family-based ON/OFF:** `RERANK_ON_QUERY_FAMILIES` vs `RERANK_OFF_QUERY_FAMILIES`.
- **Skip rerank** when **rating metadata filter** active (policy choice on narrow pools).
- **`rerank_skipped_due_to_*`** diagnostics surfaced into traces / explanation summaries (later Phase 5.1).

### Why these changes should work

Broad complaint/value pulls benefit from cross-encoder reordering of top‑N; narrow or extraction-shaped queries avoid odd reordering and save cost—aligned with later Phase 4.5 choice: **`symptom_issue_extraction` → rerank OFF**.

### Results

**Qualitative:** Rerank **on** for broad complaint/value-style families; **off** for rating-filtered and exact-issue families until Phase 4.5 further split families (§10).

### Tradeoffs / limitations

Wrong **family** assignment still routes rerank incorrectly—**routing** quality becomes the bottleneck (addressed in Phase 4.5).

### Key takeaway

- **Selective rerank** links **cost** and **behavior** to query shape.  
- Rerank **cannot** substitute for correct **filters** or **prompt templates**.

### Artifacts / evidence

- `src/rerank_policy.py`, `tests/test_rerank_policy.py`.  
- `rerank_decision` fields in retrieval diagnostics consumed by `src/explanation_builder.py`.

---

## 7. Evaluation framework (cross-cutting)

**One-line purpose:** Hold the system accountable with **manual 1–3 scores**, **failure buckets**, and reproducible scripts—so phases 4+ can show **before/after** with explicit **query_family** alignment.

### Context / starting point

Qualitative “looks better” without stable rubric; risk of **by-family** summaries lying after routing changes.

### Problem observed

Need comparable **grounded / correct / complete** scores and canonical **failure buckets** tied to the same `query_family` the runtime parser emits.

### Diagnosis

Without labeled JSON and aggregation scripts, prompt/rerank changes are not defensible numerically; **`query_family` drift** between label file and runtime invalidates by-family bars.

### Changes made

- **`eval/answer_eval_labeled.json`:** per-item scores and failure bucket; **must** keep `query_family` aligned with `QueryParser` / `infer_query_family` after taxonomy changes.
- **`src/answer_eval_constants.py`:** canonical buckets: `hallucination`, `incomplete_summary`, `wrong_scope`, `ignored_filter`, `weak_synthesis`, `overfocused_on_single_chunk`, `insufficient_evidence_handled_poorly`.
- **`scripts/run_answer_eval.py`:** runs all **12** items through `RAGPipeline.answer()`; writes `artifacts/answer_traces/answer_eval_runs.json` with runtime `query_family` and `eval_file_query_family` snapshot.
- **`scripts/summarize_answer_eval.py`:** means overall and **by `query_family`** → `answer_eval_score_summary*.json` / `.txt`.

### Why these changes should work

Fixed rubric + bucket taxonomy makes regressions visible; runtime `query_family` on run rows makes aggregates **honest** after routing edits.

### Results

Phase 4.4 and 4.5 tables in §9–§10 are entirely grounded in this framework (no new metrics invented here).

### Tradeoffs / limitations

Manual labeling cost; small **n=12** slice—**not** a production monitoring system; **grounded** hit practical ceiling **3.00** on the set by Phase 4.5.

### Key takeaway

- **Answer-level metrics** and **buckets** are the bar; retrieval-only charts are insufficient (see also **SYSTEM_OVERVIEW.md** — *Evaluation philosophy*).  
- **`query_family` on eval rows must match parser output** or by-family summaries are meaningless.

### Artifacts / evidence

- `eval/answer_eval_labeled.json`.  
- `artifacts/answer_traces/answer_eval_runs.json`, `artifacts/answer_traces/answer_eval_score_summary*.json` / `.txt`.  
- `scripts/run_answer_eval.py`, `scripts/summarize_answer_eval.py`.

---

## 8. Phase 4 — Prompt & answer improvements (task / family templates)

**One-line purpose:** Stop using one generic instruction for all tasks—introduce **family-specific** prompts, grounding blocks, and **answer traces** tied to template id and chunk ids.

### Context / starting point

Phase 3(+3.5) could surface better chunks, but generation still used a **single generic** prompt style; good chunks still produced **list-like** answers, **mixed sentiment** in complaint questions, and **rating drift**.

### Problem observed

Concrete patterns (qualitative, pre-4.4 numeric table): answers **listed** evidence without synthesis discipline; **mixed sentiment** under complaint-style questions; answers **drifted** from stated rating intent even when filters were correct.

### Diagnosis

**Evidence:** Manual scoring friction; same chunk sets producing weak answers when template did not enforce **scope** or **structure**. Root cause: **single instruction set**; no explicit **grounding** / chunk citation discipline in the prompt contract.

### Changes made

- **`src/prompt_builder.py`:** **task / family** templates (abstract complaints, value, exact issue, rating-scoped, general); **grounding** block + evidence block with `[Chunk n | id=…]`.
- **`src/answer_trace.py`** + pipeline hooks: **`prompt_template_id`**, **`chunk_ids_used`**, backend/model in JSONL for post-hoc labeling.

### Why these changes should work

Family sections constrain **answer shape** and **scope**; grounding rules tie claims to cited chunks; traces align labels with **which template** fired.

### Results

**Qualitative:** Easier manual scoring; necessary setup for Phase 4.4 **numeric** movement on **correct/complete** and buckets.

### Tradeoffs / limitations

Did not yet fix **parser order** bugs (**ae002**) or fine-grained **routing** (**ae009**, **ae012**, **ae006**)—handled in Phases 4.4–4.5.

### Key takeaway

- **Same chunks + wrong template** → **wrong_scope** and weak answers **without** changing retrieval (see §11 synthesis).  
- Tracing **template id** is essential for attributing generation failures.

### Artifacts / evidence

- `src/prompt_builder.py`, `tests/test_prompt_builder.py`.  
- `src/answer_trace.py`, `artifacts/answer_traces/` when `ANSWER_TRACE_ENABLED`.

---

## 9. Phase 4.4 — Parser ordering + prompt discipline (numeric eval pass)

**One-line purpose:** Fix **literal one-star** parsing vs broad “negative” heuristic; tighten **synthesis**, **negation**, **multi-chunk**, and **rating scope** text in prompts.

### Context / starting point

Phase 4 templates existed, but: (1) **ae002** showed **ignored_filter** when one-star language produced **`review_rating` max 3** instead of **equality 1**; (2) complaint summaries **led with positives**; (3) symptom answers **over-causal**; (4) **weak_synthesis** / **wrong_scope** still high vs desired bar.

### Problem observed

Concrete eval-linked failures:

- **ae002:** Query asked **one-star**; parser + “negative” heuristic produced **`max: 3`** → **`ignored_filter`** in labels.  
- Complaint summaries **led with positives**; symptom answers **over-causal**.  
- **weak_synthesis** and **wrong_scope** counts high relative to target.

### Diagnosis

**Evidence:** `eval/answer_eval_labeled.json` bucket tags + manual reading of answers vs `filters` in trace; **root causes:** (1) order of heuristic phrase checks in **`_maybe_add_rating_filters`**; (2) under-specified **scope** / **synthesis** / **negation** blocks in prompts.

### Changes made

**Parser (`src/query_parser.py`)**

- Literal **one-star** phrases including hyphenated **`one-star`**, checked **before** the broad **`negative` → review_rating max 3** branch → `filters["review_rating"] = 1` (equality).

**Prompts (`src/prompt_builder.py`)**

- **Structured synthesis** — theme headings + bullets + chunk tags for complaint / rating-scoped templates.  
- **Negative-only** blocks where applicable — do not pad complaint answers with unrelated praise from same excerpt.  
- **Causal / claim rules** — e.g. do not invert “no rash” into “causes rash”.  
- **Multi-chunk** rules — use ≥2 chunks when multiple are relevant.  
- **Rating scope** paragraph mirroring actual `filters` (equality vs max).

**Tests:** `tests/test_query_parser.py`, `tests/test_prompt_builder.py`.

### Why these changes should work

Parser order fix makes **one-star** intent match **`review_rating == 1`** in metadata filter; prompt blocks reduce **scope bleed** and **unsupported causal** jumps; structured synthesis attacks **weak_synthesis** listing behavior.

### Results (12-item eval, labels after Phase 4.4 rerun)

| Metric | Baseline (initial labels) | After Phase 4.4 |
|--------|---------------------------|-----------------|
| Mean **grounded** | **2.92** | **2.92** |
| Mean **correct** | **2.42** | **2.50** |
| Mean **complete** | **2.42** | **2.58** |
| **weak_synthesis** (count / 12) | 3 | **2** |
| **ignored_filter** | 1 | **0** |
| **wrong_scope** | 2 | **3** |
| **hallucination** | 1 | 1 |

**Interpretation (from eval + relabel notes):** Literal **rating** alignment improved (**ignored_filter** cleared for one-star); **correct/complete** nudged up; **wrong_scope** ticked up partly due to stricter labeling / complaint lead-in (**ae001**), not retrieval changes.

### Tradeoffs / limitations

**wrong_scope** could **increase** when labeling stricter even if retrieval unchanged; **hallucination** not eliminated until Phase 4.5 routing + extraction templates (**ae006**).

### Key takeaway

- **Parser order** bugs look like “bad answers” but are **`ignored_filter`** failures.  
- Prompt structure moves **correct/complete** before routing split in 4.5.

### Artifacts / evidence

- `eval/answer_eval_labeled.json` (post–4.4 alignment).  
- Summaries compared to baseline in internal notes; Phase 4.4 row in §10 tables.  
- `src/query_parser.py`, `src/prompt_builder.py`.

---

## 10. Phase 4.5 — Routing + extraction precision

**One-line purpose:** Split ambiguous **intent** into dedicated **`query_family`s** and **templates** (buyer risk, symptom extraction, value vs abstract complaint), align **selective rerank**, and relabel eval **`query_family`** to runtime.

### Context / starting point

After Phase 4.4, metrics improved modestly, but: **wrong_scope** persisted (buyer-risk answered as **value**; complaint summaries **mixing non-complaint** content); **hallucination** on **symptom extraction** (symptoms from *other* products or negated experiences folded into “this product”); **weak_synthesis** still **2** items—answers **over-relying on a single chunk** when multiple excerpts were relevant.

### Problem observed

Concrete examples (eval ids):

- **ae006** — symptom list: hallucination / attribution risk; later **weak_synthesis** (single-chunk lean).  
- **ae012** — buyer risk: answered with **value** framing.  
- **ae009** — value complaint: **wrong_scope** from wrong template family.  
- **ae001** / **ae010** — residual **wrong_scope** (mixed sentiment / smell-attribute noise on issue-focused queries).

### Diagnosis

**Evidence:** Failure buckets + manual read + trace/template ids:

1. **Routing ambiguity** — value vs risk vs summary mapped to overlapping template paths (`value_complaint` vs generic complaint).  
2. **No dedicated extraction task** — symptom queries treated like **`exact_issue_lookup`** summaries → model **summarized** instead of **extracting** with attribution rules.  
3. **Negation gap** — model did not separate symptoms **for this product** vs **not** / **other products**.

### Changes made

#### Routing (`src/rerank_policy.py` — `infer_query_family`)

**New families:** `buyer_risk_issues`, `symptom_issue_extraction`.

**Rules (representative):**

- Symptom extraction detected **before** generic issue keywords so `"rash"` does not hijack list-style symptom queries.  
- **Value intent** separated (explicit price / worth / value phrases → `value_complaint`, including when `task_type` is `general_qa`).  
- Buyer-risk via **`_BUYER_RISK_MARKERS`** (watch out, beware, red flag, etc.).  
- **`complaint_summary` no longer defaults to `value_complaint`** — residual complaints → **`abstract_complaint_summary`** unless value markers match.

#### Prompts (`src/prompt_builder.py`)

- **`family_buyer_risk_issues`** — risks/pitfalls; avoids value framing unless user asks price/worth.  
- **`family_symptom_issue_extraction`** — strict extraction; chunk-level discipline.

#### Negation-aware extraction rules

Tightened symptom + exact-issue prompts: only symptoms **explicitly tied to this product**; exclude other-product and **negated** statements; avoid inference beyond excerpt wording (extends existing causal / grounding blocks).

#### Selective rerank alignment

- **`buyer_risk_issues` → rerank ON** (with broad complaint/value pulls).  
- **`symptom_issue_extraction` → rerank OFF** (extraction-shaped; avoid odd reorder on narrow pools).

#### Eval / trace hygiene

- **`eval/answer_eval_labeled.json`:** each item’s **`query_family`** aligned to **`QueryParser` / `infer_query_family`** output for honest **`summarize_answer_eval.py` by-family** aggregates.  
- **`scripts/run_answer_eval.py`:** run row **`query_family`** = runtime `request.query_family` + `eval_file_query_family` snapshot.

### Why these changes should work

Dedicated **families + templates** match user intent shape; extraction template + negation rules remove **ae006**-style attribution folding; buyer-risk template removes **ae012** value leakage; value markers fix **ae009**; rerank ON/OFF per family matches when second-stage rank helps vs hurts.

### Results (3-phase comparison, 12-query manual eval)

**Overall means**

| Phase | grounded | correct | complete |
|-------|----------|---------|----------|
| **Baseline** | 2.92 | 2.42 | 2.42 |
| **Phase 4.4** | 2.92 | 2.50 | 2.58 |
| **Phase 4.5** | **3.00** | **2.75** | **2.67** |

**Interpretation:** **Grounded** reached **3.00** on this set (ceiling for rubric/data); largest jump vs 4.4 is **correct (+0.25)**; **complete (+0.09)**.

**Failure buckets (counts)**

| Phase | wrong_scope | weak_synthesis | hallucination | ignored_filter |
|-------|-------------|----------------|---------------|----------------|
| **Baseline** | 2 | 3 | 1 | 1 |
| **Phase 4.4** | 3 | 2 | 1 | 0 |
| **Phase 4.5** | **2** | **2** | **0** | **0** |

- **Hallucination:** **1 → 0** (eliminated on labeled pass vs 4.4 and baseline).  
- **wrong_scope:** **3 → 2** vs 4.4.  
- **weak_synthesis:** **2 → 2** (unchanged vs 4.4—main remaining non-scope bucket).  
- **ignored_filter:** **0** since 4.4.

**Example-level (post 4.5)**

- **ae006:** Hallucination **resolved**; still **weak_synthesis** (leans on one chunk).  
- **ae012:** Routing → `buyer_risk_issues` + template; **fully correct and complete** on Phase 4.5 relabel.  
- **ae009:** Routing → `value_complaint` + selective rerank; prior **wrong_scope** resolved.  
- **ae001** / **ae010:** Still **wrong_scope** (mixed-sentiment / attribute noise).

**By `query_family` (Phase 4.5 only):** After taxonomy split, by-family bars are **not row-comparable** to baseline / 4.4; see `artifacts/answer_traces/answer_eval_score_summary_phase45.json`.

### Tradeoffs / limitations

- **Retrieval** not the bottleneck on this 12-item slice after 4.5; **rerank** not the bottleneck once selective policy matches shape.  
- **Remaining:** prompt **scope** (**ae001**, **ae010**), **multi-chunk breadth** (**ae006**).  
- **Decision:** Further **prompt-only micro-tuning** on this slice **deprioritized**; effort shifts to **Phase 5 product layer** on top of core frozen through **4.5**.

### Key takeaway

- **Routing + extraction** eliminated **hallucination** on the labeled pass and fixed high-leverage misroutes (**ae009**, **ae012**, **ae006** attribution).  
- **weak_synthesis** persists—needs breadth / fusion work or richer eval, not rerank alone.  
- **Eval labels must track runtime `query_family`.**

### Artifacts / evidence

- `eval/answer_eval_labeled.json` (relabeled families).  
- `artifacts/answer_traces/answer_eval_score_summary_phase45.json`.  
- `src/rerank_policy.py`, `src/prompt_builder.py`, `scripts/run_answer_eval.py`.

---

## 11. Cross-phase technical synthesis

Consolidated patterns that recur across §3–§10 (no new metrics here).

### Ranking vs filtering vs recall

- **Recall failures** — correct chunks never in pool; rerank **cannot** fix absence.  
- **Filtering failures** — wrong metadata window (e.g. one-star vs max≤3); looks like “bad answers” but is **parser / filter** alignment (**ae002** path).  
- **Ranking failures** — right chunks in pool but ordered low; hybrid + rerank help **when** policy enables rerank on broad pulls.

### When reranking helps vs hurts

- **Helps** — broad semantic pulls where cross-encoder reorders **top‑N** (complaint/value-style families).  
- **Hurts or adds little** — narrow or **extraction**-shaped queries, or metadata-narrowed pools where reorder adds cost without fixing **scope** or **attribution**.

### Task-aware prompting

- **Same chunks, wrong template** → **wrong_scope** / weak answers **without** retrieval change.  
- **`query_family` / template routing** as important as retrieval quality once observability exists.

### Design learnings (carry-forward)

1. **Retrieval ≠ answer quality** — good chunks + rerank still yield **weak_synthesis**, **scope drift**, subtle **hallucination** without strict **prompt + routing**.  
2. **Ranking vs filtering** — wrong **metadata filter** is invisible to “better reranker” work.  
3. **Selective rerank** — ON for broad complaint/value; OFF for narrow / extraction-shaped per policy tables.  
4. **Prompt routing** — defaulting **all complaints** to **value** caused **wrong_scope**; explicit markers + families fixed **ae009**, **ae012**, **ae006** class issues.  
5. **Eval `query_family` must match runtime** — else by-family summaries lie after taxonomy changes.

---

## 12. Phase 5 — Product layer (access + explainability)

**One-line purpose:** Make the frozen core (through **4.5**) **usable** and **inspectable** via API and UI—**without** moving retrieval, rerank, or prompt logic into clients.

**Status:** **5.1**, **5.2**, and **5.3** implemented in repo; **5.4** planned.

### 12.1 Phase 5.1 — Explainable answers

**One-line purpose:** Expose **evidence in LLM context**, **templated reasoning summary**, and **heuristic confidence** alongside the answer (no extra LLM for “why”).

#### Context / starting point

Pipeline produced answers and optional JSONL **answer traces**, but there was no single **structured** object for product/API consumers listing **which chunks were in context**, **why rerank on/off**, or a **confidence** hint.

#### Problem observed

Trust and debuggability gaps: hard to show **what the model saw** vs **what the system decided** (rerank skips, hybrid vs vector, filters) without reading logs.

#### Diagnosis

**Evidence:** Product roadmap goals + operator need to align UI with **runtime routing** without dumping raw dataframes. Root cause: no first-class **`explanation`** object on `RAGPipeline.answer()`.

#### Changes made

- **`src/explanation_builder.py`:** `build_explanation_payload()` → `evidence[]` (rows from final context passed to LLM; **not** “which chunks the model relied on most”), `reasoning_summary{}` (template + diagnostics: `retrieval_mode`, rerank flags, `query_family`, `strategy_reason`, filters, `summary_line`, `prompt_template_id`), `confidence{}` (heuristic label, score, `confidence_reasons`).  
- **`src/rag_pipeline.py`:** `explain=True` path; passes `last_retrieval_diagnostics` into explanation.  
- **`scripts/run_query.py`:** `--explain` CLI.  
- **`tests/test_explanation_builder.py`:** regression tests.

#### Why these changes should work

Evidence list is **deterministic** from the retrieval dataframe; reasoning strings are **template-built** from known flags—no chain-of-thought leak; confidence encodes simple rules operators can sanity-check.

#### Results

**Qualitative:** Stable **`explanation`** keys for API (§12.2); supports trust copy in **API_CONTRACT.md** (heuristic confidence, no cross-chunk contradiction signal yet).

#### Tradeoffs / limitations

**Confidence (`src/explanation_builder._confidence`):**

| Implemented signal | Not implemented |
|--------------------|-----------------|
| Evidence count, underfill, “insufficient evidence” answer prefix | Cross-chunk **agreement** NLP |
| Rerank on/off + multi-chunk vs `chunk_ids_used` hint | Cross-chunk **conflict** detection |
| Heuristic score → label | Calibrated probability, rerank **stability** numeric |

**Open limitations (eval-linked):** shallow **multi-chunk** fusion (**ae006** breadth); **ae001** mixed-sentiment scope drift; **ae010** attribute/smell noise; **no** automated reasoning verification; **no** conflicting-evidence flag in payload.

#### Key takeaway

- **Explainability** stays **non-LLM** for rationale text—pipeline state only.  
- **Evidence** = **provided to model**, not attribution.  
- **Confidence** is a **directional** heuristic, not a risk score.

#### Artifacts / evidence

- `src/explanation_builder.py`, `tests/test_explanation_builder.py`.  
- **SYSTEM_OVERVIEW.md** — *Explain response envelope*, *Evaluation philosophy*.

---

### 12.2 Phase 5.2 — HTTP API

**One-line purpose:** Expose **`POST /query`** + **`GET /health`** with stable success/error JSON—thin wrapper over `RAGPipeline.answer()`.

#### Context / starting point

CLI (`scripts/run_query.py`) and in-process calls only; no programmatic contract for UI or integrations.

#### Problem observed

Need **typed request/response**, **errors without stack traces**, and **metadata** for clients without coupling to internal Python objects.

#### Diagnosis

FastAPI + Pydantic gives validation and OpenAPI; mapping pipeline dict → **`answer` / `explanation` / `metadata`** avoids leaking `RetrievalRequest` objects.

#### Changes made

- **`src/api.py`:** `QueryRequest`, `POST /query`, `GET /health`, structured `error` object; lazy `RAGPipeline` init **after** body validation to avoid loading the index on **400**s.  
- **`src/rag_pipeline.py`:** optional per-call **`llm_backend`**, **`llm_model`** overrides for API.  
- **`API_CONTRACT.md`**, **`PRODUCT_ROADMAP.md`**, **`SYSTEM_OVERVIEW.md`** updates.

#### Why these changes should work

Single endpoint reduces surface area; error codes (`invalid_request`, `backend_unavailable`, `pipeline_error`, `internal_error`) map from **known** exception classes without exposing internals.

#### Results

**Qualitative:** Documented contract + `/docs`; **`tests/test_api.py`** covers validation, success shape, error mapping (mocked pipeline).

#### Tradeoffs / limitations

**`backend_unavailable`** buckets multiple distinct LLM transport failures (finer codes possible later—see **API_CONTRACT.md**). Answer tracing remains **server-side config** only (no per-request trace flag in 5.2).

#### Key takeaway

- **API is a façade** — all intelligence stays in **`src/`** pipeline modules.  
- **Schema additive** evolution documented in **API_CONTRACT.md**.

#### Artifacts / evidence

- `src/api.py`, `tests/test_api.py`, **`API_CONTRACT.md`**.

---

### 12.3 Phase 5.3 — Streamlit chat UI

**One-line purpose:** Minimal **HTTP client** UI calling the API—display answer, optional explanation blocks, metadata, errors.

#### Context / starting point

API existed; no default graphical surface for demos.

#### Problem observed

Operators still needed curl or scripts to show the system interactively.

#### Diagnosis

Streamlit is sufficient for **local** prototyping without duplicating pipeline logic in JS.

#### Changes made

- **`ui/chat_ui.py`**, **`ui/chat_helpers.py`:** `build_query_json`, health check, markdown line builders; **`scripts/run_chat_ui.py`**.  
- **`requirements.txt`:** `streamlit`.  
- Docs: **API_CONTRACT.md** (Chat UI section), **PRODUCT_ROADMAP.md**.

#### Why these changes should work

UI only **`requests.post`** / **`requests.get`**—**zero** retrieval or prompt code in `ui/`.

#### Results

**Qualitative:** Sidebar overrides + explain toggle; **local-only** `st.session_state` history (not backend memory).

#### Tradeoffs / limitations

No auth, no persistence, no streaming; history is **presentational** only.

#### Key takeaway

- **Thin client** discipline preserved.  
- **`CHAT_API_BASE`** configures target API.

#### Artifacts / evidence

- `ui/chat_ui.py`, `ui/chat_helpers.py`, `tests/test_chat_helpers.py`, **`API_CONTRACT.md`**.

---

### 12.4 Phase 5.4 — Thin conversational layer (planned)

**One-line purpose:** Add a **conversation context builder + follow-up resolver** in front of the existing pipeline so follow-ups reuse **structured** prior state—**without** stateful retrieval, long-term memory, or LLM-inferred history.

#### Context / starting point

Phase **5.2** exposes stateless **`POST /query`**. Phase **5.3** UI sends one-shot requests; **`st.session_state`** history is **display-only** and is **not** sent to the server. There is no **normalized query rewrite** or merge of prior **`filters` / `query_family`** into a new user utterance.

#### Problem observed

Concrete gaps:

- Follow-ups like **“What about one-star reviews?”** or **“Only the negative ones”** require **prior topic + scope** to interpret.  
- **“Shorter”** / **“bullet points”** are **output-shape** refinements on the same evidence or topic.  
- **“Why?”** / **“which chunks support that?”** need the **last answer + last evidence ids**, not a brand-new retrieval from the pronoun alone.  
- **Aspect shifts** (“buyer risks now”, “value complaints”) are still **query_family-shaped** if the **subject** of the session is carried forward.

Without a resolver, the model either **guesses** from a single ambiguous string or the user must **paste** the full question every time.

#### Diagnosis

**Evidence:** UX friction in 5.3 multi-turn demos; architecturally, the fix belongs in a **layer above** `QueryParser` / `Retriever`, not inside hybrid scoring or cross-encoder weights. Root cause: **missing explicit “resolved query + merged filters” artifact** per turn, not missing embedding quality.

#### Design (target architecture)

**Separation of concerns:** conversation logic **must not** be mixed into retrieval internals or family templates as hidden globals. Target flow:

1. **User message** →  
2. **`ConversationState`** (minimal turn history + **last resolved `RetrievalRequest` fields** and last answer metadata) →  
3. **`FollowupResolver`** (heuristic **follow-up detection** + **rule-based** merge / rewrite) →  
4. **Existing pipeline** — `QueryParser` / `RAGPipeline.answer()` on the **resolved** natural-language query (and merged filters where applicable) →  
5. **Response** → append **minimal turn summary** for the next iteration.

**First implementation stance:** **UI- or API-payload–held state**; **each backend `POST /query` remains logically stateless** (full context passed in or reconstructed from `conversation_id` **only if** you later add server-side session store—**not** required for v1).

#### Changes made

None shipped yet.

#### Planned components (names indicative)

| Module | Responsibility |
|--------|----------------|
| **`src/conversation_state.py`** | Dataclass(es): turn list, **last resolved query text**, last **`query_family`**, last **`filters`**, last **answer** (or summary), last **chunk_ids**, last **`explain`** flag. |
| **`src/followup_resolver.py`** | **Detect** follow-up (short text, pronouns, cue phrases like “what about”, “only”, “more briefly”, “why”); **merge** with `ConversationState`; **emit** `resolved_query: str` and optional **filter deltas**; unit-testable without Streamlit. |
| **`src/api.py`** | Optional request fields, e.g. **`conversation_context`** JSON and/or **`conversation_id`** if server session is added later. |
| **`ui/chat_ui.py`** | Populate context from `st.session_state`, display **resolved query** in an expander when explain/debug is on. |

#### Follow-up classes (minimal v1 scope)

1. **Scope refinement** — rating / sentiment / product scoping (“one-star”, “negative only”).  
2. **Output refinement** — length / format (“shorter”, “bullets”, “top 3”).  
3. **Ask-for-explanation** — tie to last turn’s **explanation** payload or re-run with `explain=true` and pinned scope.  
4. **Aspect shift** — same session topic, different **`query_family`**-shaped intent (value vs buyer risk vs symptoms).

#### Precedence when signals combine (implementation contract)

If one utterance matches multiple classes (e.g. scope + output), apply **deterministic** ordering so behavior does not flap:

1. **Reset / cold-start** conditions (explicit user reset, stale session policy, or “new topic” guard — see **PRODUCT_ROADMAP.md**) — evaluated **first**; if fired, **clear** prior carry-forward.  
2. **Scope refinement** — merge **filters** / topic scope into `resolved_query_text` **before** other merges.  
3. **Aspect shift** — adjust `query_family` / template intent against the **post-scope** state.  
4. **Ask-for-explanation** — prefer **reuse last evidence + answer** when scope **unchanged**; if scope changed in (2), **run retrieval** on new scope then attach explanation.  
5. **Output refinement** — apply **last** (prompt / template “shape” hints), so “shorter” does not override one-star filter intent.

**Non-goal (scope discipline):** Phase **5.4** does **not** solve **long-horizon** memory, **cross-session user profiles**, or **autonomous multi-step tool-planning** chat—that stays out of this phase (see Phase **6**).

#### When **not** to reuse prior context

Mirror product spec: **explicit reset**; **clearly new topic** (no follow-up cues + different entities / low overlap heuristics); optional **idle timeout**; **semantic drift** (long query, low similarity to last resolved text). If any trigger → **empty** conversation payload for merge purposes (still allow the user query to run as a standalone question).

#### Bounded turn summary (per-turn record)

Do **not** store narrative “what we talked about” prose as the canonical state. Store a **fixed schema** (see **PRODUCT_ROADMAP.md** — *Minimal turn summary*): at minimum **`user_query_raw`**, **`resolved_query_text`**, **`query_family`**, **`filters`**, **`answer_summary`** (length-capped or hash-stable), **`evidence_chunk_ids`**, **`explain_used`**.

#### API additivity

Shipped **`POST /query`** must remain valid **without** new fields. **`conversation_context`** optional; response fields for resolution (`resolved_query_text`, `followup_resolution_applied`, etc.) **optional** for backward-compatible clients.

#### Response transparency (trust)

When resolution runs, expose in **`metadata`** and/or **`explanation`**-adjacent fields (exact keys TBD in **API_CONTRACT.md** when shipped):

- **Original** user text for the turn.  
- **`resolved_query_text`** (what the pipeline actually ran).  
- **Flag:** follow-up resolver applied.  
- **Which prior fields** were reused (`filters`, `query_family`, chunk ids, etc.).

#### Why this design should work

Rule-based merge keeps behavior **auditable**; the parser and retriever still see a **single** coherent query per request; **`query_family`** and **filters** remain the same conceptual objects as today—only the **string** feeding the parser may be **synthetic** after merge.

#### Results

N/A until implemented.

#### Tradeoffs / limitations

- Heuristics will **miss** or **mis-merge** some follow-ups; **no** promise of human-level coreference without later ML.  
- **No** long transcript summarization into prompts (avoids context explosion and silent drift).  
- **No** DB persistence in the first slice—session loss on refresh unless client replays state.

#### Key takeaway

- **5.4** = **controlled context carryover**, not “magical memory.”  
- **Resolver before pipeline** preserves Phase **4.5** retrieval / prompt investments.  
- Ships **before** Phase **6.1** tools so the product path is: answer → explain → **short converse** → **analyze**.

#### Success criteria (manual, post-ship)

One session should handle without full re-prompting: **one-star follow-up**, **buyer-risk pivot**, **shorter answer**, **evidence / why probe** (aligned with **PRODUCT_ROADMAP.md** acceptance list).

#### Artifacts / evidence

- **PRODUCT_ROADMAP.md** — Phase 5.4 product spec.  
- **API_CONTRACT.md** — *Planned: Phase 5.4* request/response notes.  
- Future: `tests/test_followup_resolver.py`, OpenAPI diff on **`POST /query`**.

---

## 13. Phase 6 — Analytics & ML tooling (planned)

**One-line purpose:** Add **tool-backed** numeric/statistical work (forecasting, diagnostics, experiments) composed with retrieval—**not** LLM-only arithmetic.

### Context / starting point

Phases 1–5: strong **grounded retrieval + summarization + explanation** path; no first-class **ML tool** router.

### Problem observed

Predictive / optimization questions cannot be answered honestly inside a single prompt without external computation.

### Diagnosis

LLMs should **route** and **explain tool outputs**, not fabricate numbers (**ML_CAPABILITIES_ROADMAP.md** — tool abstraction).

### Changes made

Roadmap and architecture sketch only (intent router → tool selection → retrieval **or** analytics path → LLM explanation).

### Why these changes should work (design intent)

Typed **tool contracts** (input/output schema, execution engine, explanation contract) prevent ad-hoc “stats in prompt” chaos.

### Results

N/A (planned).

### Tradeoffs / limitations

Scope creep risk if tools are not bounded; evaluation must extend to **tool** outputs.

### Key takeaway

- **Phase 6** is **computation composition**, not replacement of RAG.  
- See **ML_CAPABILITIES_ROADMAP.md**, **SYSTEM_OVERVIEW.md** (layers table).

### Artifacts / evidence

- **ML_CAPABILITIES_ROADMAP.md**, **PRODUCT_ROADMAP.md** (Phase 6 references).

---

### Layer separation (closing note)

- **Phases 1–4.5** — Core **intelligence quality**: parsing, retrieval, hybrid/rerank, selective policy, prompts, routing, manual eval.  
- **Phase 5** — **Productization** and **explainability** (API, UI, planned conversation) on top of that core.  
- **Phase 6** — **ML / analytics extensions** (tool-backed computation) composed with retrieval.

---

## Changelog pointer

Append-only operational notes: **DAILY_LOG.md**.
