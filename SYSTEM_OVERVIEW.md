# System overview (technical, minimal)

## Objective

Answer user questions **only from retrieved review chunks**, with observable **retrieval**, **rerank**, **generation**, and (optionally) **structured explanation** — suitable for both **direct backend use** and future **API / chat** layers.

## Layers (conceptual)

| Layer | Role | Status |
|-------|------|--------|
| **Product / access** | HTTP API, chat UI, thin conversation — thin clients over the backend | **5.1** explainability; **5.2** API; **5.3** Streamlit (**`API_CONTRACT.md`**); **5.4** thin follow-up layer (**shipped** — **`SYSTEM_EVOLUTION.md`** §12.4); **5.5** optional LLM **retrieval query planner** (**`src/query_planner.py`**, **`API_CONTRACT.md`** — *Phase 5.5*) |
| **Core pipeline** | Parse → route → retrieve / rerank → prompt → LLM → optional **explanation** | **Implemented** (Phases 1–4.5 + **5.1**) |
| **Future analytics** | Intent router → ML / statistical **tools** (not LLM-only math) | **Phase 6** — **planned** (see **`ML_CAPABILITIES_ROADMAP.md`**) |

## Components (core pipeline)

| Step | Module(s) | Role |
|------|-----------|------|
| Parse | `query_parser.py` | `task_type`, metadata `filters`, `infer_query_family()` |
| Route (policy) | `rerank_policy.py` | `query_family`; selective `use_rerank` |
| Retrieve | `retriever.py`, `retrieval_strategy.py` | FAISS + oversample; strategy on `RetrievalRequest` |
| Hybrid (optional) | `hybrid_scoring.py` | Fuse semantic + lexical on candidates |
| Filter | `metadata_filters.py` | Post-filter by `filters` |
| Rerank (optional) | `reranker.py`, `retrieval_with_rerank.py` | Cross-encoder reorder |
| Answer | `prompt_builder.py`, `llm.py` | Task/family prompt → LLM |
| Explain (optional) | `explanation_builder.py` | **`explain=True`**: chunks **provided to the LLM** + templated reasoning + heuristic confidence |
| Orchestrate | `rag_pipeline.py` | `answer()`; traces |

## Data flow (text)

**Current end-to-end (core + explanation):**

```
User (string or API caller)
    → [optional: HTTP POST /query — Phase 5.2; Streamlit UI — Phase 5.3]
    → [optional: follow-up resolver — Phase 5.4 conversation_context]
    → QueryParser → RetrievalRequest
    → [optional: LLM query planner — Phase 5.5 when query_planner enabled]
    → apply_selective_rerank_policy
    → retrieval_strategy → Retriever (FAISS, filter, optional hybrid)
    → [optional] cross-encoder rerank
    → prompt_builder → LLM → answer text
    → [optional] explanation_builder (evidence in context + reasoning_summary + confidence)
    → Response (top-level answer + optional explanation object)
```

**Forward-looking (Phase 6):** a **tool layer** will let the access tier route **analytical / predictive** questions to **ML and statistical tools**, with the LLM **explaining tool outputs** — not replacing retrieval; see **`ML_CAPABILITIES_ROADMAP.md`**.

## Evaluation artifacts (not runtime deps)

- `eval/answer_eval_labeled.json` — manual scores; **`query_family`** must match parser output for fair aggregates.
- `artifacts/answer_traces/` — eval runs and score summaries.

## Evaluation philosophy (end-to-end)

The stack is measured as a **pipeline**, not a pile of disconnected experiments.

| Layer | What we measure | Typical signals |
|-------|------------------|-----------------|
| **Retrieval quality** | Did the right evidence enter the pool? | P@k, recall@k, MRR (offline / labeled sets); trace counts (`post_filter_count`, underfill). |
| **Rerank impact** | Did second-stage ranking change outcomes vs baseline? | Delta in rank metrics or head chunk overlap vs vector-only; `rerank_decision` in traces. |
| **Answer quality** | Did the model use evidence correctly for the question? | **grounded**, **correct**, **complete** (manual 1–3 on held-out queries). |
| **Failure taxonomy** | What *kind* of failure was it? | Canonical buckets (`wrong_scope`, `hallucination`, …) for regression tracking. |

**Key principle:** A change is only “done” when it **moves answer-level metrics** (or clearly targeted failure buckets) — **not** retrieval-only charts alone. Retrieval and rerank metrics are **necessary** but **insufficient** for product quality.

## Prompt layer responsibilities (conceptual split)

Templates today bundle three roles; keeping them conceptually separate helps future refactors (e.g. when ML tools add new answer shapes):

| Role | Examples |
|------|-----------|
| **Reasoning constraints** | Grounding, causality, negation, scope (what may be inferred). |
| **Task framing** | Summary vs lookup vs extraction; complaint vs value vs buyer-risk. |
| **Output structure** | Bullets, theme headings, chunk cite format. |

**Reasoning ≠ formatting** — both live in `prompt_builder` today, but only **reasoning + framing** gate truth; **format** is presentation.

## Phase 5.1 — Explain response envelope (canonical for API)

When `explain=True`, the in-process / future HTTP response should treat this shape as **stable**:

| Field | Meaning |
|-------|---------|
| **`answer`** | Final model string (top-level on the response object — authoritative). |
| **`explanation.evidence`** | Chunks **provided to the model** in the prompt context (order = LLM input). **Not** “which chunks the model relied on most” (no attribution). |
| **`explanation.reasoning_summary`** | System-side retrieval / rerank / filter / template signals (templated, not CoT). |
| **`explanation.confidence`** | Heuristic label + score + human-readable `confidence_reasons` (not calibrated probability). |

`RAGPipeline.answer()` already returns **`answer`** at the top level; **`explanation`** contains only the three keys above (no duplicate `answer` inside `explanation`). See **SYSTEM_EVOLUTION.md** → Phase 5.1 for **what drives confidence** and known gaps (e.g. cross-chunk **conflict** is not scored yet).

**Schema stability (pre-API):** Treat this envelope as the **current canonical** success shape for Phase 5.2; **future fields should be additive** (new optional keys, nested objects) rather than **renaming or removing** existing keys without a version bump.

HTTP **error** wrapper, empty-evidence behavior, confidence disclaimers for clients, and **debug vs user-facing** JSON: see **`PRODUCT_ROADMAP.md`** (Phase 5.2 intent) and the shipped **`API_CONTRACT.md`** (runbook, `POST /query`, `/health`).

## Design constraints (current)

- **Selective rerank** — family + metadata aware.
- **Rule-based parser** — predictable routing (incl. value / buyer-risk / symptom families).
- **Grounding** — chunk citations; rating scope in prompts mirrors `filters`.
- **Explanation** — **no extra LLM** for rationale; template + diagnostics only.

See **SYSTEM_EVOLUTION.md** for history, metrics, and **Phase 5 / 6** roadmap alignment. **Quick runs:** **`RUN_COMMANDS.md`** (copy-paste) · **`RUN_GUIDE.md`** (what / why / when) · optional **`Makefile`** (`make help`).
