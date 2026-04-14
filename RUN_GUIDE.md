# Run guide (what, why, when)

Use **`RUN_COMMANDS.md`** when you already know what to run. Use **this file** when you need context, outputs, or “when should I run this?”

**Convention:** do not mix long explanations into `RUN_COMMANDS.md`—that file stays copy-paste only.

**Rule for new scripts:** (1) add the command to **`RUN_COMMANDS.md`**, (2) add a matching section here with *What it does / Why it matters / When to use / Outputs*.

## Repository root

**All commands and `Makefile` targets assume your shell’s current directory is the repository root** (the folder that contains `src/`, `scripts/`, `Makefile`, and `requirements.txt`). If you `cd` elsewhere, `PYTHONPATH=.` and relative paths to `scripts/` will not behave as documented.

---

## Environment & setup (once per machine / venv)

1. **Install dependencies:** `pip install -r requirements.txt` (ideally in a virtualenv).  
2. **Data / index:** build chunks and vector index when setting up from scratch (see **`scripts/build_chunks.py`**, **`scripts/build_index.py`** — not repeated in every workflow below).  
3. **LLM at answer time:** if `config.LLM_BACKEND` is **`ollama`**, run Ollama locally with the configured model; if **`openai`**, set **`OPENAI_API_KEY`** (e.g. via `.env` loaded by `python-dotenv`). Without this, **`run_answer_eval.py`**, **`run_query.py`**, and **`POST /query`** will fail at generation.  
4. **API + UI:** start **`make api`** before **`make ui`** (or point the UI at an already-running server via **`CHAT_API_BASE`**).

---

## Typical workflows

Short end-to-end sequences—**no new commands** beyond **`RUN_COMMANDS.md`** / **`make help`**.

### 1. Start the product locally

- **`make api`** — FastAPI on **`http://127.0.0.1:8000`**.  
- **`make ui`** — Streamlit chat against that API (default base URL).

Use this for demos, manual Q&A, and inspecting explain payloads in the browser.

### 2. Run quality checks after backend (retrieval / rerank) changes

- **`make test`** — unit tests across `tests/`.  
- **`make eval`** — labeled retrieval run → trace inputs.  
- **`make rerank-analyze`** — read traces, summarize P@k / MRR-style signals and rerank impact.

Use this when you touched retrieval, hybrid, rerank, or **`rerank_policy`**, and want confidence before regenerating full answers.

### 3. Re-run the answer-quality loop

- **`make answer-eval`** — full pipeline answers for all labeled queries → **`artifacts/answer_traces/`**.  
- **Edit the label file** — update scores / failure buckets in **`eval/answer_eval_labeled.json`** for the new run (manual step; there is no `make` target for labeling).  
- **`make answer-summary`** — aggregate means, by-**`query_family`** bars, and bucket counts into summary artifacts.

Use this after parser, routing, or prompt changes where the **12-query** manual rubric is the bar (`SYSTEM_EVOLUTION.md`).

---

## API

**Command**

```bash
PYTHONPATH=. uvicorn src.api:app --host 127.0.0.1 --port 8000
```

**What it does**  
Starts the FastAPI app in **`src/api`**: **`GET /health`**, **`POST /query`** over the existing `RAGPipeline` (Phase 5.2).

**Why it matters**  
This is the supported entry point for:

- the Streamlit chat UI (HTTP client),
- programmatic / curl / integration callers,
- anything that should not import `src/` in-process.

**When to use**  
Whenever you need the HTTP contract (`API_CONTRACT.md`), local demos against **`http://127.0.0.1:8000`**, or to run the UI against a real server.

**See also**  
`PYTHONPATH=. python scripts/run_api.py` (same server, via uvicorn module path).

---

## Chat UI

**Command**

```bash
PYTHONPATH=. streamlit run ui/chat_ui.py
```

**What it does**  
Starts the Phase 5.3 Streamlit app (**`ui/chat_ui.py`**). It only calls **`GET /health`** and **`POST /query`**—no retrieval or prompt logic in the UI layer.

**Why it matters**  
Gives you:

- answer display,
- optional explain mode (evidence, reasoning summary, confidence),
- lightweight metadata and clear API errors,

without touching the core pipeline code paths beyond HTTP.

**When to use**  
Manual testing, demos, debugging answers and explain payloads. Set **`CHAT_API_BASE`** (or the sidebar URL) if the API is not on the default host/port.

---

## Tests

**Command**

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

**What it does**  
Discovers and runs all tests under **`tests/`** (parser, prompts, rerank policy, API, chat helpers, explanation builder, etc.).

**Why it matters**  
Catches regressions in:

- pipeline and policy logic,
- API request/validation and error mapping,
- small pure helpers used by the UI.

**When to use**  
After code changes, before commits, or when CI is not available locally.

---

## Retrieval evaluation

**Command**

```bash
PYTHONPATH=. python scripts/eval_labeled_retrieval.py
```

**What it does**  
Runs retrieval for labeled queries (see script **`--help`** for flags such as trace path, **`k`**, rerank options). Writes trace / metric inputs used by downstream analysis.

**Why it matters**  
Measures retrieval behavior—vector vs hybrid, rerank on/off, selective policy effects—**before** answer generation. Complements answer-level eval, which is dominated by prompts and routing once retrieval is “good enough” on the held-out slice.

**When to use**  
When changing retrieval, hybrid weights, rerank models, or trace format—or before **`analyze_rerank_impact.py`** / **`validate_selective_rerank.py`** if traces are stale.

**Outputs**  
Typically under **`artifacts/retrieval_traces/`** (exact filenames depend on script flags; see script docstring).

---

## Rerank analysis

**Command**

```bash
PYTHONPATH=. python scripts/analyze_rerank_impact.py
```

**What it does**  
Reads retrieval trace artifacts and computes summary statistics (e.g. P@k, R@k, MRR-style signals and rerank impact summaries—see script output and **`SYSTEM_EVOLUTION.md`** for how this fits the eval story).

**Why it matters**  
Shows whether second-stage reranking is **helping, hurting, or neutral** on the traced set, separate from answer-quality labels.

**When to use**  
After a labeled retrieval run, when comparing rerank models or selective vs always-on behavior (often paired with **`validate_selective_rerank.py`**).

**Outputs**  
Console / text reports and paths printed by the script; inputs are existing **`artifacts/`** trace files.

---

## Selective rerank validation

**Command**

```bash
PYTHONPATH=. python scripts/validate_selective_rerank.py --run-eval
```

**What it does**  
Optionally runs **`eval_labeled_retrieval.py`** (when **`--run-eval`**), then builds comparison artifacts for **selective rerank vs always-on** style analysis. **`--skip-eval`** uses an existing trace file only—see script usage block.

**Why it matters**  
Validates that **`rerank_policy`** gating (family, metadata narrowness, confidence skip, etc.) behaves as intended and documents impact in **`artifacts/`**.

**When to use**  
After policy or **`infer_query_family`** changes, or when preparing evidence for **`SYSTEM_EVOLUTION.md`** rerank sections.

**Outputs**  
Under **`artifacts/`** (subdirs / filenames as configured in the script; **`--artifacts-dir`** overrideable).

---

## Answer evaluation (generation)

**Command**

```bash
PYTHONPATH=. python scripts/run_answer_eval.py
```

**What it does**  
Runs the **12** labeled items from **`eval/answer_eval_labeled.json`** through **`RAGPipeline.answer()`**, writing run rows (including runtime **`query_family`**) to JSON under **`artifacts/answer_traces/`**.

**Why it matters**  
Produces the raw generation outputs used for **grounded / correct / complete** scoring and failure buckets—**answer quality**, not retrieval-only metrics.

**When to use**  
After parser, routing, prompt, or pipeline behavior changes that should move manual eval scores.

**Outputs**  
e.g. **`artifacts/answer_traces/answer_eval_runs.json`** (see script for exact default path).

---

## Answer evaluation summary

**Command**

```bash
PYTHONPATH=. python scripts/summarize_answer_eval.py
```

**What it does**  
Joins labels in **`eval/answer_eval_labeled.json`** with run output and writes aggregated summaries (means overall and by **`query_family`**, failure bucket counts) to **`artifacts/answer_traces/`** as JSON and/or text.

**Why it matters**  
This is the primary **answer-level** scorecard used in **`SYSTEM_EVOLUTION.md`** (Phase 4.4 / 4.5 tables). **`query_family` on label rows must match runtime routing** after taxonomy changes.

**When to use**  
After every fresh **`run_answer_eval.py`** pass you intend to compare numerically.

**Outputs**  
Files matching **`answer_eval_score_summary*.json`** / **`.txt`** under **`artifacts/answer_traces/`**.

---

## Why this split works

| File | Role |
|------|------|
| **`RUN_COMMANDS.md`** | Speed—no prose, safe copy-paste. |
| **`RUN_GUIDE.md`** | Understanding—outputs, intent, when to run. |

---

## Makefile (optional)

From repo root, **`make help`** lists short targets that mirror the commands above (`api`, `ui`, `test`, etc.). Use **`make`** if you prefer not to retype **`PYTHONPATH=.`** invocations.
