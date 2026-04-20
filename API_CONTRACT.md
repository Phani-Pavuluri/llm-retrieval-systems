# HTTP API contract (Phase 5.2) + Chat UI (Phase 5.3) + optional query planner (Phase 5.5)

Minimal **backend-only** JSON API over `RAGPipeline.answer()` (Phase 5.2). No authentication, persistence, sessions, streaming, or heavy frontend in the API itself.

Phase **5.3** adds an optional **Streamlit** thin client that only calls this API — see **Chat UI (Phase 5.3)** below.

## Run locally

From the repository root (with dependencies installed):

```bash
pip install -r requirements.txt
PYTHONPATH=. uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Or:

```bash
PYTHONPATH=. python scripts/run_api.py
```

- Interactive OpenAPI / Swagger UI: `http://127.0.0.1:8000/docs`  
- Alternative docs: `http://127.0.0.1:8000/redoc`

## Chat UI (Phase 5.3)

Minimal **Streamlit** app: **`ui/chat_ui.py`**. It only uses **`requests`** against **`GET /health`** and **`POST /query`** — no retrieval, rerank, prompt, or explanation logic in the UI layer. The sidebar includes an optional **LLM query planner (retrieval)** checkbox; when enabled, requests include **`query_planner`: true** (see Phase 5.5).

**Run (API must already be listening):**

```bash
# Terminal 1 — API
pip install -r requirements.txt
PYTHONPATH=. uvicorn src.api:app --host 127.0.0.1 --port 8000

# Terminal 2 — UI
PYTHONPATH=. streamlit run ui/chat_ui.py
```

Or UI launcher (same app; pins Streamlit defaults):

```bash
PYTHONPATH=. python scripts/run_chat_ui.py
```

(Streamlit defaults to `http://127.0.0.1:8501` in `scripts/run_chat_ui.py`.)

**API base URL for the UI:** environment variable **`CHAT_API_BASE`** (default `http://127.0.0.1:8000`) or the sidebar field in the app.

**Session strip:** Any “history” in the UI is **Streamlit `st.session_state` only** (browser session, not sent to the server). Each **`POST /query`** is still **stateless** on the backend.

## Phase 5.4 (thin conversation layer)

**Shipped (additive).** The client may send a bounded **`conversation_context`** payload (recent turns only). The API runs a **rule-based resolver** first to produce a **`resolved_query`** string (and optional filter / family / explain / formatting hints) before calling the existing **`RAGPipeline.answer()`** path.

**Hard constraints (by design):**

- Backend remains **stateless per request** (no server-side chat DB).
- No long raw transcript stuffing: only a **small structured** context object.
- Resolver is **heuristic / rule-first**; retrieval internals stay unchanged.

**Backward compatibility:** requests **without** `conversation_context` behave like Phase **5.2**.

## Phase 5.5 — LLM query planner (retrieval, optional)

**Shipped (additive).** Before retrieval, the pipeline may run a **small structured LLM call** that proposes optional adjustments to the parsed **`RetrievalRequest`**: paraphrased **`query_text`** for embedding / lexical overlap, optional **`review_rating`** metadata filter, and optional **`query_family`** (allowlisted only). The model output is **JSON-parsed and validated**; invalid or empty plans are ignored.

**When it runs**

- Controlled per request by **`query_planner`** (boolean or `null`). If omitted, the server uses **`config.QUERY_PLANNER_DEFAULT`** (currently **`false`** so behavior matches pre-planner deployments unless callers opt in).
- The planner is **skipped** when: it is disabled; the rule-based parser already extracted a **`review_rating`** filter for the same utterance (avoids a redundant LLM call); or **`filter_overrides` / `reset_filters`** were applied (typical follow-up path) **and** the merged request already has **`review_rating`** in **`filters`**—so explicit resolver scope is not overridden.

**Costs and tradeoffs:** one extra LLM generation per enabled request (latency + tokens); retrieval pool can change vs rules-only parsing—enable explicitly until tuned for your traffic.

## Endpoints

### `GET /health`

Returns:

```json
{ "status": "ok" }
```

### `POST /query`

**Body (JSON)**

| Field | Required | Type | Notes |
|-------|----------|------|--------|
| `query` | yes | string | Non-empty. |
| `explain` | no | boolean | Default `false`. When `true`, runs the pipeline with `explain=True` and returns the canonical explanation object. |
| `llm_backend` | no | string or null | `ollama` \| `openai`; omit to use `config.LLM_BACKEND`. |
| `llm_model` | no | string or null | Omit to use backend default from config. |
| `k` | no | integer ≥ 1 or null | Top-k; omit to use `config.TOP_K`. |
| `selective_rerank` | no | boolean or null | Omit to use `config.RERANK_SELECTIVE`. |
| `rerank_model` | no | string or null | Cross-encoder id when rerank runs. |
| `rerank_top_n` | no | integer ≥ 1 or null | Passed through to `RetrievalRequest`. |
| `conversation_context` | no | object or null | Optional bounded turns for follow-up resolution (see `src/conversation_state.py`). |
| `query_planner` | no | boolean or null | If `true`, allow the optional LLM retrieval planner (`src/query_planner.py`). If `null`/omitted, use `config.QUERY_PLANNER_DEFAULT`. |

**Example**

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What complaints appear in reviews?","explain":true}'
```

## Example responses (full JSON)

### `200 OK` — `explain: false`

Illustrative values; your `answer` and `metadata` will differ per query and environment.

```json
{
  "answer": "Reviewers mention packaging issues and occasional shipping damage; specifics vary by listing.",
  "explanation": null,
  "metadata": {
    "query_family": "abstract_complaint_summary",
    "prompt_template_id": "family_abstract_complaint_summary",
    "llm_backend": "ollama",
    "llm_model": "llama3",
    "selective_rerank_effective": true,
    "user_query": "What complaints appear in reviews?",
    "resolved_query": "What complaints appear in reviews?",
    "is_followup": false,
    "followup_type": "none",
    "reused_fields": [],
    "filters_applied": {},
    "chunk_ids_used": [],
    "retrieval_query_text": "What complaints appear in reviews?",
    "query_plan": null,
    "explain_used": false
  }
}
```

### `200 OK` — `explain: true`

`explanation` uses the canonical keys only (`evidence`, `reasoning_summary`, `confidence`). Evidence rows reflect whatever the pipeline placed in the LLM context (scores depend on retrieval mode).

```json
{
  "answer": "Several excerpts cite a rash or skin irritation after use; others do not mention reactions.",
  "explanation": {
    "evidence": [
      {
        "chunk_id": "B00EXAMPLE_3_0",
        "source_id": "B00EXAMPLE",
        "chunk_text": "Developed a rash within two days of use.",
        "rank_position": 1,
        "score": 0.82,
        "semantic_score": 0.79,
        "keyword_score": 0.41
      }
    ],
    "reasoning_summary": {
      "retrieval_mode": "vector",
      "rerank_applied": false,
      "rerank_requested": true,
      "query_family": "exact_issue_lookup",
      "strategy_reason": "",
      "filters_applied": {},
      "rating_scope": null,
      "prompt_template_id": "family_exact_issue_lookup",
      "summary_line": "Reranking was skipped because this query was classified as exact-issue or extraction-style retrieval (family policy). Vector similarity ordering was used (no hybrid fusion on this run)."
    },
    "confidence": {
      "confidence_label": "medium",
      "confidence_score": 0.55,
      "confidence_reasons": [
        "Only one excerpt in context.",
        "Selective policy or config left rerank off for this query (see reasoning_summary.rerank_applied).",
        "Heuristic score=0.55 (not a calibrated probability)."
      ]
    }
  },
  "metadata": {
    "query_family": "exact_issue_lookup",
    "prompt_template_id": "family_exact_issue_lookup",
    "llm_backend": "ollama",
    "llm_model": "llama3",
    "selective_rerank_effective": true
  }
}
```

### `400` — validation failure

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Request validation failed.",
    "details": [
      {
        "type": "string_too_short",
        "loc": ["body", "query"],
        "msg": "String should have at least 1 character",
        "input": "",
        "ctx": { "min_length": 1 }
      }
    ]
  }
}
```

### `503` — LLM backend unreachable (example)

`details` is usually omitted for this path.

```json
{
  "error": {
    "code": "backend_unavailable",
    "message": "Ollama server not reachable at http://localhost:11434/api/generate. Is Ollama running? (...)"
  }
}
```

## Success response (`200`)

Top-level JSON:

| Field | Type | Notes |
|-------|------|--------|
| `answer` | string | Final model output. |
| `explanation` | object or `null` | `null` when `explain` is `false`. When `true`, matches pipeline output: `evidence`, `reasoning_summary`, `confidence` only (no duplicate `answer` inside `explanation`). |
| `metadata` | object | Lightweight fields only (no raw retrieval traces by default). |

**`metadata` keys (see also *Metadata stability* below)**

- `query_family` — parser / inferred family  
- `prompt_template_id`  
- `llm_backend`, `llm_model` — effective generation backend/model for this call  
- `selective_rerank_effective` — boolean selective policy value used for this request  
- `retrieval_query_text` — string passed into retrieval after parse (and optional planner); may differ from `resolved_query` when the planner paraphrases.  
- `query_plan` — structured planner result (`null` or a dict with fields such as `applied`, `source`, `notes`, `filters_patch`); see `src/query_planner.QueryPlanResult`.  
- `answer_trace_path` — present only when answer tracing wrote a path (see *Answer tracing*)

### Explainability notes

- **`evidence`** lists chunks **provided to the model** in the prompt (context window). It does **not** prove which excerpts the model relied on most (no attribution).  
- **`reasoning_summary`** and **`confidence`** reflect **pipeline / template / heuristic** state — not hidden chain-of-thought from the LLM.  
- **`confidence`** is a **heuristic** (label + score + reasons). It is **not** a calibrated probability and must not be used as a standalone risk score. It does **not** detect **cross-chunk contradiction** today. See **SYSTEM_EVOLUTION.md** (Phase 5.1) for the full rule list and limitations.

## Error response (non-2xx)

Consistent wrapper:

```json
{
  "error": {
    "code": "<string>",
    "message": "<string>",
    "details": <optional>
  }
}
```

| HTTP | `error.code` | When |
|------|----------------|------|
| `400` | `invalid_request` | Validation failure (including empty `query`), unsupported `llm_backend`, other bad input. |
| `503` | `backend_unavailable` | Ollama unreachable / HTTP error, OpenAI missing key, runtime errors from the LLM HTTP path. |
| `500` | `pipeline_error` | I/O or load failures in the retrieval stack (e.g. missing index resources). |
| `500` | `internal_error` | Unexpected server error (no stack trace in the body). |

Server-side logging should capture tracebacks for `internal_error` / `pipeline_error`.

**Error code granularity (later):** Today several distinct failures (Ollama down, OpenAI misconfig, transient LLM HTTP errors) may share **`backend_unavailable`**. That is intentional for v1. A future revision may introduce finer codes (e.g. `llm_backend_unavailable`, `reranker_unavailable`, `index_unavailable`) **without** removing the top-level `error` wrapper — new codes would be additive behind an optional API version if needed.

## Metadata stability

These **`metadata` keys are stable** for Phase 5.2 consumers — treat them as part of the public contract:

| Key | Stability |
|-----|-----------|
| `query_family` | Stable |
| `prompt_template_id` | Stable |
| `llm_backend` | Stable |
| `llm_model` | Stable |
| `selective_rerank_effective` | Stable |
| `answer_trace_path` | Stable **when present** (optional key; see *Answer tracing*) |
| `retrieval_query_text` | Stable (string or `null`) |
| `query_plan` | Stable (object or `null`; internal fields are diagnostic and may evolve additively) |

**Additive policy:** New **`metadata`** keys may appear in future releases (e.g. timing, strategy ids). Clients should **ignore unknown keys**. Keys listed in the **Stable** table above are contractually stable; some rows are **conditional** (e.g. `answer_trace_path` only when tracing wrote a file). **Renaming or removing** a documented stable key requires an explicit **API version** bump or a documented breaking-change release.

**Phase 5.2+ conversation metadata:** Current responses also emit **`user_query`**, **`resolved_query`**, **`is_followup`**, **`followup_type`**, **`reused_fields`**, **`filters_applied`**, **`chunk_ids_used`**, and **`explain_used`** (see **`src/api.py`**). Omitting a key from the stability table means it is still **additive / wire-stable** in practice until promoted into the table explicitly.

Top-level success fields **`answer`**, **`explanation`**, **`metadata`** are stable; the internal shape of **`explanation`** when non-null is defined in **SYSTEM_OVERVIEW.md** / **SYSTEM_EVOLUTION.md** (evidence / reasoning_summary / confidence only).

## User-facing vs developer / debug (planned)

**Current behavior:** The default `POST /query` response is **user-safe**: `answer`, optional `explanation`, and **lightweight `metadata`** — no raw prompts, full `rerank_decision` dumps, or retrieval dataframes.

**Likely before Phase 5.3 (chat):** introduce an explicit split, for example:

- **Default** — unchanged user-safe envelope.  
- **Opt-in debug** — separate flag or header (e.g. `X-Debug: 1` for trusted callers) that adds a sibling object such as `debug` or `trace` with internals, **never** mixed into `metadata` by default.

Until that exists, UIs should couple only to the documented stable fields and treat anything else as non-contractual.

## Answer tracing

**Today:** Whether an answer append-only JSONL trace is written is controlled **server-side** via `config.ANSWER_TRACE_ENABLED` / `ANSWER_TRACE_DIR` (and optional `answer_trace_out` in non-API scripts). **Callers cannot turn tracing on or off via `POST /query`** in Phase 5.2.

When tracing is enabled on the server, **`metadata.answer_trace_path`** may be set to the path of the written trace line for that request.

**Future:** A request field (e.g. reserved name **`answer_trace`**: boolean) or a scoped header could allow **trusted** API callers to request tracing per call; that would be documented here and in OpenAPI when implemented.

## Schema stability

Success and error **envelopes** (`answer` / `explanation` / `metadata` and `error.code` / `error.message` / optional `error.details`) are the **Phase 5.2 contract**. Evolution should be **additive** (new optional keys, new optional `metadata` fields, new optional `error.details` shapes) unless you ship an explicit **versioned** API or breaking-change release.

## Relationship to core docs

- Pipeline behavior and metrics: **SYSTEM_EVOLUTION.md**, **SYSTEM_OVERVIEW.md**  
- Product scope (what 5.2 deliberately excludes): **PRODUCT_ROADMAP.md**
