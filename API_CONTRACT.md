# HTTP API contract (Phase 5.2) + Chat UI (Phase 5.3)

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

Minimal **Streamlit** app: **`ui/chat_ui.py`**. It only uses **`requests`** against **`GET /health`** and **`POST /query`** — no retrieval, rerank, prompt, or explanation logic in the UI layer.

**Run (API must already be listening):**

```bash
# Terminal 1 — API
pip install -r requirements.txt
PYTHONPATH=. uvicorn src.api:app --host 127.0.0.1 --port 8000

# Terminal 2 — UI
PYTHONPATH=. streamlit run ui/chat_ui.py
```

Or UI only: `PYTHONPATH=. python scripts/run_chat_ui.py` (defaults to `http://127.0.0.1:8501`).

**API base URL for the UI:** environment variable **`CHAT_API_BASE`** (default `http://127.0.0.1:8000`) or the sidebar field in the app.

**Session strip:** Any “history” in the UI is **Streamlit `st.session_state` only** (browser session, not sent to the server). Each **`POST /query`** is still **stateless** on the backend.

## Planned: Phase 5.4 (thin conversation layer)

**Not shipped yet.** Goal: optional **conversation context** on the client and/or API request body so follow-ups (“What about one-star?”, “Shorter”, “Why?”) can be **resolved into a single normalized query + filters** before the existing **`POST /query`** pipeline runs—**without** making retrieval stateful, **without** long-term memory or DB chat logs, and **without** stuffing raw multi-turn history into the LLM prompt.

**Explicit non-goal:** Phase **5.4** does **not** target **long-horizon memory**, **user-profile** persistence, or **autonomous tool-planning / multi-agent** dialogue—only **short-term, structured** carryover with auditable resolution.

**Backward compatibility (must hold when implemented):**

- Requests **without** conversation fields behave **identically** to Phase **5.2** today.  
- **`conversation_context`** / `conversation_id` (if any) are **optional** request keys.  
- **`resolved_query_text`**, **`followup_resolution_applied`**, and related fields on the response are **optional** so existing clients keep working.

**Likely request additions (additive):** e.g. `conversation_id` and/or a **`conversation_context`** object built from a **bounded turn summary** (see **PRODUCT_ROADMAP.md** — fixed fields: `user_query_raw`, `resolved_query_text`, `query_family`, `filters`, capped `answer_summary`, `evidence_chunk_ids`, `explain_used`). **Likely response additions:** e.g. `original_user_text`, **`resolved_query_text`**, **`followup_resolution_applied`**, which prior fields were reused.

**Precedence / anti-carry-forward:** When multiple follow-up cues appear in one message, **scope refinement precedes output refinement** (and other ordering rules); prior context is **not** reused on explicit reset, cold topic, idle timeout, or drift heuristics—see **`PRODUCT_ROADMAP.md`** and **`SYSTEM_EVOLUTION.md`** §12.4.

**Implementation sketch:** `src/conversation_state.py` (turn state), `src/followup_resolver.py` (heuristic follow-up detection + rule-based merge / rewrite) → then today’s **`RAGPipeline.answer()`** unchanged in spirit; UI stores turns and passes context into the resolver path.

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
    "selective_rerank_effective": true
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

**Additive policy:** New **`metadata`** keys may appear in future releases (e.g. timing, strategy ids). Clients should **ignore unknown keys** and must not assume every key is always present except the five always-emitted fields above (the sixth, `answer_trace_path`, is conditional). **Renaming or removing** a documented stable key requires an explicit **API version** bump or a documented breaking-change release.

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
