# Product roadmap

## Goal

Define the **user-facing** evolution of the system **on top of** the core backend pipeline (parsing, retrieval, rerank, prompts, optional explanation).

---

## Phase 5.1 — Explainable answers (implemented / current)

- Expose **evidence provided to the model** (chunk ids, text, scores — context window, not attribution of reliance).  
- Expose **prompt / routing metadata** (template id, `query_family`, strategy, filters) via structured **`reasoning_summary`**.  
- Add **confidence** summary (heuristic label + reasons — not calibrated probability).  
- Improve **answer transparency** without chain-of-thought (`explain=True`, `scripts/run_query.py --explain`).  

---

## Phase 5.2 — API layer **(implemented)**

**Shipped contract & runbook:** **`API_CONTRACT.md`** · **`src/api.py`** · local dev: `PYTHONPATH=. uvicorn src.api:app --host 127.0.0.1 --port 8000` or `PYTHONPATH=. python scripts/run_api.py`.

### Minimal scope (intentionally small)

- **One** query endpoint (POST or GET with a typed body — pick one style and stick to it).  
- **One** boolean **`explain`** (maps to `RAGPipeline.answer(..., explain=…)`).  
- **Structured** JSON request / response aligned with the pipeline success envelope (see **SYSTEM_OVERVIEW.md** → *Phase 5.1 — Explain response envelope*).  
- Optional **backend / model** overrides on the request **only if** needed for ops; otherwise omit from v1.  

**Explicitly out of scope for 5.2:** authentication / authorization, persistence, sessions, streaming, and any **frontend-specific** coupling (the API is a thin backend contract).

### Success vs error contract (design now; implement incrementally)

| Situation | Intended HTTP / body shape (contract) |
|-----------|----------------------------------------|
| **No rows after retrieval** (empty evidence) | Prefer **`200 OK`** with the **normal success envelope**: `answer` may be the grounded “insufficient evidence…” string; if `explain=true`, `explanation.evidence` is `[]` and confidence reflects thin context — **not** a generic 404 unless product policy changes. |
| **Malformed request** (invalid JSON, validation failure, bad types) | **`400`**, `error.code` = **`invalid_request`** (optional **`details`**: Pydantic issues — no stack traces). |
| **LLM unreachable / provider error / missing OpenAI key** | **`503`**, `error.code` = **`backend_unavailable`**. |
| **Reranker unavailable** (model load failure, etc.) | **Degrade by default** when the pipeline can skip rerank; otherwise may surface as **`backend_unavailable`** / **`pipeline_error`** depending on exception type (see **`API_CONTRACT.md`**). |
| **Retrieval I/O / missing index** | **`500`**, `error.code` = **`pipeline_error`**. |
| **Unexpected server bug** | **`500`**, `error.code` = **`internal_error`**; log traceback server-side only. |

**Error body shape (stable keys):** `{ "error": { "code": "<string>", "message": "<string>", "details": …? } }` — same wrapper for non-2xx responses.

### Confidence — API trust (must appear in API docs / OpenAPI description)

Clients should be told explicitly (summary + field descriptions):

- **`confidence` is a heuristic** — rules over pipeline state, **not** a calibrated probability.  
- It is **not** suitable for automated risk decisions without further work.  
- It does **not** detect **cross-chunk contradiction**; conflicting excerpts can both appear under `evidence` with no disagreement flag.  

(Full behavioral table remains in **SYSTEM_EVOLUTION.md** → Phase 5.1.)

### User-facing response vs debug / trace

- **Default JSON** = what an end user or product needs: `answer`, optional `explanation` (evidence + reasoning_summary + confidence), and **no** raw stack traces or full internal diagnostics.  
- **Debug / trace** (prompt text, full `rerank_decision`, raw dataframe dumps, etc.) = **opt-in only** (e.g. header `X-Debug: 1` or `debug=true` reserved for trusted callers) — **do not** enable by default so the explain schema stays clean and safe to expose externally.

### Checklist (implementation order suggestion)

- [x] Wire **`POST /query`** → `rag_pipeline.answer()`.  
- [x] Map success body (`answer`, `explanation`, `metadata`); map exceptions to **`error.code`**.  
- [x] Document in **`API_CONTRACT.md`** and FastAPI **`/docs`**.

---

## Phase 5.3 — Chat interface **(implemented)**

**Shipped:** Streamlit thin client **`ui/chat_ui.py`** (+ **`ui/chat_helpers.py`**, **`scripts/run_chat_ui.py`**) — calls Phase 5.2 **`POST /query`** only; see **`API_CONTRACT.md`** → *Chat UI (Phase 5.3)*.

- **Answer** (prominent), optional **explain** mode: **confidence**, **Evidence provided to the model**, **reasoning summary**, **metadata**.  
- Sidebar: API base URL (`CHAT_API_BASE` / default), optional overrides (LLM backend/model, selective rerank, `k`, rerank fields).  
- **Health check** via **`GET /health`**; clear error when API is down or request fails.  
- **Local-only** turn list in `st.session_state` (presentational; not backend memory).  

---

## Phase 5.4 — Thin conversational layer **(implemented)**

**Intent:** Better **follow-up UX** and **continuity** without turning the backend into a chat database or an autonomous agent.

**Shipped:** **`conversation_context`** on **`POST /query`**, **`src/conversation_state.py`**, **`src/followup_resolver.py`**, resolver wiring in **`src/api.py`** / **`src/rag_pipeline.py`**, Streamlit client payloads — see **`SYSTEM_EVOLUTION.md`** §12.4 and **`API_CONTRACT.md`** (Phase 5.4).

### What 5.4 *is*

- **Follow-up questions** that depend on the prior turn (“What about one-star reviews?”, “Summarize that more briefly”, “Only the negative ones”, “What are the buyer risks?”, “Why did you say that?”).  
- **Referring to the previous answer** and **reusing prior filters / scope** when the user’s new message is clearly a refinement or aspect shift—not a full restatement.  
- **Short-term context** held in a **small structured object** (UI session and/or optional API request payload), **not** inside FAISS or the retriever.

### What 5.4 is *not*

- Long-term or **user-profile** memory.  
- **Autonomous planning** or multi-agent orchestration.  
- **Persisted** chat history in a database (initially).  
- **Hidden** state inferred only from a long raw transcript in the prompt.  
- Changing retrieval from **vague** chat context without **explicit** merge rules.  
- **Explicit non-goal (one line):** Phase **5.4** does **not** attempt **long-horizon memory**, **user-profile** recall across sessions, or **tool-planning / multi-step autonomous** conversations—that remains **Phase 6+** and out of scope here.

### Design principle: separate context layer

Keep **conversation support** out of core retrieval and prompt templates:

```
User message
  → Conversation context builder (turn state)
  → Follow-up detection + query normalization / rewrite (rule-based first)
  → Existing RAG pipeline (parse → retrieve → … → answer)
  → Response
  → Store minimal turn summary for next turn
```

**Backend retrieval stays stateless per request** at first: each call is still “run pipeline on **resolved** query + filters”; enrichment happens **before** `RAGPipeline.answer()`, not by making the vector store session-aware.

### Carry-forward state (minimal, explicit)

Enough for most follow-ups—**not** full transcript reasoning:

- Last user text; last **normalized / resolved** query text.  
- Last **`query_family`**, last **`filters`**.  
- Last answer (or short summary); last **evidence chunk ids**; last **`explain`** flag if useful.

### “Minimal turn summary” — bounded structure (no free-form prose)

Each completed turn should append a **small, typed record** (e.g. JSON-serializable dict / dataclass), **not** an open-ended summary paragraph. Planned fields (names indicative; all optional in wire format except what the client chooses to send):

| Field | Role |
|-------|------|
| `user_query_raw` | Exact last user string for the turn. |
| `resolved_query_text` | What was passed into `QueryParser` / pipeline after merge. |
| `query_family` | Runtime family after parse (or resolver override if any). |
| `filters` | Structured filter dict (or empty). |
| `answer_summary` | **Short** bounded string (e.g. first *N* chars + hash or fixed cap), **or** full answer if size limits acceptable—not unbounded “model recap”. |
| `evidence_chunk_ids` | Ordered list of chunk ids from last context (for “which chunks?” follow-ups). |
| `explain_used` | Boolean for last request. |

This keeps turn state **stable, inspectable, and diffable** in logs and API payloads.

### Precedence when multiple follow-up signals appear

If a single user message triggers **more than one** class (e.g. *“What about one-star reviews? Make it shorter.”*), resolution must be **deterministic**.

**Planned default precedence (highest wins first):**

1. **Explicit reset / new session** (see below) — if triggered, **drop** prior context before other rules.  
2. **Scope refinement** (rating, sentiment, product scope, “only …”) — **mutates `filters` and/or resolved query topic** first so retrieval targets the right pool.  
3. **Aspect shift** (same session topic, different `query_family` intent) — applied against the **scope already set** in step 2 if both present.  
4. **Ask-for-explanation** (“why?”, “which chunks?”) — if detected, may **short-circuit** into explanation behavior using **prior** answer + ids without re-retrieval, unless scope also changed (then **retrieve** with new scope first, then explain).  
5. **Output refinement** (“shorter”, “bullets”) — applied **last** as **generation / template instructions** on the already-scoped request (or as a post-pass instruction to the prompt builder), so format does not silently override rating filters.

**Rule of thumb:** **scope > aspect > explain-without-retrieve > output format**, with **reset** above all. Document any deviation in **`SYSTEM_EVOLUTION.md`** when implementing.

### When **not** to reuse prior context

Define **sticky-off** conditions so the layer does not merge incorrectly:

| Condition | Behavior |
|-----------|----------|
| **Explicit reset** | User says “new question”, “start over”, “ignore that”, or UI **Reset** — clear `ConversationState` (or send empty context). |
| **Clearly new topic** | Long self-contained question with **no** follow-up cues and **different** product/entity keywords (heuristic / token overlap threshold—TBD in code). **Do not merge.** |
| **Stale UI session** | Optional: if wall-clock **idle > T** minutes (configurable), treat as **cold start** unless user confirms “continue previous”. |
| **Semantic drift** | If new message is **long** and **low overlap** with `resolved_query_text` / last topic string, prefer **no merge** (tunable threshold). |

If none of the above fire **and** follow-up heuristics match, then **reuse** prior fields per precedence rules.

### Follow-up handling (phased)

1. **Detection** — simple heuristics: short message, pronouns (“that”, “those”, “it”), refinement phrases (“what about”, “only”, “more briefly”, “compare”, “why”).  
2. **Resolution / rewrite** — rule-based merge into a **single** query string + filter updates (e.g. “What about one-star?” → explicit one-star scoped query aligned with prior topic).  
3. **Transparency** — in **explain / debug** surfaces, show **original user text**, **resolved query**, whether follow-up logic ran, and **which prior fields** were reused.

### Follow-up *classes* to support first

| Class | Examples |
|-------|-----------|
| **Scope refinement** | “Only one-star”, “Only negative ones”, “For this product only” |
| **Output refinement** | “Shorter”, “Bullet points”, “Top 3 issues only” |
| **Ask-for-explanation** | “Why?”, “Which chunks support that?”, “How confident?” |
| **Aspect shift (same topic)** | “What about value complaints?”, “Buyer risks?”, “Any symptom mentions?” — aligned with existing **`query_family`** taxonomy |

### Code / API / UI touchpoints (shipped)

| Area | Role |
|------|------|
| **`src/conversation_state.py`** | Typed **turn / session** state; serializable for API body. |
| **`src/followup_resolver.py`** | Detect follow-up, merge with prior state, emit **resolved query + filters**. |
| **`src/api.py`** | Optional **`conversation_context`** on **`POST /query`**; pass resolved text into pipeline. |
| **`ui/chat_ui.py`** | Keeps **recent turns** in `st.session_state`, sends **context payload** with each request. |

### API additivity (implementation gate)

When using 5.4:

- **`POST /query` without conversation fields** must behave **exactly** as today (no required new keys).  
- **`conversation_context`** (and any `conversation_id`) remain **optional**.  
- **`resolved_query_text`**, **`followup_resolution_applied`**, and related **metadata** must be **optional** on the response so old clients ignore them.

### Success criteria (manual)

After 5.4, a single session should handle **without retyping the whole question**:

- “What about one-star reviews?”  
- “Now just give me buyer risks.”  
- “Can you make that shorter?”  
- “Which evidence supports that?”  

### After 5.4

**Phase 5.5** (optional) — LLM **retrieval query planner**; then **Phase 6.1** — first real **analytics / ML tool** behind the same API—progression: answer well → explain well → **converse briefly** → **optional planner-tuned retrieval** → then **analyze**.

---

## Phase 5.5 — LLM query planner (retrieval) **(implemented, opt-in)**

**Goal:** Improve **retrieval request shape** (paraphrase for embed / keyword overlap, optional **`review_rating`**, optional allowlisted **`query_family`**) via one **small structured LLM JSON** call, **validated** before merge—without changing retriever/rerank internals.

- **Request:** **`query_planner`** on **`POST /query`** (or server default **`QUERY_PLANNER_DEFAULT`** in **`src/config.py`** — currently **`false`**).  
- **UI:** Streamlit sidebar **“LLM query planner (retrieval)”** sends **`query_planner`: true** when checked.  
- **Docs / code:** **`API_CONTRACT.md`** (Phase 5.5), **`SYSTEM_EVOLUTION.md`** §12.5, **`src/query_planner.py`**.

**Product stance:** Default **off** for predictable behavior, tests, and cost; turn **on** when negative/low-rating phrasing and parser gaps justify the extra LLM call.

---

## Design principles

- **Thin frontend** — intelligence stays in **`src/`** pipeline.  
- **Expose system reasoning** — template + diagnostics — **not** hidden chain-of-thought.  
- Prioritize **debuggability** and **trust**.  

---

## Relationship to ML roadmap

| Document | Focus |
|----------|--------|
| **This file (`PRODUCT_ROADMAP.md`)** | **How users interact** with the system (API, chat, conversation). |
| **`ML_CAPABILITIES_ROADMAP.md`** | **Future analytical / predictive** capabilities behind the interface (Phase 6 tools). |

Core retrieval quality through **Phase 4.5** and explainability **5.1** are documented in **SYSTEM_EVOLUTION.md**.
