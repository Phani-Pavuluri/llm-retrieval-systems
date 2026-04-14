"""
Phase 5.3 — Minimal Streamlit chat UI (thin HTTP client to Phase 5.2 API).

Run from repo root:
  PYTHONPATH=. streamlit run ui/chat_ui.py

Or: python scripts/run_chat_ui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests
import streamlit as st

from ui.chat_helpers import (
    DEFAULT_API_BASE,
    build_query_json,
    check_api_health,
    confidence_markdown_lines,
    conversation_context_from_turn_history,
    evidence_score_parts,
    format_api_error,
    metadata_markdown_lines,
    reasoning_summary_lines,
)

st.set_page_config(page_title="RAG Chat", layout="wide", initial_sidebar_state="expanded")

st.title("Review Q&A")
st.caption(
    "Thin client over **`POST /query`** — retrieval, prompts, and answers all run in the API / "
    "backend. **Session history** (below) is **local display only**; each call is independent "
    "(no server-side memory)."
)

# --- Sidebar: connection + advanced ---
st.sidebar.header("Connection")
api_base = st.sidebar.text_input(
    "API base URL",
    value=DEFAULT_API_BASE,
    help="Default from `CHAT_API_BASE` env or http://127.0.0.1:8000. Start API: `PYTHONPATH=. uvicorn src.api:app`",
)

if "health_for_base" not in st.session_state:
    st.session_state.health_for_base = None
if "health_ok" not in st.session_state:
    st.session_state.health_ok = None
if "health_msg" not in st.session_state:
    st.session_state.health_msg = None

base_norm = api_base.strip().rstrip("/") or DEFAULT_API_BASE
if st.sidebar.button("Check API connection") or st.session_state.health_for_base != base_norm:
    ok, msg = check_api_health(base_norm)
    st.session_state.health_for_base = base_norm
    st.session_state.health_ok = ok
    st.session_state.health_msg = msg

if st.session_state.health_ok is False:
    st.error(f"**API not reachable** at `{base_norm}` — {st.session_state.health_msg}")
elif st.session_state.health_ok is True:
    st.sidebar.success("API health: ok")

st.sidebar.header("Advanced (optional)")
with st.sidebar.expander("Overrides", expanded=False):
    _back_opts = ("(API default)", "ollama", "openai")
    _bi = st.selectbox("LLM backend", range(len(_back_opts)), format_func=lambda i: _back_opts[i])
    llm_backend = None if _bi == 0 else _back_opts[_bi]
    llm_model = st.text_input("LLM model", value="", help="Empty = backend default")
    llm_model_v = llm_model.strip() or None

    _sr_opts = ("(API default)", "On", "Off")
    _sri = st.selectbox("Selective rerank", range(len(_sr_opts)), format_func=lambda i: _sr_opts[i])
    selective_rerank: bool | None
    if _sri == 0:
        selective_rerank = None
    elif _sri == 1:
        selective_rerank = True
    else:
        selective_rerank = False

    rerank_model = st.text_input("Rerank model id", value="").strip() or None
    k_raw = st.number_input("k (top chunks)", min_value=0, value=0, help="0 = use API default")
    k_val = None if int(k_raw) == 0 else int(k_raw)
    rn_raw = st.number_input("rerank_top_n", min_value=0, value=0, help="0 = omit / API default")
    rerank_top_n = None if int(rn_raw) == 0 else int(rn_raw)

# --- Main: query ---
if "turns" not in st.session_state:
    st.session_state.turns = []

query = st.text_area("Your question", height=120, placeholder="Ask about the review corpus…")
explain = st.checkbox("Explainable answer (evidence + reasoning + confidence)", value=False)

submitted = st.button("Submit", type="primary")

if submitted:
    if not (query or "").strip():
        st.warning("Enter a question.")
    else:
        conv_ctx = conversation_context_from_turn_history(st.session_state.turns)
        payload = build_query_json(
            query=query,
            explain=explain,
            llm_backend=llm_backend,
            llm_model=llm_model_v,
            k=k_val,
            selective_rerank=selective_rerank,
            rerank_model=rerank_model,
            rerank_top_n=rerank_top_n,
            conversation_context=conv_ctx,
        )
        url = f"{base_norm}/query"
        try:
            resp = requests.post(url, json=payload, timeout=300)
        except requests.RequestException as e:
            st.session_state.turns.append(
                {"query": query, "ok": False, "error": f"HTTP client error: {e}"}
            )
            st.error(f"Could not reach `{url}`: {e}")
        else:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    st.error("API returned non-JSON body.")
                    st.session_state.turns.append(
                        {"query": query, "ok": False, "error": "non-json response"}
                    )
                else:
                    st.session_state.turns.append({"query": query, "ok": True, "data": data})
            else:
                try:
                    body = resp.json()
                except ValueError:
                    body = {}
                code, msg, det = format_api_error(body) if body else (
                    "http_error",
                    (resp.text or "")[:800],
                    None,
                )
                err_display = f"**{code}** (HTTP {resp.status_code}): {msg}"
                st.error(err_display)
                if det is not None:
                    with st.expander("Error details"):
                        st.json(det)
                st.session_state.turns.append(
                    {
                        "query": query,
                        "ok": False,
                        "error": err_display,
                        "details": det,
                    }
                )

# --- Render latest turn prominently ---
if st.session_state.turns:
    last = st.session_state.turns[-1]
    st.divider()
    st.subheader("Current turn")
    st.markdown(f"**Query:** {last.get('query', '')}")
    if not last.get("ok"):
        st.markdown(last.get("error", "Unknown error"))
    else:
        data = last["data"]
        meta = data.get("metadata") or {}
        if meta.get("is_followup"):
            st.info(
                "**Resolved query** (rule-based merge for this turn):\n\n"
                f"```\n{(meta.get('resolved_query') or '').strip()}\n```"
            )
        st.markdown("### Answer")
        st.markdown(data.get("answer", "") or "_Empty answer_")

        ex = data.get("explanation")

        if ex:
            ct = ex.get("conversation_transparency")
            if isinstance(ct, dict) and ct:
                with st.expander("Conversation resolution (explain mode)", expanded=False):
                    st.markdown(f"**Original query:** `{ct.get('original_query', '')}`")
                    st.markdown(f"**Resolved query:** `{ct.get('resolved_query', '')}`")
                    st.markdown(f"**Reused fields:** `{ct.get('reused_fields', [])!r}`")
                    prior = ct.get("prior_turn")
                    if isinstance(prior, dict) and prior:
                        st.markdown("**Prior turn (client-supplied context):**")
                        st.json(prior)
                    elif ct.get("prior_turn") is None:
                        st.caption("No prior turn in `conversation_context` for this request.")

            conf = ex.get("confidence") or {}
            if conf:
                st.markdown("### Confidence (heuristic)")
                st.markdown("\n\n".join(confidence_markdown_lines(conf)))

            evs = ex.get("evidence") or []
            st.markdown("### Evidence provided to the model")
            st.caption(
                "Chunks in the **LLM context window** — not proof of which excerpts the model relied on most."
            )
            if not evs:
                st.info("No evidence rows (empty retrieval context).")
            for i, ev in enumerate(evs, start=1):
                cid = ev.get("chunk_id", "—")
                sid = ev.get("source_id") or "—"
                rank = ev.get("rank_position", "—")
                txt = ev.get("chunk_text") or ev.get("text") or ""
                scores = evidence_score_parts(ev)
                score_line = " · ".join(scores) if scores else ""
                with st.container():
                    st.markdown(f"**{i}.** `{cid}` · source=`{sid}` · rank=`{rank}`")
                    if score_line:
                        st.caption(score_line)
                    excerpt = txt if len(txt) <= 12000 else txt[:12000] + "\n…"
                    st.text(excerpt)

            rs = ex.get("reasoning_summary") or {}
            if rs:
                st.markdown("### Reasoning summary (system)")
                st.markdown("\n\n".join(reasoning_summary_lines(rs)))

        st.markdown("### Metadata")
        mlines = metadata_markdown_lines(meta)
        if mlines:
            st.markdown("\n".join(mlines))
        else:
            st.caption("No metadata returned.")

    if len(st.session_state.turns) > 1:
        with st.expander("Previous turns (local only, display-only)", expanded=False):
            for t in reversed(st.session_state.turns[:-1]):
                st.markdown(f"**Q:** {t.get('query','')}")
                if t.get("ok"):
                    st.markdown((t.get("data") or {}).get("answer", "")[:500] + ("…" if len((t.get("data") or {}).get("answer", "")) > 500 else ""))
                else:
                    st.caption(t.get("error", "error"))
                st.divider()

st.divider()
st.caption(
    "Start API: `PYTHONPATH=. uvicorn src.api:app --host 127.0.0.1 --port 8000` · "
    f"Contract: **API_CONTRACT.md** · Default base: `{DEFAULT_API_BASE}`"
)
