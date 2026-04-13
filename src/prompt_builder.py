"""
Grounded, task-aware prompts for RAG answers (Phase 4).

Builds a single user message for existing OpenAI/Ollama backends. Does not call the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.retrieval_request import RetrievalRequest

# Align with rerank / eval taxonomy where possible.
_FAMILY_TEMPLATE_IDS: dict[str, str] = {
    "abstract_complaint_summary": "family_abstract_complaint_summary",
    "value_complaint": "family_value_complaint",
    "exact_issue_lookup": "family_exact_issue_lookup",
    "rating_scoped_summary": "family_rating_scoped_summary",
    "buyer_risk_issues": "family_buyer_risk_issues",
    "symptom_issue_extraction": "family_symptom_issue_extraction",
}

_TASK_FALLBACK_TEMPLATE_IDS: dict[str, str] = {
    "complaint_summary": "task_complaint_summary",
    "general_qa": "task_general_qa",
}

_DEFAULT_TEMPLATE_ID = "grounded_default"


@dataclass(frozen=True)
class BuiltAnswerPrompt:
    """One assembled prompt for the answer step."""

    prompt: str
    template_id: str
    template_label: str
    chunk_ids: list[str]


def _family_for_prompting(request: RetrievalRequest) -> str | None:
    qf = getattr(request, "query_family", None)
    if qf and str(qf).strip():
        return str(qf).strip()
    return None


def select_prompt_template_id(request: RetrievalRequest) -> str:
    """
    Choose template by query_family first, then task_type, else strong default.
    """
    fam = _family_for_prompting(request)
    if fam and fam in _FAMILY_TEMPLATE_IDS:
        return _FAMILY_TEMPLATE_IDS[fam]
    tt = (request.task_type or "general_qa").strip()
    return _TASK_FALLBACK_TEMPLATE_IDS.get(tt, _DEFAULT_TEMPLATE_ID)


def format_evidence_block(retrieved: pd.DataFrame) -> tuple[str, list[str]]:
    """
    Build numbered excerpt block and list of chunk_ids (stable order).
    If empty, returns an explicit empty-evidence placeholder (still grounded).
    """
    if retrieved is None or retrieved.empty or "chunk_id" not in retrieved.columns:
        return (
            "(No review excerpts were retrieved. Do not invent reviews.)",
            [],
        )
    lines: list[str] = []
    ids: list[str] = []
    for i, (_, row) in enumerate(retrieved.iterrows(), start=1):
        cid = str(row.get("chunk_id", ""))
        ids.append(cid)
        text = row.get("text", "")
        lines.append(f"[Chunk {i} | id={cid}]\n{text}")
    return "\n\n".join(lines), ids


_GROUNDING_RULES = """Grounding rules (non-negotiable):
- Use ONLY information supported by the excerpts above. If the excerpts do not contain enough information, say so clearly (e.g. start with "Insufficient evidence in the retrieved excerpts to …").
- Do NOT invent product names, ratings, dates, or review quotes that are not in the excerpts.
- If excerpts conflict, acknowledge the conflict briefly instead of smoothing it away.
- Cite evidence by referring to "Chunk N" when helpful; do not fabricate chunk content."""

_CAUSAL_AND_CLAIM_RULES = """Claims and causation (strict):
- Do NOT infer causes, blame, medical diagnoses, or product defects beyond what the reviewer explicitly states about this product.
- If a reviewer says they did not experience a symptom with this product, do NOT rephrase that as the product causing that symptom (e.g. "no rash" is not evidence that the product caused a rash).
- Absence of a topic in an excerpt is not evidence that the topic occurred; do not fill gaps with plausible stories."""

_MULTI_CHUNK_RULES_MULTI = """Using multiple excerpts:
- When two or more chunks contain relevant material, your answer must use at least two different Chunk numbers for substantive claims (do not answer from a single chunk while ignoring other clearly relevant chunks).
- Group insights by theme; under each theme, tie bullets to the chunks that support them. If another chunk adds no on-topic detail, say so briefly instead of inventing relevance."""

_MULTI_CHUNK_RULES_SINGLE = """Using multiple excerpts:
- Only one excerpt was retrieved. Do not imply multiple independent reviewers or broad survey-style conclusions unless that single excerpt clearly supports it."""


def _rating_scope_instruction(request: RetrievalRequest) -> str:
    """Tell the model exactly what metadata filter was applied (generation-side scope)."""
    rr = request.filters.get("review_rating")
    if rr is None:
        return ""
    if isinstance(rr, dict):
        lo = rr.get("min")
        hi = rr.get("max")
        if lo is not None and hi is not None:
            return f"""Retrieval scope (metadata — obey exactly in wording and spirit):
- Only excerpts with review_rating between {lo} and {hi} (inclusive) were retrieved. Do NOT summarize as if higher- or lower-star reviews outside that band were in scope."""
        if hi is not None:
            return f"""Retrieval scope (metadata — obey exactly in wording and spirit):
- Only excerpts with review_rating ≤ {hi} were retrieved. Do NOT describe 4–5 star praise as if it were part of this retrieval unless a retrieved excerpt explicitly quotes or references it."""
        if lo is not None:
            return f"""Retrieval scope (metadata — obey exactly in wording and spirit):
- Only excerpts with review_rating ≥ {lo} were retrieved. Do NOT collapse this into "low rated" or complaints unless the excerpts support it."""
        return ""
    # Scalar → equality (e.g. literal one-star)
    return f"""Retrieval scope (metadata — obey exactly in wording and spirit):
- Only excerpts with review_rating equal to {rr} were retrieved. Do NOT generalize to other star levels or imply a wider rating mix than this filter allows."""


def _needs_negative_only_complaint_synthesis(
    template_id: str, request: RetrievalRequest
) -> bool:
    if template_id in (
        "family_abstract_complaint_summary",
        "task_complaint_summary",
        "family_buyer_risk_issues",
    ):
        return True
    if template_id == "family_rating_scoped_summary":
        return "review_rating" in request.filters
    return False


_NEGATIVE_ONLY_COMPLAINT_RULES = """Complaint / negative synthesis scope (strict):
- ONLY include negative feedback, problems, risks, or dissatisfaction that is directly supported by the excerpts. Do NOT include positive experiences, endorsements, or unrelated praise (even if the same chunk also contains praise).
- If a chunk mixes praise and complaints, use only the on-complaint lines for this answer; do not "balance" with the positive lines unless the user question explicitly asks for both sides."""

_STRUCTURED_OUTPUT_COMPLAINT = """Output format (required):
- Use markdown: 2–4 short theme headings (e.g. "### Irritation / skin"), each followed by bullet points.
- Each bullet: one insight, phrased as synthesis (not a raw dump), and tagged with "Chunk N" (and Chunk M if two chunks jointly support the same theme).
- Avoid a single flat list of quotes with no grouping."""

_NEGATION_SYMPTOM_EXTRACTION = """Symptom / issue extraction (strict — negation-aware):
- ONLY list symptoms, reactions, or skin problems that the reviewer explicitly attributes to **this product** (the product under review in that chunk).
- If a chunk says they did **not** get a symptom with this product, or that a symptom came from **other** products/brands/categories, **do not** list that symptom as something this product caused.
- When a reviewer contrasts past experiences (other deodorants, antiperspirants, etc.) with this product, treat symptoms tied only to the others as **out of scope** for this answer unless the text clearly blames **this** product.
- If no qualifying in-scope symptoms appear in the excerpts, say so explicitly (e.g. "No excerpts attribute rash or irritation to this product").
- Output: short bullet list; each bullet must include Chunk N and a paraphrase tight to the wording of that chunk."""


def _section_for_template(template_id: str, request: RetrievalRequest) -> str:
    """Task-specific answer shape (Milestone B + Phase 4.4 synthesis)."""
    neg = (
        _NEGATIVE_ONLY_COMPLAINT_RULES + "\n\n" + _STRUCTURED_OUTPUT_COMPLAINT + "\n\n"
        if _needs_negative_only_complaint_synthesis(template_id, request)
        else ""
    )

    if template_id == "family_abstract_complaint_summary":
        return f"""{neg}Answer shape for this question:
- Summarize recurring problems or negative themes as organized insights (not a laundry list of every sentence).
- Merge overlapping points across chunks under the same theme; call out a theme as thinly evidenced if only one chunk supports it."""

    if template_id == "family_value_complaint":
        return """Answer shape for this question:
- Focus on value-for-money, price, misleading claims, quality-for-price, or whether the product was "worth it."
- Do not pad with generic unrelated praise; if a chunk praises unrelated aspects, omit unless it directly answers the value question.
- Output format: use bullet points under 2–3 short theme headings (e.g. "### Price vs quality"); cite Chunk N per bullet."""

    if template_id == "family_buyer_risk_issues":
        return f"""{neg}Answer shape for this question:
- This is a **buyer risk / pitfalls** answer, not a value-for-money essay: prioritize concrete problems, safety or authenticity concerns, usability failures, or strongly negative outcomes a purchaser should know.
- Do NOT reframe the question around price or "worth it" unless the user question explicitly asked about value or cost.
- Keep the answer brief and actionable; use theme headings and bullets with Chunk N cites."""

    if template_id == "family_symptom_issue_extraction":
        return f"""{_NEGATION_SYMPTOM_EXTRACTION}

Answer shape for this question:
- Treat this as an **extraction** task: answer only what is explicitly stated about the requested symptoms/issues.
- Do not summarize unrelated praise or general satisfaction unless it states a symptom tied to this product.
- If chunks disagree, note the disagreement in one line with Chunk cites."""

    if template_id == "family_exact_issue_lookup":
        return """Answer shape for this question:
- Treat this as a targeted lookup: state exactly what the excerpts say about the specific issue (defect, smell, authenticity concern, texture, etc.).
- Apply the causation rules above strictly: report what is stated, not what might explain it.
- For symptoms or skin reactions: only count issues explicitly tied to **this** product in the excerpt; if the reviewer attributes a symptom to other products or says it did not occur with this product, say so and do not list it as a product-caused issue.
- If the issue is not mentioned, say the retrieved evidence does not address it. Short bullets are fine; cite Chunk N."""

    if template_id == "family_rating_scoped_summary":
        # When `neg` is set, structured bullets are already in that block; avoid duplicating.
        tail = (
            ""
            if _needs_negative_only_complaint_synthesis(template_id, request)
            else "- Prefer bullet points with Chunk N cites.\n"
        )
        return f"""{neg}Answer shape for this question:
- Summarize experiences that match the rating scope implied by the question and by any retrieval filter described above.
- If the excerpts are not clearly from that scope, say the evidence is ambiguous or insufficient.
{tail}"""

    if template_id == "task_complaint_summary":
        return f"""{neg}Answer shape for this question:
- Emphasize concrete problems, defects, safety concerns, or strongly negative experiences grounded in the excerpts.
- If there are no substantive complaints in the excerpts, say so explicitly (do not invent complaints to satisfy the question)."""

    if template_id == "task_general_qa":
        return """Answer shape for this question:
- Answer directly and briefly, grounded in the excerpts.
- If the question is broad, prioritize the strongest supported points; use bullets when listing multiple points, with Chunk N cites."""

    # grounded_default and unknown
    return """Answer shape:
- Direct, concise answer grounded in the excerpts.
- If evidence is thin, say so explicitly; cite Chunk N where it helps."""


def _template_label(template_id: str) -> str:
    labels: dict[str, str] = {
        "grounded_default": "Strong default grounded QA",
        "family_abstract_complaint_summary": "Abstract / thematic complaint summary",
        "family_value_complaint": "Value-related complaints",
        "family_exact_issue_lookup": "Exact issue lookup",
        "family_rating_scoped_summary": "Rating-scoped summary",
        "family_buyer_risk_issues": "Buyer risk / pitfalls",
        "family_symptom_issue_extraction": "Symptom / reaction extraction",
        "task_complaint_summary": "Complaint-style QA (task)",
        "task_general_qa": "General QA (task)",
    }
    return labels.get(template_id, template_id)


def build_answer_prompt(
    request: RetrievalRequest,
    user_question: str,
    retrieved: pd.DataFrame,
    *,
    template_id: str | None = None,
) -> BuiltAnswerPrompt:
    """
    Assemble the full user prompt for one RAG turn.

    `user_question` is usually `request.original_query` or the raw user string.
    """
    tid = template_id or select_prompt_template_id(request)
    if tid not in _all_known_template_ids():
        tid = _DEFAULT_TEMPLATE_ID

    context, chunk_ids = format_evidence_block(retrieved)
    section = _section_for_template(tid, request)
    n_chunks = len(chunk_ids)
    multi = (
        _MULTI_CHUNK_RULES_MULTI if n_chunks >= 2 else _MULTI_CHUNK_RULES_SINGLE
    )
    rating_scope = _rating_scope_instruction(request)
    rating_block = f"{rating_scope}\n\n" if rating_scope else ""

    prompt = f"""You are a careful assistant answering questions about product reviews.

{_GROUNDING_RULES}

{_CAUSAL_AND_CLAIM_RULES}

{multi}

{rating_block}{section}

Excerpts (only source of truth):
{context}

User question:
{user_question.strip()}

Write your answer now:"""

    return BuiltAnswerPrompt(
        prompt=prompt.strip(),
        template_id=tid,
        template_label=_template_label(tid),
        chunk_ids=chunk_ids,
    )


def _all_known_template_ids() -> frozenset[str]:
    return frozenset(
        {
            _DEFAULT_TEMPLATE_ID,
            *_FAMILY_TEMPLATE_IDS.values(),
            *_TASK_FALLBACK_TEMPLATE_IDS.values(),
        }
    )


def describe_prompt_routing(request: RetrievalRequest) -> dict[str, Any]:
    """Small debug payload for traces / UI."""
    tid = select_prompt_template_id(request)
    return {
        "prompt_template_id": tid,
        "prompt_template_label": _template_label(tid),
        "query_family": getattr(request, "query_family", None),
        "task_type": request.task_type,
    }
