"""
Canonical answer-eval failure buckets (Phase 4.3).

Keep in sync with `failure_buckets` in eval/answer_eval_labeled.json.
"""
from __future__ import annotations

# Simple, reusable bucket ids for manual labeling after runs.
ANSWER_EVAL_FAILURE_BUCKETS: tuple[str, ...] = (
    "hallucination",
    "incomplete_summary",
    "wrong_scope",
    "ignored_filter",
    "weak_synthesis",
    "overfocused_on_single_chunk",
    "insufficient_evidence_handled_poorly",
)

ANSWER_EVAL_FAILURE_BUCKET_SET: frozenset[str] = frozenset(ANSWER_EVAL_FAILURE_BUCKETS)
