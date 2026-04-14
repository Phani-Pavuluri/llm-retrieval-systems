"""Phase 5.4 — rule-based follow-up detection and resolver (no LLM)."""
from __future__ import annotations

import unittest

from src.conversation_state import ConversationContext, TurnRecord
from src.followup_resolver import detect_followup, resolve_followup


def _ctx(last_resolved: str, **kwargs: object) -> ConversationContext:
    defaults: dict = {
        "user_query_raw": "seed",
        "resolved_query": last_resolved,
        "query_family": "abstract_complaint_summary",
        "filters": {},
        "answer_summary": "prior answer",
        "chunk_ids": ["c1"],
        "explain_used": False,
    }
    defaults.update(kwargs)
    return ConversationContext(turns=[TurnRecord.model_validate(defaults)])


class TestDetectFollowup(unittest.TestCase):
    def test_no_context(self) -> None:
        self.assertEqual(detect_followup("what about it", None), (False, "none"))

    def test_short_pronoun_triggers(self) -> None:
        ctx = _ctx("Summarize complaints about battery life")
        is_fb, ft = detect_followup("what about that", ctx)
        self.assertTrue(is_fb)
        self.assertEqual(ft, "aspect")

    def test_new_topic_long_query_no_cues(self) -> None:
        ctx = _ctx("Summarize complaints")
        q = "Write a detailed essay on quantum computing including history and recent papers"
        self.assertEqual(detect_followup(q, ctx), (False, "none"))


class TestResolveFollowup(unittest.TestCase):
    def test_case1_scope_one_star(self) -> None:
        ctx = _ctx("Summarize complaints")
        r = resolve_followup("what about one-star", ctx)
        self.assertTrue(r.is_followup)
        self.assertEqual(r.followup_type, "scope")
        self.assertEqual(r.filter_overrides, {"review_rating": 1})
        self.assertIn("Summarize complaints", r.resolved_query)
        self.assertIn("one-star", r.resolved_query.lower())
        self.assertIn("filters", r.resolver_metadata.get("reused_fields", []))

    def test_case2_output_shorter(self) -> None:
        ctx = _ctx("Summarize issues with this product")
        r = resolve_followup("make it shorter", ctx)
        self.assertTrue(r.is_followup)
        self.assertEqual(r.followup_type, "output")
        self.assertEqual(r.resolved_query, "Summarize issues with this product")
        self.assertEqual((r.output_style_hints or {}).get("brevity"), "short")
        self.assertFalse(r.reset_filters)

    def test_case3_aspect_value_family(self) -> None:
        ctx = _ctx("buyer complaints about durability", query_family="buyer_risk_issues")
        r = resolve_followup("what about value", ctx)
        self.assertTrue(r.is_followup)
        self.assertEqual(r.followup_type, "aspect")
        self.assertEqual(r.query_family_override, "value_complaint")
        self.assertTrue(r.reset_filters)

    def test_case4_explain_why(self) -> None:
        ctx = _ctx("Summarize issues")
        r = resolve_followup("why", ctx)
        self.assertTrue(r.is_followup)
        self.assertEqual(r.followup_type, "explain")
        self.assertTrue(r.explain_force)
        self.assertEqual(r.resolved_query, "Summarize issues")

    def test_case5_non_followup_without_cues(self) -> None:
        ctx = _ctx("Summarize issues")
        r = resolve_followup("List every SKU in the catalog with prices", ctx)
        self.assertFalse(r.is_followup)
        self.assertEqual(r.followup_type, "none")
        self.assertEqual(r.resolved_query, "List every SKU in the catalog with prices")
        self.assertEqual(r.resolver_metadata.get("reused_fields"), [])


if __name__ == "__main__":
    unittest.main()
