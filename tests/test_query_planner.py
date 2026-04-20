from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.query_parser import QueryParser
from src.query_planner import apply_llm_query_plan, maybe_apply_query_planner
from src.retrieval_request import RetrievalRequest


class TestQueryPlanner(unittest.TestCase):
    def test_apply_sets_rating_and_retrieval_text(self) -> None:
        llm = MagicMock()
        llm.generate.return_value = (
            '{"needs_low_rating_evidence": true, '
            '"retrieval_query_text": "deodorant negative reviews pain rash"}'
        )
        req = RetrievalRequest(query_text="worst deodorant experience", top_k=5, filters={})
        res = apply_llm_query_plan(user_query="worst deodorant experience", request=req, llm=llm)
        self.assertTrue(res.applied)
        self.assertEqual(req.filters.get("review_rating"), {"max": 3})
        self.assertEqual(req.query_text, "deodorant negative reviews pain rash")

    def test_maybe_skips_when_rules_have_rating(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = AssertionError("should not be called")
        parser = QueryParser()
        req = RetrievalRequest(
            query_text="negative deodorant experiences",
            top_k=5,
            filters={},
        )
        res = maybe_apply_query_planner(
            enabled=True,
            user_query="negative deodorant experiences",
            request=req,
            llm=llm,
            parser=parser,
            skip_if_followup_filters=False,
        )
        self.assertFalse(res.applied)
        self.assertEqual(res.notes, "skipped_rules_already_have_rating_filter")


if __name__ == "__main__":
    unittest.main()
