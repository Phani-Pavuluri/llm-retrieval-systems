from __future__ import annotations

import unittest

import pandas as pd

from src.evidence_negation_filter import (
    filter_absence_focused_excerpts,
    health_intent_query,
    is_primarily_absence_excerpt,
    user_seeks_reassurance,
)


class TestNegationFilter(unittest.TestCase):
    def test_reassurance_query_skips_filter(self) -> None:
        self.assertTrue(user_seeks_reassurance("Is it safe for sensitive skin?"))
        df = pd.DataFrame(
            {
                "chunk_id": ["a"],
                "text": ["I did not experience any rash with this product."],
            }
        )
        out, stats = filter_absence_focused_excerpts(
            df, user_question="Is it safe for sensitive skin?"
        )
        self.assertTrue(stats.get("skipped"))
        self.assertEqual(len(out), 1)

    def test_non_health_query_skips(self) -> None:
        self.assertFalse(health_intent_query("What is the return policy?"))
        df = pd.DataFrame(
            {
                "chunk_id": ["a"],
                "text": ["I never had any issues with shipping."],
            }
        )
        out, stats = filter_absence_focused_excerpts(
            df, user_question="What is the return policy?"
        )
        self.assertTrue(stats.get("skipped"))
        self.assertEqual(len(out), 1)

    def test_drops_absence_only_chunk(self) -> None:
        long_absence = (
            "No rash and no irritation for me after two weeks of daily use; "
            "I also did not notice any redness or sensitivity."
        )
        self.assertTrue(is_primarily_absence_excerpt(long_absence))
        df = pd.DataFrame(
            {
                "chunk_id": ["neg", "pos"],
                "text": [
                    (
                        "I did not experience any rash while using this deodorant "
                        "and I never saw any irritation either."
                    ),
                    "After a week I had severe burning and raw underarms.",
                ],
            }
        )
        out, stats = filter_absence_focused_excerpts(
            df, user_question="product with health issues"
        )
        self.assertFalse(stats.get("skipped"))
        self.assertEqual(list(out["chunk_id"]), ["pos"])

    def test_keeps_mixed_chunk_with_clear_adverse_detail(self) -> None:
        txt = (
            "I did not get a rash, but I did experience severe burning pain "
            "and redness that lasted days."
        )
        self.assertFalse(is_primarily_absence_excerpt(txt))
        df = pd.DataFrame({"chunk_id": ["mix"], "text": [txt]})
        out, stats = filter_absence_focused_excerpts(
            df, user_question="any skin reactions?"
        )
        self.assertEqual(len(out), 1)
        self.assertTrue(stats.get("skipped"))
        self.assertEqual(stats.get("reason"), "no_rows_matched")


if __name__ == "__main__":
    unittest.main()
