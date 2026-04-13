from __future__ import annotations

import unittest

import pandas as pd

from src.hybrid_scoring import apply_hybrid_scoring, keyword_overlap_score, tokenize


class TestHybridScoring(unittest.TestCase):
    def test_tokenize(self) -> None:
        self.assertEqual(tokenize("Hello, World!"), {"hello", "world"})

    def test_keyword_overlap(self) -> None:
        self.assertEqual(keyword_overlap_score("fake product", "this is fake"), 0.5)
        self.assertEqual(keyword_overlap_score("", "x"), 0.0)

    def test_apply_hybrid_scoring_order(self) -> None:
        df = pd.DataFrame(
            {
                "score": [0.9, 0.5, 0.6],
                "text": [
                    "great deodorant love it",
                    "counterfeit fake product not real",
                    "okay average product",
                ],
            }
        )
        out = apply_hybrid_scoring(
            df, "counterfeit fake not authentic", alpha=0.3, beta=0.7
        )
        # Hybrid should rank the lexically matching chunk above pure high semantic
        top_text = out.sort_values("score", ascending=False).iloc[0]["text"]
        self.assertIn("counterfeit", top_text)

    def test_constant_semantic_column(self) -> None:
        df = pd.DataFrame({"score": [0.5, 0.5], "text": ["a b", "a b c"]})
        out = apply_hybrid_scoring(df, "c", alpha=0.5, beta=0.5)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
