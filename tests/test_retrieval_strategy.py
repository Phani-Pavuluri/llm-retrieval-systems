from __future__ import annotations

import unittest

from src.retrieval_request import RetrievalRequest
from src.retrieval_strategy import apply_strategy_to_request, select_retrieval_strategy


class TestRetrievalStrategy(unittest.TestCase):
    def test_issue_keywords_hybrid_strong(self) -> None:
        s = select_retrieval_strategy("general_qa", "Is this counterfeit or fake?")
        self.assertTrue(s.use_hybrid)
        self.assertEqual(s.hybrid_alpha, 0.6)
        self.assertEqual(s.hybrid_beta, 0.4)
        self.assertEqual(s.reason, "issue_keywords")

    def test_summary_phrase_vector(self) -> None:
        s = select_retrieval_strategy(
            "complaint_summary", "Give a summary of common complaints"
        )
        self.assertFalse(s.use_hybrid)
        self.assertIsNone(s.hybrid_alpha)
        self.assertEqual(s.reason, "abstract_summary_vector")

    def test_complaint_task_light_hybrid(self) -> None:
        s = select_retrieval_strategy(
            "complaint_summary", "What did people dislike about shipping speed"
        )
        self.assertTrue(s.use_hybrid)
        self.assertEqual(s.hybrid_alpha, 0.85)
        self.assertEqual(s.hybrid_beta, 0.15)
        self.assertEqual(s.reason, "complaint_task_light_hybrid")

    def test_default_general_vector(self) -> None:
        s = select_retrieval_strategy("general_qa", "What is the brand name")
        self.assertFalse(s.use_hybrid)
        self.assertEqual(s.reason, "default_vector")

    def test_apply_mutates_request(self) -> None:
        r = RetrievalRequest(
            query_text="rash after using stick",
            top_k=5,
            task_type="general_qa",
        )
        apply_strategy_to_request(r)
        self.assertTrue(r.use_hybrid)
        self.assertEqual(r.strategy_reason, "issue_keywords")
        self.assertEqual(r.candidate_pool_multiplier, 1.5)

    def test_pool_multiplier_rating_filter(self) -> None:
        r = RetrievalRequest(
            query_text="What is the brand name",
            top_k=5,
            task_type="general_qa",
            filters={"review_rating": {"max": 3}},
        )
        apply_strategy_to_request(r)
        self.assertEqual(r.candidate_pool_multiplier, 1.4)


if __name__ == "__main__":
    unittest.main()
