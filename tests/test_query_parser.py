"""Light tests for rule-based QueryParser."""
from __future__ import annotations

import unittest

from src.query_parser import QueryParser


class TestQueryParser(unittest.TestCase):
    def setUp(self) -> None:
        self.p = QueryParser()

    def test_general_qa_default(self) -> None:
        r = self.p.parse("What is the return policy?", top_k=3)
        self.assertEqual(r.task_type, "general_qa")
        self.assertEqual(r.top_k, 3)
        self.assertEqual(r.query_text, "What is the return policy?")
        self.assertEqual(r.filters, {})

    def test_complaint_task(self) -> None:
        r = self.p.parse("Summarize defective product complaints", top_k=5)
        self.assertEqual(r.task_type, "complaint_summary")

    def test_low_rated_filter(self) -> None:
        r = self.p.parse("Show low-rated reviews about quality", top_k=4)
        self.assertIn("review_rating", r.filters)
        self.assertEqual(r.filters["review_rating"], {"max": 3})

    def test_one_star_filter(self) -> None:
        r = self.p.parse("1-star reviews only", top_k=2)
        self.assertEqual(r.filters["review_rating"], 1)

    def test_one_star_hyphen_with_negative_word(self) -> None:
        """'one-star' must map to rating == 1, not broad negative max<=3."""
        r = self.p.parse("negative one-star experiences about rash", top_k=4)
        self.assertEqual(r.filters["review_rating"], 1)

    def test_infer_buyer_risk_family(self) -> None:
        r = self.p.parse("Briefly: what problems should a buyer watch out for?", top_k=5)
        self.assertEqual(r.query_family, "buyer_risk_issues")

    def test_infer_value_family(self) -> None:
        r = self.p.parse(
            "Is the product considered good value for money in the reviews?", top_k=5
        )
        self.assertEqual(r.query_family, "value_complaint")

    def test_infer_symptom_extraction_family(self) -> None:
        r = self.p.parse(
            "List specific symptoms like rash or skin irritation if mentioned.", top_k=5
        )
        self.assertEqual(r.query_family, "symptom_issue_extraction")

    def test_brand_from_phrase(self) -> None:
        r = self.p.parse("Issues from Schmidt's deodorant on Amazon", top_k=5)
        self.assertIn("brand", r.filters)
        self.assertIn("Schmidt", r.filters["brand"])


if __name__ == "__main__":
    unittest.main()
