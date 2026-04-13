"""Unit tests for task-aware selective rerank policy (rule-based)."""
from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from src import config
from src.rerank_policy import (
    apply_selective_rerank_policy,
    infer_query_family,
    should_skip_rerank_for_confidence,
)
from src.retrieval_request import RetrievalRequest


def _req(**kwargs: object) -> RetrievalRequest:
    defaults: dict[str, object] = {
        "query_text": "test query",
        "top_k": 5,
        "filters": {},
        "task_type": "general_qa",
        "use_rerank": None,
        "query_family": None,
        "strategy_reason": "",
    }
    defaults.update(kwargs)
    return RetrievalRequest(
        query_text=str(defaults["query_text"]),
        top_k=int(defaults["top_k"]),
        filters=dict(defaults["filters"]),  # type: ignore[arg-type]
        task_type=str(defaults["task_type"]),
        use_rerank=defaults["use_rerank"],  # type: ignore[arg-type]
        query_family=defaults["query_family"],  # type: ignore[arg-type]
        strategy_reason=str(defaults["strategy_reason"]),
    )


class TestInferQueryFamily(unittest.TestCase):
    def test_explicit_family_preserved(self) -> None:
        r = _req(query_family="value_complaint")
        self.assertEqual(infer_query_family(r), "value_complaint")

    def test_rating_filter_maps_to_rating_scoped(self) -> None:
        r = _req(query_family=None, filters={"review_rating": 5})
        self.assertEqual(infer_query_family(r), "rating_scoped_summary")

    def test_value_for_money_text_maps_value(self) -> None:
        r = _req(
            query_text="Is the product considered good value for money in the reviews?",
            task_type="general_qa",
        )
        self.assertEqual(infer_query_family(r), "value_complaint")

    def test_buyer_watch_out_maps_buyer_risk(self) -> None:
        r = _req(
            query_text="Briefly: what problems should a buyer watch out for?",
            task_type="complaint_summary",
        )
        self.assertEqual(infer_query_family(r), "buyer_risk_issues")

    def test_symptom_list_maps_extraction_family(self) -> None:
        r = _req(
            query_text="List specific symptoms like rash or skin irritation if mentioned.",
            task_type="general_qa",
        )
        self.assertEqual(infer_query_family(r), "symptom_issue_extraction")

    def test_complaint_without_value_maps_abstract_not_value(self) -> None:
        r = _req(
            query_text="What are common complaints about smell?",
            task_type="complaint_summary",
        )
        self.assertEqual(infer_query_family(r), "abstract_complaint_summary")


class TestSelectivePolicy(unittest.TestCase):
    def test_family_on_enables_rerank(self) -> None:
        r = _req(query_family="value_complaint")
        apply_selective_rerank_policy(r, selective_enabled=True)
        self.assertTrue(r.use_rerank)
        self.assertIn("family_on", r.rerank_reason)

    def test_family_off_disables_rerank(self) -> None:
        r = _req(query_family="exact_issue_lookup")
        apply_selective_rerank_policy(r, selective_enabled=True)
        self.assertFalse(r.use_rerank)
        self.assertTrue(r.rerank_skipped_due_to_query_family)

    def test_symptom_family_off_disables_rerank(self) -> None:
        r = _req(query_family="symptom_issue_extraction")
        apply_selective_rerank_policy(r, selective_enabled=True)
        self.assertFalse(r.use_rerank)
        self.assertIn("family_off", r.rerank_reason)

    def test_buyer_risk_family_on_enables_rerank(self) -> None:
        r = _req(query_family="buyer_risk_issues")
        apply_selective_rerank_policy(r, selective_enabled=True)
        self.assertTrue(r.use_rerank)
        self.assertIn("family_on", r.rerank_reason)

    def test_rating_metadata_disables_rerank(self) -> None:
        r = _req(query_family="value_complaint", filters={"review_rating": 1})
        apply_selective_rerank_policy(r, selective_enabled=True)
        self.assertFalse(r.use_rerank)
        self.assertTrue(r.rerank_skipped_due_to_metadata_filters)

    def test_selective_disabled_leaves_use_rerank_none(self) -> None:
        r = _req(query_family="value_complaint")
        apply_selective_rerank_policy(r, selective_enabled=False)
        self.assertIsNone(r.use_rerank)

    def test_explicit_use_rerank_not_overwritten(self) -> None:
        r = _req(query_family="rating_scoped_summary", use_rerank=True)
        apply_selective_rerank_policy(r, selective_enabled=True)
        self.assertTrue(r.use_rerank)

    def test_issue_keywords_strategy_off(self) -> None:
        r = _req(query_family="unknown", strategy_reason="issue_keywords")
        apply_selective_rerank_policy(r, selective_enabled=True)
        self.assertFalse(r.use_rerank)
        self.assertTrue(r.rerank_skipped_due_to_query_family)

    def test_family_override_for_eval(self) -> None:
        r = _req(query_text="raw", query_family=None)
        apply_selective_rerank_policy(
            r, selective_enabled=True, query_family_override="abstract_complaint_summary"
        )
        self.assertTrue(r.use_rerank)
        self.assertEqual(r.query_family, "abstract_complaint_summary")


class TestConfidenceGuard(unittest.TestCase):
    def test_skip_when_top_score_high_and_hybrid_columns(self) -> None:
        df = pd.DataFrame(
            [{"score": 0.99, "semantic_score": 0.5, "keyword_score": 0.2}]
        )
        with mock.patch.object(config, "RERANK_SKIP_IF_TOP_SCORE_AT_LEAST", 0.97):
            with mock.patch.object(
                config, "RERANK_CONFIDENCE_REQUIRES_HYBRID_SCORES", True
            ):
                self.assertTrue(should_skip_rerank_for_confidence(df))

    def test_no_skip_vector_only_frame(self) -> None:
        df = pd.DataFrame([{"score": 0.99}])
        with mock.patch.object(config, "RERANK_SKIP_IF_TOP_SCORE_AT_LEAST", 0.97):
            with mock.patch.object(
                config, "RERANK_CONFIDENCE_REQUIRES_HYBRID_SCORES", True
            ):
                self.assertFalse(should_skip_rerank_for_confidence(df))

    def test_no_skip_low_score(self) -> None:
        df = pd.DataFrame(
            [{"score": 0.5, "semantic_score": 0.3, "keyword_score": 0.2}]
        )
        with mock.patch.object(config, "RERANK_SKIP_IF_TOP_SCORE_AT_LEAST", 0.97):
            self.assertFalse(should_skip_rerank_for_confidence(df))


if __name__ == "__main__":
    unittest.main()
