"""Tests for Phase 5.1 explanation_builder (no LLM)."""
from __future__ import annotations

import unittest

import pandas as pd

from src.explanation_builder import build_explanation_payload
from src.retrieval_request import RetrievalRequest


class TestExplanationBuilder(unittest.TestCase):
    def test_empty_retrieval_low_confidence(self) -> None:
        req = RetrievalRequest(
            query_text="q",
            top_k=5,
            task_type="general_qa",
            query_family="unknown",
        )
        diag = {"underfilled_after_filtering": True, "rerank_decision": {}}
        ans = "Insufficient evidence in the retrieved excerpts."
        ex = build_explanation_payload(
            request=req,
            retrieved=pd.DataFrame(),
            answer=ans,
            chunk_ids_used=[],
            prompt_template_id="task_general_qa",
            diagnostics=diag,
        )
        self.assertTrue(ans.lower().startswith("insufficient evidence"))
        self.assertEqual(ex["evidence"], [])
        self.assertEqual(ex["confidence"]["confidence_label"], "low")

    def test_evidence_scores_and_reasoning(self) -> None:
        req = RetrievalRequest(
            query_text="rash complaints",
            top_k=2,
            task_type="complaint_summary",
            query_family="exact_issue_lookup",
            filters={},
        )
        df = pd.DataFrame(
            [
                {
                    "chunk_id": "A_1_0",
                    "asin": "B00TEST",
                    "text": "Got a rash from use.",
                    "score": 0.4,
                    "semantic_score": 0.5,
                    "keyword_score": 0.2,
                    "rerank_score": 0.9,
                    "final_rank": 1,
                    "retrieval_score": 0.4,
                },
                {
                    "chunk_id": "A_2_0",
                    "asin": "B00TEST",
                    "text": "Burning sensation.",
                    "score": 0.35,
                    "semantic_score": 0.45,
                    "keyword_score": 0.1,
                    "rerank_score": 0.7,
                    "final_rank": 2,
                    "retrieval_score": 0.35,
                },
            ]
        )
        diag = {
            "underfilled_after_filtering": False,
            "rerank_decision": {
                "use_rerank_requested": True,
                "use_rerank_effective": True,
                "rerank_skipped_due_to_query_family": False,
                "rerank_skipped_due_to_metadata_filters": False,
                "rerank_skipped_due_to_confidence": False,
            },
        }
        ex = build_explanation_payload(
            request=req,
            retrieved=df,
            answer="Some reviewers report rash and burning.",
            chunk_ids_used=["A_1_0", "A_2_0"],
            prompt_template_id="family_exact_issue_lookup",
            diagnostics=diag,
        )
        self.assertEqual(len(ex["evidence"]), 2)
        self.assertEqual(ex["evidence"][0]["chunk_id"], "A_1_0")
        self.assertEqual(ex["evidence"][0]["source_id"], "B00TEST")
        self.assertEqual(ex["evidence"][0]["rank_position"], 1)
        self.assertTrue(ex["reasoning_summary"]["rerank_applied"])
        self.assertIn("semantic_score", ex["evidence"][0])


if __name__ == "__main__":
    unittest.main()
