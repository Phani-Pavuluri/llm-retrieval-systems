from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src import config
from src.retrieval_with_rerank import retrieve_with_optional_rerank
from src.reranker import (
    CrossEncoderReranker,
    apply_rerank_to_candidates,
    effective_rerank,
    effective_rerank_model,
)
from src.retrieval_request import RetrievalRequest
from src.retrieval_trace import build_retrieval_trace_record


class _FakeRetriever:
    def __init__(self, df_pre_rerank: pd.DataFrame) -> None:
        self._df = df_pre_rerank
        self.last_trace_build_context: dict | None = None
        self.last_retrieval_diagnostics: dict | None = None

    def retrieve(
        self,
        request: RetrievalRequest,
        trace_extra: dict | None = None,
        write_trace: bool | None = None,
    ) -> pd.DataFrame:
        df = self._df.copy()
        self.last_trace_build_context = {
            "df_pre_filter": df,
            "df_post_filter": df,
            "df_pre_rerank": df,
            "k_fetch_requested": 10,
        }
        self.last_retrieval_diagnostics = {
            "candidate_pool_size": len(df),
            "post_filter_count": len(df),
            "final_returned_count": len(df),
            "underfilled_after_filtering": False,
            "requested_top_k": request.top_k,
            "rerank_candidate_count": len(df),
        }
        return df


class TestCrossEncoderReranker(unittest.TestCase):
    def test_rerank_reorders_by_mock_scores(self) -> None:
        df = pd.DataFrame(
            {
                "chunk_id": ["a", "b", "c"],
                "text": ["x", "y", "z"],
                "score": [0.9, 0.5, 0.7],
            }
        )
        rer = CrossEncoderReranker(model_name="dummy")
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.3]
        rer._model = mock_model
        out = rer.rerank("q", df)
        self.assertEqual(out.iloc[0]["chunk_id"], "b")
        self.assertEqual(list(out["rerank_score"]), [0.9, 0.3, 0.1])

    def test_apply_rerank_preserves_metadata_columns(self) -> None:
        df = pd.DataFrame(
            {
                "chunk_id": ["a", "b"],
                "text": ["hello", "world"],
                "score": [0.2, 0.8],
                "brand": ["X", "Y"],
            }
        )
        rer = CrossEncoderReranker(model_name="dummy")
        mock_model = MagicMock()
        mock_model.predict.return_value = [1.0, 0.0]
        rer._model = mock_model
        out = apply_rerank_to_candidates(rer, "q", df, final_top_k=2)
        self.assertIn("brand", out.columns)
        self.assertEqual(out.iloc[0]["brand"], "X")
        self.assertIn("rerank_score", out.columns)
        self.assertEqual(list(out["final_rank"]), [1, 2])


class TestPipelineRerank(unittest.TestCase):
    def test_use_rerank_false_skips_second_stage(self) -> None:
        df = pd.DataFrame(
            {
                "chunk_id": ["a", "b"],
                "text": ["t1", "t2"],
                "score": [0.9, 0.1],
            }
        )
        fake = _FakeRetriever(df)
        req = RetrievalRequest(
            query_text="q",
            top_k=2,
            use_rerank=False,
        )
        mock_rer = MagicMock()
        out = retrieve_with_optional_rerank(fake, req, trace_extra=None, reranker=mock_rer)
        mock_rer.rerank.assert_not_called()
        self.assertEqual(len(out), 2)
        self.assertNotIn("rerank_score", out.columns)

    def test_trace_record_includes_rerank_when_applied(self) -> None:
        req = RetrievalRequest(query_text="q", top_k=2, use_rerank=True)
        df_pre = pd.DataFrame(
            {
                "chunk_id": ["a", "b"],
                "text": ["x", "y"],
                "score": [0.5, 0.6],
                "retrieval_score": [0.5, 0.6],
            }
        )
        df_final = pd.DataFrame(
            {
                "chunk_id": ["b", "a"],
                "text": ["y", "x"],
                "score": [0.6, 0.5],
                "retrieval_score": [0.6, 0.5],
                "rerank_score": [9.0, 1.0],
                "final_rank": [1, 2],
            }
        )
        rec = build_retrieval_trace_record(
            req,
            {"gold_chunk_ids": ["a"]},
            df_pre,
            df_pre,
            df_pre,
            df_final,
            10,
            use_rerank_effective=True,
            rerank_top_n_effective=25,
            rerank_applied=True,
            rerank_model="cross-encoder/test",
        )
        self.assertTrue(rec["use_rerank"])
        self.assertTrue(rec["rerank_applied"])
        self.assertEqual(rec["rerank_model"], "cross-encoder/test")
        self.assertEqual(rec["rerank_candidate_count"], 2)
        self.assertEqual(rec["pre_rerank_top_chunk_ids"], ["a", "b"])
        self.assertEqual(rec["post_rerank_top_chunk_ids"], ["b", "a"])
        self.assertIn("rerank_score", rec["hits"][0])
        self.assertIn("gold_in_pre_rerank_top_k", rec)
        self.assertIn("gold_in_post_rerank_top_k", rec)

    def test_mocked_rerank_improves_ordering(self) -> None:
        df = pd.DataFrame(
            {
                "chunk_id": ["bad", "good"],
                "text": ["nope", "yes"],
                "score": [0.99, 0.1],
                "retrieval_score": [0.99, 0.1],
            }
        )
        fake = _FakeRetriever(df)
        req = RetrievalRequest(query_text="q", top_k=1, use_rerank=True)
        rer = CrossEncoderReranker(model_name="dummy")
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.0, 1.0]
        rer._model = mock_model
        with patch("src.retrieval_with_rerank.emit_retrieval_trace_record"):
            out = retrieve_with_optional_rerank(fake, req, trace_extra=None, reranker=rer)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["chunk_id"], "good")
        self.assertGreater(float(out.iloc[0]["rerank_score"]), 0.5)


class TestEffectiveRerank(unittest.TestCase):
    def test_explicit_false_overrides_config(self) -> None:
        req = RetrievalRequest(query_text="q", top_k=3, use_rerank=False)
        with patch.object(config, "RERANK_ENABLED", True):
            self.assertFalse(effective_rerank(req))


class TestEffectiveRerankModel(unittest.TestCase):
    def test_request_override(self) -> None:
        req = RetrievalRequest(
            query_text="q",
            top_k=3,
            rerank_model="cross-encoder/custom",
        )
        with patch.object(config, "RERANK_MODEL", "cross-encoder/default"):
            self.assertEqual(effective_rerank_model(req), "cross-encoder/custom")

    def test_falls_back_to_config(self) -> None:
        req = RetrievalRequest(query_text="q", top_k=3)
        with patch.object(config, "RERANK_MODEL", "cross-encoder/default"):
            self.assertEqual(effective_rerank_model(req), "cross-encoder/default")


if __name__ == "__main__":
    unittest.main()
