from __future__ import annotations

import unittest
from unittest.mock import patch

from src.retrieval_request import RetrievalRequest
from src.retriever import Retriever


class TestRetrieverPool(unittest.TestCase):
    def setUp(self) -> None:
        self.r = object.__new__(Retriever)

    def test_candidate_count_config(self) -> None:
        req = RetrievalRequest(query_text="q", top_k=5)
        with (
            patch("src.retriever.config.RETRIEVAL_OVERSAMPLE_MULTIPLIER", 5),
            patch("src.retriever.config.RETRIEVAL_MIN_CANDIDATES", 20),
        ):
            self.assertEqual(Retriever._candidate_count(self.r, req), 25)

    def test_candidate_count_respects_multiplier(self) -> None:
        req = RetrievalRequest(
            query_text="q", top_k=5, candidate_pool_multiplier=2.0
        )
        with (
            patch("src.retriever.config.RETRIEVAL_OVERSAMPLE_MULTIPLIER", 5),
            patch("src.retriever.config.RETRIEVAL_MIN_CANDIDATES", 20),
        ):
            self.assertEqual(Retriever._candidate_count(self.r, req), 50)

    def test_candidate_count_uses_rerank_top_n_when_rerank_on(self) -> None:
        req = RetrievalRequest(
            query_text="q",
            top_k=5,
            use_rerank=True,
            rerank_top_n=30,
        )
        with (
            patch("src.retriever.config.RETRIEVAL_OVERSAMPLE_MULTIPLIER", 5),
            patch("src.retriever.config.RETRIEVAL_MIN_CANDIDATES", 20),
        ):
            # max(5, 30) * 5 = 150 vs 20 -> 150
            self.assertEqual(Retriever._candidate_count(self.r, req), 150)


if __name__ == "__main__":
    unittest.main()
