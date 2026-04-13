from __future__ import annotations

import unittest

from src.retrieval_metrics import (
    aggregate_mean,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


class TestRetrievalMetrics(unittest.TestCase):
    def test_precision_at_k(self) -> None:
        gold = {"a", "b"}
        self.assertEqual(precision_at_k(["a", "c", "d"], gold, 2), 0.5)
        self.assertEqual(precision_at_k([], gold, 5), 0.0)

    def test_recall_at_k(self) -> None:
        gold = {"a", "b", "c"}
        self.assertEqual(recall_at_k(["a", "x"], gold, 5), 1.0 / 3.0)

    def test_mrr(self) -> None:
        self.assertEqual(mean_reciprocal_rank(["x", "a", "b"], {"a"}), 0.5)
        self.assertEqual(mean_reciprocal_rank(["x", "y"], {"a"}), 0.0)

    def test_aggregate_mean(self) -> None:
        self.assertEqual(aggregate_mean([1.0, None, 3.0]), 2.0)


if __name__ == "__main__":
    unittest.main()
