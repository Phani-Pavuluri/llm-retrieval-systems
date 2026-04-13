"""Unit tests for scripts/analyze_rerank_impact helpers (no trace file I/O)."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_analyze_module():  # type: ignore[no-untyped-def]
    path = _ROOT / "scripts" / "analyze_rerank_impact.py"
    spec = importlib.util.spec_from_file_location("analyze_rerank_impact", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ar = _load_analyze_module()


class TestRerankPrePostCategory(unittest.TestCase):
    def test_fixed(self) -> None:
        row = {
            "eval_mode": "auto_rerank",
            "gold_chunk_ids": ["g1"],
            "gold_in_pre_rerank_top_k": False,
            "gold_in_post_rerank_top_k": True,
        }
        self.assertEqual(ar._rerank_pre_post_category(row), "fixed")

    def test_both_false(self) -> None:
        row = {
            "eval_mode": "auto_rerank",
            "gold_chunk_ids": ["g1"],
            "gold_in_pre_rerank_top_k": False,
            "gold_in_post_rerank_top_k": False,
        }
        self.assertEqual(ar._rerank_pre_post_category(row), "both_false")


class TestRerankGoldMovement(unittest.TestCase):
    def test_helped(self) -> None:
        row = {
            "eval_mode": "auto_rerank",
            "gold_chunk_ids": ["g1"],
            "gold_in_pre_rerank_top_k": False,
            "gold_in_post_rerank_top_k": True,
        }
        self.assertEqual(ar._rerank_gold_movement(row), "helped")

    def test_hurt(self) -> None:
        row = {
            "eval_mode": "hybrid_rerank",
            "gold_chunk_ids": ["g1"],
            "gold_in_pre_rerank_top_k": True,
            "gold_in_post_rerank_top_k": False,
        }
        self.assertEqual(ar._rerank_gold_movement(row), "hurt")

    def test_non_rerank_mode_none(self) -> None:
        row = {
            "eval_mode": "auto",
            "gold_chunk_ids": ["g1"],
            "gold_in_pre_rerank_top_k": False,
            "gold_in_post_rerank_top_k": True,
        }
        self.assertIsNone(ar._rerank_gold_movement(row))

    def test_selective_rerank_mode_counts(self) -> None:
        row = {
            "eval_mode": "auto_selective_rerank",
            "gold_chunk_ids": ["g1"],
            "gold_in_pre_rerank_top_k": True,
            "gold_in_post_rerank_top_k": True,
        }
        self.assertEqual(ar._rerank_gold_movement(row), "unchanged")


class TestModeStats(unittest.TestCase):
    def test_mode_stats_empty_mode(self) -> None:
        rows = [{"eval_mode": "vector", "gold_chunk_ids": ["a"], "final_top_chunk_ids": ["a"]}]
        p, r, m, n = ar._mode_stats(rows, "hybrid", 5)
        self.assertEqual(n, 0)
        self.assertIsNone(p)
        self.assertIsNone(r)


class TestSelectiveValidationBuilders(unittest.TestCase):
    def test_comparison_and_policy_text_non_empty(self) -> None:
        rows = [
            {
                "eval_mode": "auto",
                "eval_id": "q1",
                "query_family": "value_complaint",
                "query": "q",
                "gold_chunk_ids": ["g1"],
                "requested_top_k": 5,
                "final_top_chunk_ids": ["g1", "x"],
                "gold_in_pre_rerank_top_k": True,
                "gold_in_post_rerank_top_k": True,
            },
            {
                "eval_mode": "auto_rerank",
                "eval_id": "q1",
                "query_family": "value_complaint",
                "query": "q",
                "gold_chunk_ids": ["g1"],
                "requested_top_k": 5,
                "final_top_chunk_ids": ["g1", "x"],
                "gold_in_pre_rerank_top_k": True,
                "gold_in_post_rerank_top_k": False,
                "pre_rerank_top_chunk_ids": ["g1"],
                "post_rerank_top_chunk_ids": ["x"],
            },
            {
                "eval_mode": "auto_selective_rerank",
                "eval_id": "q1",
                "query_family": "value_complaint",
                "query": "q",
                "gold_chunk_ids": ["g1"],
                "requested_top_k": 5,
                "final_top_chunk_ids": ["g1", "x"],
                "rerank_applied": False,
                "rerank_skipped_due_to_query_family": True,
                "gold_in_pre_rerank_top_k": True,
                "gold_in_post_rerank_top_k": True,
            },
        ]
        cmp_txt = ar.build_selective_rerank_comparison_text(rows, 5)
        self.assertIn("auto_selective_rerank vs auto_rerank", cmp_txt)
        pol_txt = ar.build_selective_rerank_policy_stats_text(rows)
        self.assertIn("Total queries", pol_txt)
        self.assertIn("auto_selective_rerank", pol_txt)
        fam_txt = ar.build_selective_rerank_by_query_family_text(rows)
        self.assertIn("value_complaint", fam_txt)
        ex_txt = ar.build_selective_rerank_examples_text(rows, 5)
        self.assertIn("representative examples", ex_txt)


if __name__ == "__main__":
    unittest.main()
