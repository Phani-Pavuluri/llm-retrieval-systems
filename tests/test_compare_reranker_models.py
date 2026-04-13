"""Lightweight tests for compare_reranker_models helpers (no full eval)."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]


def _load_compare():
    path = _ROOT / "scripts" / "compare_reranker_models.py"
    spec = importlib.util.spec_from_file_location("compare_reranker_models", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cm = _load_compare()


class TestTraceSlug(unittest.TestCase):
    def test_slug_stable(self) -> None:
        s = cm.trace_slug("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertNotIn("/", s)
        self.assertTrue(s.startswith("cross"))


class TestDedupeModels(unittest.TestCase):
    def test_dedupe(self) -> None:
        self.assertEqual(
            cm._dedupe_preserve(["a", "a", " b ", "b"]),
            ["a", "b"],
        )


class TestTryLoadGraceful(unittest.TestCase):
    @patch("sentence_transformers.CrossEncoder", side_effect=RuntimeError("mock load fail"))
    def test_try_load_returns_false_on_error(self, _mock_ce: object) -> None:
        from src.reranker import try_load_cross_encoder

        ok, err = try_load_cross_encoder("any/model-id")
        self.assertFalse(ok)
        self.assertIn("RuntimeError", err or "")


if __name__ == "__main__":
    unittest.main()
