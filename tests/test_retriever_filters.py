"""Unit tests for metadata filter helpers (pandas only)."""
from __future__ import annotations

import unittest

import pandas as pd

from src.metadata_filters import apply_metadata_filters


class TestMetadataFilters(unittest.TestCase):
    def test_unknown_filter_keys_ignored(self) -> None:
        df = pd.DataFrame({"foo": [1], "score": [0.9]})
        out = apply_metadata_filters(df, {"missing_col": "x"})
        self.assertEqual(len(out), 1)

    def test_equality_string(self) -> None:
        df = pd.DataFrame(
            {"brand": ["Acme", "Beta"], "score": [0.5, 0.9]},
        )
        out = apply_metadata_filters(df, {"brand": "acme"})
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["brand"], "Acme")

    def test_range_max(self) -> None:
        df = pd.DataFrame({"rating": [1.0, 4.0, 5.0], "score": [0.1, 0.2, 0.3]})
        out = apply_metadata_filters(df, {"rating": {"max": 3}})
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out.iloc[0]["rating"]), 1.0)

    def test_range_min_max(self) -> None:
        df = pd.DataFrame({"rating": [1.0, 3.0, 5.0], "score": [0.1, 0.2, 0.3]})
        out = apply_metadata_filters(df, {"rating": {"min": 2, "max": 4}})
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out.iloc[0]["rating"]), 3.0)


if __name__ == "__main__":
    unittest.main()
