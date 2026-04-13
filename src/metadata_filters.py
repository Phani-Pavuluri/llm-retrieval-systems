"""Dataset-agnostic metadata post-filtering for retrieval results."""
from __future__ import annotations

from typing import Any

import pandas as pd


def apply_metadata_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """
    Apply equality or min/max range filters. Unknown columns are skipped.
    Range spec: {"min": optional, "max": optional} (numeric columns).
    """
    if not filters:
        return df

    out = df
    for key, value in filters.items():
        if key not in out.columns:
            continue
        if isinstance(value, dict) and ("min" in value or "max" in value):
            out = _apply_range_filter(out, key, value)
        else:
            out = _apply_equality_filter(out, key, value)
    return out


def _apply_range_filter(
    df: pd.DataFrame, col: str, spec: dict[str, Any]
) -> pd.DataFrame:
    series = pd.to_numeric(df[col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if spec.get("min") is not None:
        mask &= series >= float(spec["min"])
    if spec.get("max") is not None:
        mask &= series <= float(spec["max"])
    return df.loc[mask]


def _apply_equality_filter(df: pd.DataFrame, col: str, value: Any) -> pd.DataFrame:
    col_series = df[col]
    numeric = pd.to_numeric(col_series, errors="coerce")
    if numeric.notna().all():
        try:
            target = float(value)
            return df.loc[numeric == target]
        except (TypeError, ValueError):
            pass
    return df.loc[
        col_series.astype(str).str.strip().str.lower()
        == str(value).strip().lower()
    ]
