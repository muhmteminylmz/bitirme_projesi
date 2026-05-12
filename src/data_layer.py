"""Data layer utilities for quality checks and split integrity."""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

MAX_ACCEPTABLE_JUMP_RATIO = 0.20


def quality_gate(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    max_missing_ratio: float,
    max_daily_return: float,
) -> Dict[str, float | bool]:
    if df.empty:
        raise ValueError("Data quality gate failed: raw DataFrame is empty.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Data quality gate failed: index is not monotonic increasing.")
    if not df.index.is_unique:
        raise ValueError("Data quality gate failed: duplicated timestamps detected.")
    required = set(required_columns)
    if set(df.columns) != required:
        raise ValueError(f"Data quality gate failed: columns must be exactly {sorted(required)}.")

    missing_ratio = float(df.isna().mean().max())
    if missing_ratio > max_missing_ratio:
        raise ValueError(f"Data quality gate failed: missing ratio {missing_ratio:.4f} exceeds threshold {max_missing_ratio:.4f}.")
    non_positive_cols = df.columns[(df <= 0).any()].tolist()
    if non_positive_cols:
        raise ValueError(f"Data quality gate failed: non-positive prices in columns: {non_positive_cols}")

    jump_ratio = float((df.pct_change().abs() > max_daily_return).mean().max())
    return {
        "missing_ratio_max": missing_ratio,
        "jump_ratio_max": jump_ratio,
        "quality_pass": jump_ratio < MAX_ACCEPTABLE_JUMP_RATIO,
    }


def assert_split_integrity(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Split integrity failed: one of train/val/test is empty.")
    if train_df.index.max() >= val_df.index.min():
        raise ValueError("Split integrity failed: train must end before validation starts.")
    if val_df.index.max() >= test_df.index.min():
        raise ValueError("Split integrity failed: validation must end before test starts.")
    if len(train_df.index.intersection(val_df.index)) > 0:
        raise ValueError("Split integrity failed: overlap between train and validation.")
    if len(val_df.index.intersection(test_df.index)) > 0:
        raise ValueError("Split integrity failed: overlap between validation and test.")
    if len(train_df.index.intersection(test_df.index)) > 0:
        raise ValueError("Split integrity failed: overlap between train and test.")
