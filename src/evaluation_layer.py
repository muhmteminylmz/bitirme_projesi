"""Evaluation layer utilities for official acceptance criteria."""

from __future__ import annotations

from typing import Dict


def acceptance_summary(
    official_metric: str,
    support_metric: str,
    stable_rmse_upper: float,
    success_pass: bool,
) -> Dict[str, float | bool | str]:
    return {
        "official_metric": official_metric,
        "support_metric": support_metric,
        "stable_rmse_upper": stable_rmse_upper,
        "pass": bool(success_pass),
    }
