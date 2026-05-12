"""Reporting layer utilities."""

from __future__ import annotations

import pandas as pd


def first_breakpoint_report(module_breakdown: pd.DataFrame, threshold: float) -> str:
    if module_breakdown.empty:
        return "Kırılma analizi yapılamadı: modül metriği boş."
    over = module_breakdown[module_breakdown["rmse"] > threshold]
    if over.empty:
        return f"Kırılma yok: tüm modüller RMSE<{threshold:.3f} bandında."
    first = over.iloc[0]
    return f"İlk kırılma noktası: {first['module']} (RMSE={float(first['rmse']):.4f}, eşik={threshold:.4f})"
