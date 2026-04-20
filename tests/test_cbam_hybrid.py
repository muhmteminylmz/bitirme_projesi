import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cbam_hybrid import (
    clean_and_scale_data,
    enforce_stationarity,
    train_test_split_time_series,
    build_residual_training_frame,
)


def test_clean_and_scale_data_ffill_bfill_and_range():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "Y_EREGL": [np.nan, 10.0, 11.0, np.nan, 13.0, 14.0],
            "X1_KEA": [1.0, np.nan, 2.0, 3.0, np.nan, 5.0],
            "X2_TIO": [50.0, 51.0, np.nan, 53.0, 54.0, np.nan],
        },
        index=idx,
    )

    scaled_df, _ = clean_and_scale_data(df)

    assert not scaled_df.isna().any().any()
    assert (scaled_df.min() >= 0).all()
    assert (scaled_df.max() <= 1).all()


def test_enforce_stationarity_applies_diff_when_needed():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    trending = pd.DataFrame(
        {
            "Y_EREGL": np.arange(40, dtype=float),
            "X1_KEA": np.arange(40, dtype=float) * 2,
            "X2_TIO": np.arange(40, dtype=float) * 3,
        },
        index=idx,
    )

    stationary_df, stationarity = enforce_stationarity(trending)

    assert not all(stationarity.values())
    assert len(stationary_df) < len(trending)


def test_train_test_split_time_series_keeps_order():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"a": np.arange(10)}, index=idx)

    train, test = train_test_split_time_series(df, train_ratio=0.8)

    assert len(train) == 8
    assert len(test) == 2
    assert train.index.max() < test.index.min()


def test_build_residual_training_frame_creates_lagged_feature():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    x1 = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], index=idx)
    residuals = pd.Series([0.0, 0.1, -0.1, 0.2, -0.2, 0.3], index=idx)

    X, y = build_residual_training_frame(x1, residuals)

    assert list(X.columns) == ["x1", "residual_lag1"]
    assert len(X) == len(y) == 5
    assert not X.isna().any().any()
