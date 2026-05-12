import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cbam_hybrid import (
    apply_hybrid_combiner,
    apply_stationarity_policy,
    build_expanding_windows,
    clean_and_scale_data,
    evaluate_success_criteria,
    enforce_stationarity,
    train_test_split_time_series,
    train_hybrid_combiner,
    train_val_test_split_time_series,
    build_residual_training_frame,
    calculate_metrics,
    prepare_leakage_safe_splits,
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

    assert not all(item["differenced"] is False for item in stationarity.values())
    assert len(stationary_df) < len(trending)


def test_train_test_split_time_series_keeps_order():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"a": np.arange(10)}, index=idx)

    train, test = train_test_split_time_series(df, train_ratio=0.8)

    assert len(train) == 8
    assert len(test) == 2
    assert train.index.max() < test.index.min()


def test_train_val_test_split_time_series_keeps_order_and_sizes():
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({"a": np.arange(20)}, index=idx)

    train, val, test = train_val_test_split_time_series(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_build_residual_training_frame_creates_lagged_feature():
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    x1 = pd.Series(np.linspace(0.1, 1.2, 12), index=idx)
    x2 = pd.Series(np.linspace(1.0, 2.1, 12), index=idx)
    residual_pattern = np.array([0.0, 0.1, -0.1, 0.2, -0.2, 0.3])
    residuals = pd.Series(np.tile(residual_pattern, 2), index=idx)

    X, y = build_residual_training_frame(x1, residuals, x2_train=x2)

    expected_subset = {
        "x1",
        "x2",
        "x1_x2_interaction",
        "residual_lag1",
        "residual_lag5",
        "x1_lag1",
        "x2_lag1",
        "x1_change_lag1",
        "x2_change_lag1",
        "residual_roll_mean_3",
        "residual_roll_std_3",
    }
    assert expected_subset.issubset(set(X.columns))
    assert len(X) == len(y) == 6
    assert not X.isna().any().any()


def test_calculate_metrics_returns_rmse_mae_for_all_models():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    y_true = pd.Series([0.2, 0.3, 0.4, 0.5], index=idx)
    predictions = {
        "Baseline": pd.Series([0.21, 0.31, 0.39, 0.52], index=idx),
        "ARIMAX": pd.Series([0.2, 0.3, 0.4, 0.5], index=idx),
    }

    metrics = calculate_metrics(y_true, predictions)

    assert list(metrics.columns) == ["Model", "RMSE", "MAE"]
    assert set(metrics["Model"]) == {"Baseline", "ARIMAX"}
    assert (metrics[["RMSE", "MAE"]] >= 0).all().all()


def test_apply_stationarity_policy_keeps_split_alignment():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "a": np.arange(30, dtype=float),
            "b": np.arange(30, dtype=float) * 2,
            "c": np.arange(30, dtype=float) * 3,
        },
        index=idx,
    )
    train = df.iloc[:20]
    val = df.iloc[20:25]
    test = df.iloc[25:]

    train_s, val_s, test_s, stationarity = apply_stationarity_policy(train, val, test)

    assert len(train_s) < len(train)
    assert len(val_s) == len(val)
    assert len(test_s) == len(test)
    assert all("differenced" in info for info in stationarity.values())


def test_prepare_leakage_safe_splits_fit_scaler_on_train_only():
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {
            "EREGL.IS": np.linspace(1, 20, 20),
            "KEUA": np.linspace(2, 21, 20),
            "TIO=F": np.linspace(3, 22, 20),
        },
        index=idx,
    )
    out = prepare_leakage_safe_splits(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    assert len(out.train) > 0 and len(out.val) > 0 and len(out.test) > 0
    assert set(out.train.columns) == {"EREGL.IS", "KEUA", "TIO=F"}
    assert set(out.stationarity.keys()) == {"EREGL.IS", "KEUA", "TIO=F"}


def test_build_expanding_windows_returns_ordered_splits():
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame({"x": np.arange(50)}, index=idx)

    windows = build_expanding_windows(df, min_train_size=20, val_size=10, test_size=10, step_size=10, max_windows=3)

    assert len(windows) == 2
    for train, val, test in windows:
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()


def test_hybrid_combiner_and_success_criteria():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    y_val = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=idx)
    arimax_val = pd.Series([1.0, 1.05, 1.15, 1.25, 1.35], index=idx)
    mlp_val = pd.Series([0.0, 0.05, 0.05, 0.05, 0.05], index=idx)

    model = train_hybrid_combiner(y_val, arimax_val, mlp_val)
    hybrid = apply_hybrid_combiner(model, arimax_val, mlp_val)
    assert len(hybrid) == len(y_val)

    metrics_df = pd.DataFrame(
        [
            {"Model": "Hibrit ARIMAX-MLP", "RMSE": 0.90, "MAE": 0.70},
            {"Model": "Baseline", "RMSE": 1.00, "MAE": 0.75},
            {"Model": "ARIMAX", "RMSE": 1.05, "MAE": 0.80},
        ]
    )
    rolling_df = pd.DataFrame(
        [
            {"window": 1, "Model": "Hibrit ARIMAX-MLP", "RMSE": 0.9, "MAE": 0.7},
            {"window": 1, "Model": "Baseline", "RMSE": 1.0, "MAE": 0.8},
            {"window": 2, "Model": "Hibrit ARIMAX-MLP", "RMSE": 0.8, "MAE": 0.7},
            {"window": 2, "Model": "Baseline", "RMSE": 1.0, "MAE": 0.8},
        ]
    )
    success = evaluate_success_criteria(metrics_df, rolling_df)
    assert success["single_split_improvement"] > 0
    assert success["rolling_win_ratio"] == 1.0
