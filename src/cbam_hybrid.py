"""ARIMAX + MLP hybrid pipeline for corporate carbon stress testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yfinance as yf
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
sns.set_style("whitegrid")

TARGET_TICKER = "EREGL.IS"
CARBON_TICKER = "KEUA"
IRON_TICKER = "TIO=F"
FX_TICKER = "USDTRY=X"
XGBOOST_PARAMS = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5}
MLP_PARAMS = {"epochs": 300, "batch_size": 16, "validation_split": 0.0, "patience": 20, "learning_rate": 0.0005}
SUCCESS_MIN_SINGLE_SPLIT_IMPROVEMENT = 0.03
SUCCESS_MIN_ROLLING_WIN_RATIO = 0.60
SUCCESS_MIN_ROLLING_MEAN_IMPROVEMENT = 0.0
OFFICIAL_METRIC = "RMSE"
SUPPORT_METRIC = "MAE"
STABLE_RMSE_BAND_UPPER = 0.35
DEFAULT_SEED = 42
REPEATED_SEEDS = [42, 123, 2024]
MAX_ALLOWED_DAILY_RETURN = 0.25
RESIDUAL_LAG_COUNT = 5
RESIDUAL_ROLL_WINDOW = 3
ARIMAX_P_RANGE = range(0, 4)
ARIMAX_D_RANGE = range(0, 2)
ARIMAX_Q_RANGE = range(0, 4)
ROLLING_MIN_TRAIN_SIZE_RATIO = 0.5
ROLLING_VAL_SIZE_RATIO = 0.15
ROLLING_TEST_SIZE_RATIO = 0.10
ROLLING_MIN_TRAIN_ROWS = 40
ROLLING_MIN_VAL_ROWS = 20
ROLLING_MIN_TEST_ROWS = 20
MODEL_COLORS = {
    "Baseline": "#6c7a89",
    "ARIMAX": "#4c78a8",
    "XGBoost": "#72b7b2",
    "Hibrit ARIMAX-MLP": "#8b0000",
}
DEFAULT_MODEL_COLOR = "#808080"
STRESS_BANDS = {"S1 (+%30)": 0.30, "S2 (+%60)": 0.60, "S3 (+%100)": 1.00}


SuccessSummary = TypedDict(
    "SuccessSummary",
    {
        "single_split_improvement": float,
        "rolling_win_ratio": float,
        "rolling_mean_improvement": float,
        "single_split_rmse": float,
        "rolling_hybrid_rmse_mean": float,
        "pass": bool,
        "decision_reason": str,
    },
)


@dataclass
class PipelineResult:
    metrics_table: pd.DataFrame
    predictions: pd.DataFrame
    residuals: pd.DataFrame
    diagnostics: pd.DataFrame
    stress_table: pd.DataFrame
    rolling_metrics: pd.DataFrame
    success_summary: SuccessSummary
    acceptance_summary: Dict[str, float | bool | str]
    diagnostic_baseline: pd.DataFrame
    module_breakdown: pd.DataFrame
    ablation_table: pd.DataFrame
    root_cause_report: str


@dataclass
class PreprocessOutput:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scaler: MinMaxScaler
    stationarity: Dict[str, Dict[str, float]]


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def validate_data_quality(df: pd.DataFrame, max_missing_ratio: float = 0.05, max_daily_return: float = MAX_ALLOWED_DAILY_RETURN) -> Dict[str, float | bool]:
    if df.empty:
        raise ValueError("Data quality gate failed: raw dataframe is empty.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Data quality gate failed: index is not monotonic increasing.")
    if not df.index.is_unique:
        raise ValueError("Data quality gate failed: duplicated timestamps detected.")
    expected_cols = {TARGET_TICKER, CARBON_TICKER, IRON_TICKER, FX_TICKER}
    if set(df.columns) != expected_cols:
        raise ValueError(f"Data quality gate failed: columns must be exactly {sorted(expected_cols)}.")

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
        "quality_pass": jump_ratio < 0.20,
    }


def validate_split_integrity(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
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


def fetch_yfinance_data(
    start: str = "2021-04-01",
    end: str = "2026-04-25",
    interval: str = "1d",
) -> pd.DataFrame:
    tickers = [TARGET_TICKER, CARBON_TICKER, IRON_TICKER, FX_TICKER]
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise ValueError(f"yfinance veri çekimi boş döndü (tickerlar: {tickers}). Ağ bağlantısı veya ticker geçerliliğini kontrol edin.")

    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw["Close"].copy()
    else:
        close_df = raw.rename(columns={"Close": tickers[0]})[[tickers[0]]]

    close_df = close_df.reindex(columns=tickers)
    business_index = pd.date_range(close_df.index.min(), close_df.index.max(), freq="B")
    close_df = close_df.reindex(business_index).ffill().bfill()

    if close_df.isna().any().any():
        missing_cols = close_df.columns[close_df.isna().any()].tolist()
        raise ValueError(f"Veri ffill/bfill sonrasında hâlâ eksik değer içeriyor: {missing_cols}")
    invalid_fx = (close_df[FX_TICKER] <= 0).any()
    if invalid_fx:
        raise ValueError(f"{FX_TICKER} serisinde sıfır veya negatif değer bulundu, döviz dönüşümü yapılamadı.")

    standardized = pd.DataFrame(index=close_df.index)
    standardized[TARGET_TICKER] = close_df[TARGET_TICKER] / close_df[FX_TICKER]
    standardized[CARBON_TICKER] = close_df[CARBON_TICKER]
    standardized[IRON_TICKER] = close_df[IRON_TICKER]
    standardized[FX_TICKER] = close_df[FX_TICKER]
    return standardized


def clean_and_scale_data(df: pd.DataFrame, fit_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    cleaned = df.copy().sort_index().ffill().bfill()
    fit_cleaned = cleaned if fit_df is None else fit_df.copy().sort_index().ffill().bfill()
    scaler = MinMaxScaler()
    scaler.fit(fit_cleaned)
    scaled = pd.DataFrame(scaler.transform(cleaned), columns=cleaned.columns, index=cleaned.index)
    return scaled, scaler


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    if (df <= 0).any().any():
        bad_cols = df.columns[(df <= 0).any()].tolist()
        raise ValueError(f"Logaritmik getiri için tüm sütunlar pozitif olmalı. Sorunlu sütunlar: {bad_cols}")
    return np.log(df / df.shift(1)).dropna()


def safe_adf_pvalue(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty or s.nunique() <= 1:
        return 1.0
    return float(adfuller(s)[1])


def enforce_stationarity(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    stationary_df = df.copy()
    adf_results: Dict[str, Dict[str, float]] = {}

    for col in df.columns:
        initial_p = safe_adf_pvalue(df[col])
        print(f"{col} - ADF p-değeri: {initial_p:.6f}")

        if initial_p > 0.05:
            stationary_df[col] = df[col].diff()
            final_p = safe_adf_pvalue(stationary_df[col])
            print(f"  -> Birinci fark alındı, yeni ADF p-değeri: {final_p:.6f}")
            adf_results[col] = {"initial_p": initial_p, "differenced": True, "final_p": final_p}
        else:
            print("  -> Seri durağan, fark alınmadı.")
            adf_results[col] = {"initial_p": initial_p, "differenced": False, "final_p": initial_p}

    return stationary_df.dropna(), adf_results


def apply_stationarity_policy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()
    stationarity: Dict[str, Dict[str, float]] = {}

    for col in train_df.columns:
        initial_p = safe_adf_pvalue(train_df[col])
        differenced = initial_p > 0.05

        if differenced:
            train_out[col] = train_df[col].diff()
            val_out[col] = pd.concat([train_df[col].tail(1), val_df[col]]).diff().iloc[1:]
            test_out[col] = pd.concat([val_df[col].tail(1), test_df[col]]).diff().iloc[1:]
            final_p = safe_adf_pvalue(train_out[col])
        else:
            final_p = initial_p

        stationarity[col] = {"initial_p": initial_p, "differenced": differenced, "final_p": final_p}

    return train_out.dropna(), val_out.dropna(), test_out.dropna(), stationarity


def prepare_leakage_safe_splits(
    df_raw: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> PreprocessOutput:
    train_raw, val_raw, test_raw = train_val_test_split_time_series(df_raw, train_ratio, val_ratio, test_ratio)
    validate_split_integrity(train_raw, val_raw, test_raw)

    train_clean = train_raw.sort_index().ffill().bfill()
    val_clean = val_raw.sort_index().ffill().bfill()
    test_clean = test_raw.sort_index().ffill().bfill()

    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_clean), columns=train_clean.columns, index=train_clean.index)
    val_scaled = pd.DataFrame(scaler.transform(val_clean), columns=val_clean.columns, index=val_clean.index)
    test_scaled = pd.DataFrame(scaler.transform(test_clean), columns=test_clean.columns, index=test_clean.index)

    train_stationary, val_stationary, test_stationary, stationarity = apply_stationarity_policy(
        train_scaled, val_scaled, test_scaled
    )
    validate_split_integrity(train_stationary, val_stationary, test_stationary)
    columns = [TARGET_TICKER, CARBON_TICKER, IRON_TICKER, FX_TICKER]

    return PreprocessOutput(
        train=train_stationary[columns],
        val=val_stationary[columns],
        test=test_stationary[columns],
        scaler=scaler,
        stationarity=stationarity,
    )


def train_test_split_time_series(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_size = int(n * train_ratio)
    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()


def train_val_test_split_time_series(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")

    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    if train_size <= 0 or val_size <= 0 or n - train_size - val_size <= 0:
        raise ValueError("Dataset is too small for the specified split ratios.")

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()
    return train_df, val_df, test_df


def create_exog_candidates(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    x1 = df[CARBON_TICKER]
    x2 = df[IRON_TICKER]
    x3 = df[FX_TICKER]
    candidates = {
        "x1": pd.DataFrame({CARBON_TICKER: x1}, index=df.index),
        "x2": pd.DataFrame({IRON_TICKER: x2}, index=df.index),
        "x1_x2": pd.DataFrame({CARBON_TICKER: x1, IRON_TICKER: x2}, index=df.index),
        "x1_x3": pd.DataFrame({CARBON_TICKER: x1, FX_TICKER: x3}, index=df.index),
        "x2_x3": pd.DataFrame({IRON_TICKER: x2, FX_TICKER: x3}, index=df.index),
        "x1_x2_x3": pd.DataFrame({CARBON_TICKER: x1, IRON_TICKER: x2, FX_TICKER: x3}, index=df.index),
        "x1_x2_x3_lag_roll": pd.DataFrame(
            {
                CARBON_TICKER: x1,
                IRON_TICKER: x2,
                FX_TICKER: x3,
                f"{CARBON_TICKER}_lag1": x1.shift(1),
                f"{IRON_TICKER}_lag1": x2.shift(1),
                f"{FX_TICKER}_lag1": x3.shift(1),
                f"{CARBON_TICKER}_roll3": x1.rolling(3).mean(),
                f"{IRON_TICKER}_roll3": x2.rolling(3).mean(),
                f"{FX_TICKER}_roll3": x3.rolling(3).mean(),
            },
            index=df.index,
        ),
    }
    return {k: v.ffill().bfill() for k, v in candidates.items()}


def fit_sarimax_with_validation(
    y_train: pd.Series,
    exog_train_candidates: Dict[str, pd.DataFrame],
    y_val: pd.Series,
    exog_val_candidates: Dict[str, pd.DataFrame],
) -> Tuple[SARIMAXResultsWrapper, Dict[str, float | str | int]]:
    print("\nARIMAX modeli eğitiliyor (validation + diagnostik seçimi)...")
    best_model: Optional[SARIMAXResultsWrapper] = None
    best_meta: Dict[str, float | str | int] = {}
    best_score = np.inf

    for exog_name, exog_train in exog_train_candidates.items():
        exog_val = exog_val_candidates[exog_name]
        y_train_aligned = y_train.loc[exog_train.index]
        y_val_aligned = y_val.loc[exog_val.index]

        for p in ARIMAX_P_RANGE:
            for d in ARIMAX_D_RANGE:
                for q in ARIMAX_Q_RANGE:
                    try:
                        candidate = SARIMAX(
                            y_train_aligned,
                            exog=exog_train,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                    except Exception:
                        continue

                    try:
                        val_pred = pd.Series(
                            candidate.forecast(steps=len(y_val_aligned), exog=exog_val).values,
                            index=y_val_aligned.index,
                        )
                    except Exception:
                        continue

                    val_rmse = float(np.sqrt(mean_squared_error(y_val_aligned, val_pred)))
                    resid = pd.Series(candidate.resid).dropna()
                    resid_std = float(resid.std()) if not resid.empty else 1.0
                    if len(resid) > 10:
                        lag = min(10, max(1, len(resid) - 1))
                        ljung_box_pvalue = float(acorr_ljungbox(resid, lags=[lag], return_df=True)["lb_pvalue"].iloc[0])
                    else:
                        ljung_box_pvalue = 0.5
                    aic_penalty = float(candidate.aic) * 1e-4 if np.isfinite(candidate.aic) else 1.0
                    stability_penalty = max(0.0, 0.05 - ljung_box_pvalue) + 0.05 * resid_std
                    score = val_rmse + stability_penalty + aic_penalty

                    if score < best_score:
                        best_score = score
                        best_model = candidate
                        best_meta = {
                            "exog_name": exog_name,
                            "p": p,
                            "d": d,
                            "q": q,
                            "val_rmse": val_rmse,
                            "ljung_box_pvalue": ljung_box_pvalue,
                            "resid_std": resid_std,
                            "aic": float(candidate.aic) if np.isfinite(candidate.aic) else np.inf,
                            "score": score,
                        }

    if best_model is None:
        fallback_exog = exog_train_candidates["x2"]
        best_model = SARIMAX(
            y_train.loc[fallback_exog.index],
            exog=fallback_exog,
            order=(1, 0, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        best_meta = {
            "exog_name": "x2",
            "p": 1,
            "d": 0,
            "q": 1,
            "val_rmse": 0.0,
            "ljung_box_pvalue": 0.0,
            "resid_std": 0.0,
            "aic": 0.0,
            "score": 0.0,
        }

    print(
        f"Seçilen ARIMAX: exog={best_meta['exog_name']}, "
        f"order=({best_meta['p']},{best_meta['d']},{best_meta['q']}), "
        f"val_rmse={best_meta['val_rmse']}, score={best_meta['score']}"
    )
    return best_model, best_meta


def fit_tuned_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    from xgboost import XGBRegressor

    search_space = [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 120, "learning_rate": 0.1, "max_depth": 5},
    ]
    best_model = None
    best_rmse = np.inf
    best_params = None
    for params in search_space:
        model = XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params
    if best_model is None:
        raise RuntimeError(f"XGBoost model could not be trained. Search space: {search_space}")
    print(f"Seçilen XGBoost parametreleri: {best_params}, val_rmse={best_rmse:.6f}")
    return best_model


def build_residual_training_frame(
    x1_train: pd.Series,
    residuals_train: pd.Series,
    x2_train: Optional[pd.Series] = None,
    x3_train: Optional[pd.Series] = None,
    feature_mode: str = "full",
) -> Tuple[pd.DataFrame, pd.Series]:
    if x2_train is None:
        x2_train = pd.Series(0.0, index=x1_train.index)
    if x3_train is None:
        x3_train = pd.Series(0.0, index=x1_train.index)
    X = pd.DataFrame({"x1": x1_train, "x2": x2_train, "x3": x3_train})
    X["x1_x2_interaction"] = X["x1"] * X["x2"]
    if feature_mode == "full":
        X["x1_x3_interaction"] = X["x1"] * X["x3"]
        X["x2_x3_interaction"] = X["x2"] * X["x3"]
    x1_change = x1_train.diff()
    x2_change = x2_train.diff()
    x3_change = x3_train.diff()
    lag_count = 3 if feature_mode == "compact" else RESIDUAL_LAG_COUNT
    for lag in range(1, lag_count + 1):
        X[f"residual_lag{lag}"] = residuals_train.shift(lag)
        X[f"x1_lag{lag}"] = x1_train.shift(lag)
        X[f"x2_lag{lag}"] = x2_train.shift(lag)
        X[f"x3_lag{lag}"] = x3_train.shift(lag)
    for lag in range(1, lag_count + 1):
        X[f"x1_change_lag{lag}"] = x1_change.shift(lag)
        X[f"x2_change_lag{lag}"] = x2_change.shift(lag)
        X[f"x3_change_lag{lag}"] = x3_change.shift(lag)
    X["residual_roll_mean_3"] = residuals_train.shift(1).rolling(RESIDUAL_ROLL_WINDOW).mean()
    X["residual_roll_std_3"] = residuals_train.shift(1).rolling(RESIDUAL_ROLL_WINDOW).std()
    X = X.dropna()
    y = residuals_train.loc[X.index]
    return X, y


def build_and_train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    hidden_layers: Tuple[int, int, int] = (64, 32, 16),
    dropout_rate: float = 0.2,
    learning_rate: float = MLP_PARAMS["learning_rate"],
    patience: int = MLP_PARAMS["patience"],
    epochs: int = MLP_PARAMS["epochs"],
    batch_size: int = MLP_PARAMS["batch_size"],
    seed: int = 42,
):
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(hidden_layers[0], activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(hidden_layers[1], activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(hidden_layers[2], activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    has_validation = X_val is not None and y_val is not None and len(X_val) > 0 and len(X_val) == len(y_val)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss" if has_validation else "loss",
        patience=patience,
        restore_best_weights=True,
    )

    fit_kwargs = {
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks": [early_stopping],
        "verbose": 0,
    }
    if has_validation:
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        fit_kwargs["validation_split"] = MLP_PARAMS["validation_split"]

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def select_best_mlp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    search_space = [
        {"hidden_layers": (64, 32, 16), "dropout_rate": 0.2, "learning_rate": 0.0005, "patience": 20, "epochs": 250, "batch_size": 16},
        {"hidden_layers": (128, 64, 32), "dropout_rate": 0.2, "learning_rate": 0.0003, "patience": 30, "epochs": 300, "batch_size": 16},
    ]
    seeds = [42, 123]
    best_model = None
    best_score = np.inf
    evaluated: List[Tuple[dict, int, float]] = []
    for config in search_space:
        config_runs: List[Tuple[float, object]] = []
        for seed in seeds:
            model = build_and_train_mlp(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                seed=seed,
                **config,
            )
            pred = model.predict(X_val, verbose=0).ravel()
            rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
            evaluated.append((config, seed, rmse))
            config_runs.append((rmse, model))
        mean_rmse = float(np.mean([x[0] for x in config_runs]))
        std_rmse = float(np.std([x[0] for x in config_runs]))
        score = mean_rmse + 0.10 * std_rmse
        if score < best_score:
            best_score = score
            best_model = sorted(config_runs, key=lambda x: x[0])[0][1]
    if best_model is None:
        raise RuntimeError(f"MLP model selection failed. Tried {len(evaluated)} config/seed combinations.")
    return best_model


def forecast_mlp_residuals(
    mlp_model,
    x1_test: pd.Series,
    train_x1_history: pd.Series,
    train_residual_history: pd.Series,
    lag_count: int = RESIDUAL_LAG_COUNT,
    x2_test: Optional[pd.Series] = None,
    train_x2_history: Optional[pd.Series] = None,
    x3_test: Optional[pd.Series] = None,
    train_x3_history: Optional[pd.Series] = None,
) -> pd.Series:
    if x2_test is None:
        x2_test = pd.Series(0.0, index=x1_test.index)
    if train_x2_history is None:
        train_x2_history = pd.Series(0.0, index=train_x1_history.index)
    if x3_test is None:
        x3_test = pd.Series(0.0, index=x1_test.index)
    if train_x3_history is None:
        train_x3_history = pd.Series(0.0, index=train_x1_history.index)

    preds = []
    x1_history = list(train_x1_history.astype(float).values)
    x2_history = list(train_x2_history.astype(float).values)
    x3_history = list(train_x3_history.astype(float).values)
    residual_history = list(train_residual_history.astype(float).values)

    required_x1_len = lag_count + 1
    if len(x1_history) < required_x1_len:
        raise ValueError(
            f"train_x1_history must include at least {required_x1_len} values to compute {lag_count} lagged changes."
        )
    if len(x2_history) < required_x1_len:
        raise ValueError(
            f"train_x2_history must include at least {required_x1_len} values to compute {lag_count} lagged changes."
        )
    if len(x3_history) < required_x1_len:
        raise ValueError(
            f"train_x3_history must include at least {required_x1_len} values to compute {lag_count} lagged changes."
        )
    # En uzun geçmiş ihtiyacı lag_count ve rolling pencere koşullarının maksimumudur.
    required_residual_len = max(lag_count, RESIDUAL_ROLL_WINDOW)
    if len(residual_history) < required_residual_len:
        raise ValueError(
            f"train_residual_history must include at least {required_residual_len} values for lag_count={lag_count}."
        )

    for i in range(len(x1_test)):
        curr_x1 = float(x1_test.iloc[i])
        curr_x2 = float(x2_test.iloc[i])
        curr_x3 = float(x3_test.iloc[i])
        x1_change_history = [x1_history[j] - x1_history[j - 1] for j in range(1, len(x1_history))]
        x2_change_history = [x2_history[j] - x2_history[j - 1] for j in range(1, len(x2_history))]
        x3_change_history = [x3_history[j] - x3_history[j - 1] for j in range(1, len(x3_history))]
        roll_source = (
            residual_history[-RESIDUAL_ROLL_WINDOW:]
            if len(residual_history) >= RESIDUAL_ROLL_WINDOW
            else residual_history
        )
        features: List[float] = [curr_x1, curr_x2, curr_x3, curr_x1 * curr_x2, curr_x1 * curr_x3, curr_x2 * curr_x3]

        for lag in range(1, lag_count + 1):
            features.append(float(residual_history[-lag]))
            features.append(float(x1_history[-lag]))
            features.append(float(x2_history[-lag]))
            features.append(float(x3_history[-lag]))
            features.append(float(x1_change_history[-lag]))
            features.append(float(x2_change_history[-lag]))
            features.append(float(x3_change_history[-lag]))
        features.append(float(np.mean(roll_source)))
        features.append(float(np.std(roll_source)))

        X_input = np.array([features], dtype=float)
        pred_res = float(mlp_model.predict(X_input, verbose=0).ravel()[0])
        preds.append(pred_res)
        residual_history.append(pred_res)
        x1_history.append(curr_x1)
        x2_history.append(curr_x2)
        x3_history.append(curr_x3)

    return pd.Series(preds, index=x1_test.index)


def train_hybrid_combiner(
    y_val: pd.Series,
    arimax_val_pred: pd.Series,
    mlp_val_residual_pred: pd.Series,
) -> RidgeCV:
    X_meta = pd.DataFrame(
        {
            "arimax": arimax_val_pred,
            "mlp_residual": mlp_val_residual_pred,
        },
        index=y_val.index,
    )
    model = RidgeCV(alphas=np.array([0.01, 0.1, 1.0, 10.0]))
    model.fit(X_meta, y_val)
    return model


def apply_hybrid_combiner(
    model,
    arimax_pred: pd.Series,
    mlp_residual_pred: pd.Series,
) -> pd.Series:
    X_meta = pd.DataFrame(
        {
            "arimax": arimax_pred,
            "mlp_residual": mlp_residual_pred,
        },
        index=arimax_pred.index,
    )
    return pd.Series(model.predict(X_meta), index=arimax_pred.index, name="Hibrit ARIMAX-MLP")


def calculate_metrics(y_true: pd.Series, predictions_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    metrics = []
    for model_name, y_pred in predictions_dict.items():
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        metrics.append({"Model": model_name, "RMSE": rmse, "MAE": mae})
    return pd.DataFrame(metrics).sort_values(by="RMSE").reset_index(drop=True)


def evaluate_success_criteria(
    metrics_df: pd.DataFrame,
    rolling_window_results_df: pd.DataFrame,
    min_single_split_improvement: float = SUCCESS_MIN_SINGLE_SPLIT_IMPROVEMENT,
    min_rolling_win_ratio: float = SUCCESS_MIN_ROLLING_WIN_RATIO,
    min_rolling_mean_improvement: float = SUCCESS_MIN_ROLLING_MEAN_IMPROVEMENT,
    stable_rmse_upper: float = STABLE_RMSE_BAND_UPPER,
) -> SuccessSummary:
    sorted_metrics = metrics_df.sort_values("RMSE").reset_index(drop=True)
    hybrid_row = sorted_metrics[sorted_metrics["Model"] == "Hibrit ARIMAX-MLP"]
    if hybrid_row.empty:
        return {
            "single_split_improvement": 0.0,
            "rolling_win_ratio": 0.0,
            "rolling_mean_improvement": 0.0,
            "single_split_rmse": float("inf"),
            "rolling_hybrid_rmse_mean": float("inf"),
            "pass": False,
            "decision_reason": "Hybrid row missing in metrics table.",
        }

    hybrid_rmse = float(hybrid_row.iloc[0]["RMSE"])
    baseline_comp = sorted_metrics[sorted_metrics["Model"] != "Hibrit ARIMAX-MLP"].iloc[0]
    second_best_rmse = float(baseline_comp["RMSE"])
    single_split_improvement = (second_best_rmse - hybrid_rmse) / second_best_rmse if second_best_rmse > 0 else 0.0

    hybrid_wins = 0
    total_windows = 0
    rolling_improvements: List[float] = []
    for _, window_df in rolling_window_results_df.groupby("window"):
        total_windows += 1
        sorted_window = window_df.sort_values("RMSE").reset_index(drop=True)
        if sorted_window.iloc[0]["Model"] == "Hibrit ARIMAX-MLP":
            hybrid_wins += 1
        hybrid_window = sorted_window[sorted_window["Model"] == "Hibrit ARIMAX-MLP"]
        non_hybrid = sorted_window[sorted_window["Model"] != "Hibrit ARIMAX-MLP"]
        if not hybrid_window.empty and not non_hybrid.empty:
            h_rmse = float(hybrid_window.iloc[0]["RMSE"])
            n_rmse = float(non_hybrid.iloc[0]["RMSE"])
            if n_rmse > 0:
                rolling_improvements.append((n_rmse - h_rmse) / n_rmse)

    rolling_win_ratio = (hybrid_wins / total_windows) if total_windows else 0.0
    rolling_mean_improvement = float(np.mean(rolling_improvements)) if rolling_improvements else 0.0
    rolling_hybrid = rolling_window_results_df[rolling_window_results_df["Model"] == "Hibrit ARIMAX-MLP"]
    rolling_hybrid_rmse_mean = float(rolling_hybrid["RMSE"].mean()) if not rolling_hybrid.empty else float("inf")
    criteria_pass = (
        single_split_improvement >= min_single_split_improvement
        and rolling_win_ratio >= min_rolling_win_ratio
        and rolling_mean_improvement >= min_rolling_mean_improvement
        and hybrid_rmse <= stable_rmse_upper
        and rolling_hybrid_rmse_mean <= stable_rmse_upper
    )
    decision_reason = (
        "PASS"
        if criteria_pass
        else (
            f"FAIL: split_improvement={single_split_improvement:.4f}, "
            f"rolling_win_ratio={rolling_win_ratio:.4f}, "
            f"rolling_mean_improvement={rolling_mean_improvement:.4f}, "
            f"single_split_rmse={hybrid_rmse:.4f}, "
            f"rolling_rmse_mean={rolling_hybrid_rmse_mean:.4f}"
        )
    )
    return {
        "single_split_improvement": float(single_split_improvement),
        "rolling_win_ratio": float(rolling_win_ratio),
        "rolling_mean_improvement": float(rolling_mean_improvement),
        "single_split_rmse": float(hybrid_rmse),
        "rolling_hybrid_rmse_mean": float(rolling_hybrid_rmse_mean),
        "pass": bool(criteria_pass),
        "decision_reason": decision_reason,
    }


def build_expanding_windows(
    df: pd.DataFrame,
    min_train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
    max_windows: int = 5,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    windows: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    n = len(df)
    start = min_train_size + val_size
    while start + test_size <= n and len(windows) < max_windows:
        train_end = start - val_size
        val_end = start
        test_end = start + test_size
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:test_end].copy()
        if len(train_df) and len(val_df) and len(test_df):
            windows.append((train_df, val_df, test_df))
        start += step_size
    return windows


def run_single_window_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.Series, Dict[str, pd.Series], SARIMAXResultsWrapper, Dict[str, float | str | int]]:
    y_train = train_df[TARGET_TICKER]
    y_val = val_df[TARGET_TICKER]
    y_test = test_df[TARGET_TICKER]
    x1_train = train_df[CARBON_TICKER]
    x1_val = val_df[CARBON_TICKER]
    x1_test = test_df[CARBON_TICKER]
    x2_train = train_df[IRON_TICKER]
    x2_val = val_df[IRON_TICKER]
    x2_test = test_df[IRON_TICKER]
    x3_train = train_df[FX_TICKER]
    x3_val = val_df[FX_TICKER]
    x3_test = test_df[FX_TICKER]

    exog_train_candidates = create_exog_candidates(train_df)
    exog_val_candidates = create_exog_candidates(val_df)
    exog_test_candidates = create_exog_candidates(test_df)
    arimax_model, arimax_meta = fit_sarimax_with_validation(
        y_train=y_train,
        exog_train_candidates=exog_train_candidates,
        y_val=y_val,
        exog_val_candidates=exog_val_candidates,
    )
    selected_exog = str(arimax_meta["exog_name"])
    arimax_val_pred = pd.Series(
        arimax_model.forecast(steps=len(y_val), exog=exog_val_candidates[selected_exog]).values,
        index=y_val.index,
        name="ARIMAX_VAL",
    )
    arimax_test_pred = pd.Series(
        arimax_model.forecast(steps=len(y_test), exog=exog_test_candidates[selected_exog]).values,
        index=y_test.index,
        name="ARIMAX",
    )

    residuals_train = pd.Series(arimax_model.resid, index=y_train.index)
    residuals_val_actual = y_val - arimax_val_pred
    x1_train_val = pd.concat([x1_train, x1_val])
    x2_train_val = pd.concat([x2_train, x2_val])
    x3_train_val = pd.concat([x3_train, x3_val])
    residuals_train_val = pd.concat([residuals_train, residuals_val_actual])
    X_mlp_all, y_mlp_all = build_residual_training_frame(
        x1_train_val, residuals_train_val, x2_train=x2_train_val, x3_train=x3_train_val, feature_mode="compact"
    )
    split_labels = pd.concat([pd.Series("train", index=x1_train.index), pd.Series("val", index=x1_val.index)]).loc[X_mlp_all.index]
    X_mlp_train = X_mlp_all.loc[split_labels.eq("train")]
    y_mlp_train = y_mlp_all.loc[split_labels.eq("train")]
    X_mlp_val = X_mlp_all.loc[split_labels.eq("val")]
    y_mlp_val = y_mlp_all.loc[split_labels.eq("val")]

    mlp_model = select_best_mlp_model(
        X_train=X_mlp_train.values,
        y_train=y_mlp_train.values,
        X_val=X_mlp_val.values,
        y_val=y_mlp_val.values,
    )
    mlp_residual_val_pred = forecast_mlp_residuals(
        mlp_model=mlp_model,
        x1_test=x1_val,
        x2_test=x2_val,
        x3_test=x3_val,
        train_x1_history=x1_train.tail(RESIDUAL_LAG_COUNT + 1),
        train_x2_history=x2_train.tail(RESIDUAL_LAG_COUNT + 1),
        train_x3_history=x3_train.tail(RESIDUAL_LAG_COUNT + 1),
        train_residual_history=residuals_train.tail(max(RESIDUAL_LAG_COUNT, RESIDUAL_ROLL_WINDOW)),
    )
    mlp_residual_test_pred = forecast_mlp_residuals(
        mlp_model=mlp_model,
        x1_test=x1_test,
        x2_test=x2_test,
        x3_test=x3_test,
        train_x1_history=x1_train_val.tail(RESIDUAL_LAG_COUNT + 1),
        train_x2_history=x2_train_val.tail(RESIDUAL_LAG_COUNT + 1),
        train_x3_history=x3_train_val.tail(RESIDUAL_LAG_COUNT + 1),
        train_residual_history=residuals_train_val.tail(max(RESIDUAL_LAG_COUNT, RESIDUAL_ROLL_WINDOW)),
    )
    combiner = train_hybrid_combiner(
        y_val=y_val,
        arimax_val_pred=arimax_val_pred,
        mlp_val_residual_pred=mlp_residual_val_pred,
    )
    hybrid_pred = apply_hybrid_combiner(combiner, arimax_test_pred, mlp_residual_test_pred)

    baseline_model = LinearRegression()
    X_train_bench = train_df[[CARBON_TICKER, IRON_TICKER, FX_TICKER]]
    X_val_bench = val_df[[CARBON_TICKER, IRON_TICKER, FX_TICKER]]
    X_test_bench = test_df[[CARBON_TICKER, IRON_TICKER, FX_TICKER]]
    baseline_model.fit(X_train_bench, y_train)
    baseline_pred = pd.Series(baseline_model.predict(X_test_bench), index=y_test.index, name="Baseline")

    xgb_model = fit_tuned_xgboost(X_train_bench, y_train, X_val_bench, y_val)
    xgb_pred = pd.Series(xgb_model.predict(X_test_bench), index=y_test.index, name="XGBoost")

    predictions = {"Baseline": baseline_pred, "ARIMAX": arimax_test_pred, "XGBoost": xgb_pred, "Hibrit ARIMAX-MLP": hybrid_pred}
    return y_test, predictions, arimax_model, arimax_meta


def run_rolling_backtest(df_raw: pd.DataFrame, max_windows: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    min_train_size = int(len(df_raw) * ROLLING_MIN_TRAIN_SIZE_RATIO)
    val_size = max(ROLLING_MIN_VAL_ROWS, int(len(df_raw) * ROLLING_VAL_SIZE_RATIO))
    test_size = max(ROLLING_MIN_TEST_ROWS, int(len(df_raw) * ROLLING_TEST_SIZE_RATIO))
    windows = build_expanding_windows(
        df=df_raw,
        min_train_size=min_train_size,
        val_size=val_size,
        test_size=test_size,
        step_size=test_size,
        max_windows=max_windows,
    )
    rolling_rows: List[pd.DataFrame] = []
    for i, (train_raw, val_raw, test_raw) in enumerate(windows, start=1):
        validate_split_integrity(train_raw, val_raw, test_raw)
        scaler = MinMaxScaler()
        train_scaled = pd.DataFrame(scaler.fit_transform(train_raw), columns=train_raw.columns, index=train_raw.index)
        val_scaled = pd.DataFrame(scaler.transform(val_raw), columns=val_raw.columns, index=val_raw.index)
        test_scaled = pd.DataFrame(scaler.transform(test_raw), columns=test_raw.columns, index=test_raw.index)
        train_df, val_df, test_df, _ = apply_stationarity_policy(train_scaled, val_scaled, test_scaled)
        validate_split_integrity(train_df, val_df, test_df)
        if len(train_df) < ROLLING_MIN_TRAIN_ROWS or len(val_df) < ROLLING_MIN_VAL_ROWS or len(test_df) < ROLLING_MIN_TEST_ROWS:
            continue
        y_test, preds, _, _ = run_single_window_models(train_df, val_df, test_df)
        metrics = calculate_metrics(y_test, preds)
        metrics["window"] = i
        rolling_rows.append(metrics)

    if not rolling_rows:
        empty_summary = pd.DataFrame(columns=["Model", "RMSE_mean", "RMSE_std", "wins"])
        empty_windows = pd.DataFrame(columns=["window", "Model", "RMSE", "MAE"])
        return empty_summary, empty_windows

    rolling_window_results = pd.concat(rolling_rows, ignore_index=True)
    wins = []
    for model in rolling_window_results["Model"].unique():
        win_count = 0
        for _, window_df in rolling_window_results.groupby("window"):
            if window_df.sort_values("RMSE").iloc[0]["Model"] == model:
                win_count += 1
        wins.append({"Model": model, "wins": win_count})
    wins_df = pd.DataFrame(wins)
    summary = (
        rolling_window_results.groupby("Model", as_index=False)
        .agg(RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"))
        .merge(wins_df, on="Model", how="left")
        .sort_values("RMSE_mean")
        .reset_index(drop=True)
    )
    return summary, rolling_window_results


def build_module_breakdown(y_true: pd.Series, predictions: Dict[str, pd.Series], rolling_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    ordered_models = ["Baseline", "ARIMAX", "XGBoost", "Hibrit ARIMAX-MLP"]
    for model in ordered_models:
        if model in predictions:
            rmse = float(np.sqrt(mean_squared_error(y_true, predictions[model])))
            rows.append({"module": model, "rmse": rmse})
    hybrid_rolling = rolling_summary_df[rolling_summary_df["Model"] == "Hibrit ARIMAX-MLP"]
    if not hybrid_rolling.empty:
        rows.append({"module": "Rolling(Hibrit mean)", "rmse": float(hybrid_rolling.iloc[0]["RMSE_mean"])})
    return pd.DataFrame(rows)


def find_first_breakpoint(module_breakdown: pd.DataFrame, threshold: float = STABLE_RMSE_BAND_UPPER) -> str:
    if module_breakdown.empty:
        return "Kırılma analizi yapılamadı: modül metriği boş."
    over = module_breakdown[module_breakdown["rmse"] > threshold]
    if over.empty:
        return f"Kırılma yok: tüm modüller RMSE<{threshold:.3f} bandında."
    first = over.iloc[0]
    return f"İlk kırılma noktası: {first['module']} (RMSE={float(first['rmse']):.4f}, eşik={threshold:.4f})"


def run_diagnostic_baseline(
    df_model: pd.DataFrame,
    seeds: List[int] = REPEATED_SEEDS,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int]] = []
    for seed in seeds:
        set_global_seed(seed)
        splits = prepare_leakage_safe_splits(df_model)
        y_test, predictions, _, _ = run_single_window_models(splits.train, splits.val, splits.test)
        metrics_df = calculate_metrics(y_test, predictions)
        hybrid_rmse = float(metrics_df.loc[metrics_df["Model"] == "Hibrit ARIMAX-MLP", "RMSE"].iloc[0])
        hybrid_mae = float(metrics_df.loc[metrics_df["Model"] == "Hibrit ARIMAX-MLP", "MAE"].iloc[0])
        rows.append({"seed": seed, "RMSE": hybrid_rmse, "MAE": hybrid_mae})
    out = pd.DataFrame(rows)
    out["RMSE_mean"] = out["RMSE"].mean()
    out["RMSE_std"] = out["RMSE"].std(ddof=0)
    return out


def run_ablation_experiments(
    y_test: pd.Series,
    predictions: Dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    candidates = {
        "A_ARIMAX_only": ["ARIMAX"],
        "B_ARIMAX_plus_linear_baseline": ["ARIMAX", "Baseline"],
        "C_XGBoost_only": ["XGBoost"],
        "D_Hybrid_enabled": ["Hibrit ARIMAX-MLP"],
        "E_All_models_ensemble_mean": ["Baseline", "ARIMAX", "XGBoost", "Hibrit ARIMAX-MLP"],
    }
    for name, model_names in candidates.items():
        selected = [predictions[m] for m in model_names if m in predictions]
        if not selected:
            continue
        combined_pred = pd.concat(selected, axis=1).mean(axis=1)
        rmse = float(np.sqrt(mean_squared_error(y_test, combined_pred)))
        mae = float(mean_absolute_error(y_test, combined_pred))
        rows.append({"Variant": name, "RMSE": rmse, "MAE": mae, "ModelCount": len(selected)})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def plot_module_visualizations(
    y_test: pd.Series, predictions_df: pd.DataFrame, metrics_df: pd.DataFrame, residuals_df: pd.DataFrame
):
    plt.figure(figsize=(8, 6))
    bar_colors = [MODEL_COLORS.get(m, DEFAULT_MODEL_COLOR) for m in metrics_df["Model"]]
    ax = sns.barplot(data=metrics_df, x="Model", y="RMSE", hue="Model", palette=bar_colors, legend=False)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.6f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )
    plt.title("Şekil 4.1: Modellerin Test Kümesi RMSE Karşılaştırması", pad=15, fontsize=12, fontweight="bold")
    plt.ylabel("RMSE Değeri")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("Grafik_1_RMSE_Bar.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, color="black", label="Gerçek EREGL.IS", linewidth=2.2, linestyle="--")
    for model_col in predictions_df.columns:
        lw = 2.5 if "Hibrit" in model_col else 1.2
        alpha = 1.0 if "Hibrit" in model_col else 0.75
        plt.plot(
            predictions_df.index,
            predictions_df[model_col],
            label=model_col,
            color=MODEL_COLORS.get(model_col, DEFAULT_MODEL_COLOR),
            linewidth=lw,
            alpha=alpha,
        )
    plt.title("Şekil 4.2: Zaman Serisi Tahmin Performansı (Gerçek vs. Modeller)", pad=15, fontsize=12, fontweight="bold")
    plt.ylabel("Fiyat / Getiri (Fark Serisi)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("Grafik_2_Prediction_Line.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=residuals_df["XGBoost"],
        label="XGBoost Hata Dağılımı",
        color=MODEL_COLORS["XGBoost"],
        fill=True,
        alpha=0.3,
    )
    sns.kdeplot(
        data=residuals_df["Hibrit ARIMAX-MLP"],
        label="Hibrit Model Hata Dağılımı",
        color=MODEL_COLORS["Hibrit ARIMAX-MLP"],
        fill=True,
        alpha=0.5,
    )
    plt.title("Şekil 4.3: Hata Dağılımı Çekirdek Yoğunluk Tahmini (KDE)", pad=15, fontsize=12, fontweight="bold")
    plt.xlabel("Tahmin Hatası (Gerçek - Tahmin)")
    plt.ylabel("Yoğunluk (Density)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Grafik_3_Residual_KDE.png")
    plt.close()


def plot_stress_test_fan_chart(last_test_date: pd.Timestamp, base_price: float):
    dates = pd.bdate_range(last_test_date + pd.offsets.BDay(1), periods=30)
    base = np.full(len(dates), base_price)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, base, color="black", linestyle=":", linewidth=2.2, label="Baz Senaryo Fiyatı")

    colors = {"S1 (+%30)": "#ffb703", "S2 (+%60)": "#fb8500", "S3 (+%100)": "#d00000"}
    prev_upper = base.copy()
    prev_lower = base.copy()

    for scenario, shock in STRESS_BANDS.items():
        upper = base * (1 + shock)
        lower = base * (1 - shock)
        plt.plot(dates, upper, color=colors[scenario], linewidth=1.4, alpha=0.9)
        plt.plot(dates, lower, color=colors[scenario], linewidth=1.4, alpha=0.9)
        plt.fill_between(dates, prev_upper, upper, color=colors[scenario], alpha=0.12, label=f"{scenario} Üst Bant")
        plt.fill_between(dates, lower, prev_lower, color=colors[scenario], alpha=0.12, label=f"{scenario} Alt Bant")
        prev_upper = upper
        prev_lower = lower

    plt.title("Şekil 4.4: Test Sonrası 30 Gün Karbon Stres Testi Yelpaze Grafiği (Gerçek Fiyat, USD)", pad=15, fontsize=12, fontweight="bold")
    plt.xlabel("Tarih")
    plt.ylabel("Hisse Fiyatı (USD)")
    plt.legend(loc="upper left", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("Grafik_4_Stres_Testi_Fan.png")
    plt.close()


def run_stress_test(base_price: float) -> pd.DataFrame:
    rows = []
    for scenario, shock in STRESS_BANDS.items():
        rows.append(
            {
                "Senaryo": scenario,
                "Şok Oranı": f"+%{int(shock * 100)}",
                "Alt Bant Fiyatı": base_price * (1 - shock),
                "Baz Senaryo Fiyatı": base_price,
                "Üst Bant Fiyatı": base_price * (1 + shock),
            }
        )
    return pd.DataFrame(rows)

def plot_raw_data_summary(df_raw: pd.DataFrame):
    # 4 satır, 1 sütunluk ortak X eksenli bir figür oluşturuyoruz
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # 1. Panel: Hedef Değişken (EREGL.IS)
    axes[0].plot(df_raw.index, df_raw[TARGET_TICKER], color="#1f77b4", linewidth=1.5)
    axes[0].set_title(f"{TARGET_TICKER} - Kapanış Fiyatı (USD olarak standardize)", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Fiyat")
    
    # 2. Panel: Karbon Fonu (KEUA)
    axes[1].plot(df_raw.index, df_raw[CARBON_TICKER], color="#2ca02c", linewidth=1.5)
    axes[1].set_title(f"{CARBON_TICKER} - Karbon Fonu Fiyatı", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Fiyat")
    
    # 3. Panel: Demir Cevheri (TIO=F)
    axes[2].plot(df_raw.index, df_raw[IRON_TICKER], color="#d62728", linewidth=1.5)
    axes[2].set_title(f"{IRON_TICKER} - Demir Cevheri Vadeli İşlem Fiyatı", fontweight="bold", fontsize=11)
    axes[2].set_ylabel("Fiyat")

    # 4. Panel: USD/TRY
    axes[3].plot(df_raw.index, df_raw[FX_TICKER], color="#9467bd", linewidth=1.5)
    axes[3].set_title(f"{FX_TICKER} - USD/TRY Kuru", fontweight="bold", fontsize=11)
    axes[3].set_ylabel("Kur")
    axes[3].set_xlabel("Tarih")
    
    # Ana Başlık ve Kaydetme İşlemleri
    fig.suptitle("Şekil 3.1: Ham Veri Zaman Serisi Özeti", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig("Grafik_0a_Ham_Veri_Ozeti.png", dpi=300)
    plt.close()
    print("Ham veri özet grafiği kaydedildi: Grafik_0a_Ham_Veri_Ozeti.png")

def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".4f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "Şekil 0: Temel Değişkenler Korelasyon Isı Haritası\n(EREGL.IS, KEUA, TIO=F)",
        pad=15,
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("Grafik_0_Korelasyon_Heatmap.png", dpi=300)
    plt.close()
    print("Korelasyon ısı haritası kaydedildi: Grafik_0_Korelasyon_Heatmap.png")


def write_thesis_report(
    basic_stats: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    diff_stats: pd.DataFrame,
    adf_results: Dict[str, Dict[str, float]],
    arimax_summary: str,
    arimax_coef_table_str: str,
    diagnostics_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    rolling_summary_df: pd.DataFrame,
    success_summary: SuccessSummary,
    acceptance_summary: Dict[str, float | bool | str],
    diagnostic_baseline_df: pd.DataFrame,
    module_breakdown_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    root_cause_report: str,
    stress_table_df: pd.DataFrame,
    report_path: Path,
):
    lines = [
        "TEZ BULGULARI RAPORU",
        "=" * 80,
        "",
        "TABLO 4.1: BETİMSEL İSTATİSTİKLER (BİRİNCİ FARKI ALINMIŞ VE TEMİZLENMİŞ SERİLER)",
        "-" * 80,
        diff_stats.to_string(float_format=lambda x: f"{x:.6f}"),
        "",
        "=" * 80,
        "",
        "1) TEMEL İSTATİSTİKLER VE KORELASYON ANALİZİ (HAM VERİ)",
        "-" * 80,
        "1a) Temel İstatistikler (Ham Kapanış Fiyatları):",
        basic_stats.to_string(float_format=lambda x: f"{x:.4f}"),
        "",
        "1b) Korelasyon Matrisi (Ham Kapanış Fiyatları):",
        corr_matrix.to_string(float_format=lambda x: f"{x:.6f}"),
        "",
             "2) ADF TEST SONUÇLARI (YALNIZCA EĞİTİM BÖLÜMÜ REFERANSLI)",
             "-" * 80,
    ]
    for col, result in adf_results.items():
        lines.append(
            f"{col}: initial_p={result['initial_p']:.6f}, differenced={result['differenced']}, final_p={result['final_p']:.6f}"
        )

    lines.extend(
        [
            "",
            "3) ARIMAX MODEL ÖZETİ",
            "-" * 80,
            arimax_summary,
            "",
            "TABLO 4.2: ARIMAX KATSAYI TABLOSU (AR/MA/d ve exog aday seçimi)",
            "-" * 80,
            arimax_coef_table_str,
            "",
            "4) RESIDUAL TANI TESTLERİ (LJUNG-BOX / ARCH-LM)",
            "-" * 80,
            diagnostics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
            "",
            "5) TEST KÜMESİ PERFORMANS METRİKLERİ (RMSE / MAE)",
            "-" * 80,
            metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
            "",
             "6) ROLLING BACKTEST ÖZETİ (RMSE ortalama / std / kazanılan pencere)",
             "-" * 80,
             rolling_summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             "",
             "7) BAŞARI KRİTERİ KONTROLÜ (ANA METRİK RMSE + ROLLING KURALI)",
             "-" * 80,
             f"Tek split iyileşme oranı: {float(success_summary['single_split_improvement']):.4%}",
             f"Rolling pencere kazanma oranı: {float(success_summary['rolling_win_ratio']):.4%}",
             f"Rolling ortalama iyileşme: {float(success_summary['rolling_mean_improvement']):.4%}",
             f"Tek split RMSE: {float(success_summary['single_split_rmse']):.6f}",
             f"Rolling Hibrit RMSE ortalaması: {float(success_summary['rolling_hybrid_rmse_mean']):.6f}",
             f"Karar nedeni: {success_summary['decision_reason']}",
             f"Sonuç: {'PASS' if bool(success_summary['pass']) else 'FAIL'}",
             "",
             "8) KABUL EŞİĞİ / RESMİ HEDEF ÖZETİ",
             "-" * 80,
             f"Resmi metrik: {acceptance_summary['official_metric']}",
             f"Destek metrik: {acceptance_summary['support_metric']}",
             f"Stabil RMSE üst bant: {float(acceptance_summary['stable_rmse_upper']):.6f}",
             f"Kural sonucu: {'PASS' if bool(acceptance_summary['pass']) else 'FAIL'}",
             "",
             "9) DİAGNOSTİK BASELINE (SEED TEKRARLI KOŞU)",
             "-" * 80,
             diagnostic_baseline_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             "",
             "10) MODÜL BAZLI AYRIŞTIRMA VE KIRILMA NOKTASI",
             "-" * 80,
             module_breakdown_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             root_cause_report,
             "",
             "11) ABLATION KARŞILAŞTIRMA TABLOSU",
             "-" * 80,
             ablation_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             "",
             "12) STRES TESTİ SONUÇ TABLOSU (GERÇEK USD FİYATI ÜZERİNDEN)",
             "-" * 80,
             stress_table_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
             "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTez raporu oluşturuldu: {report_path}")


def run_pipeline(interval: str = "1d") -> PipelineResult:
    set_global_seed(DEFAULT_SEED)
    print("=== Modül 1: Veri Çekme ve Ön İşleme ===")
    df_raw = fetch_yfinance_data(interval=interval)
    quality_report = validate_data_quality(df_raw)
    if not bool(quality_report["quality_pass"]):
        raise ValueError(
            f"Data quality gate failed: jump_ratio_max={quality_report['jump_ratio_max']:.4f} exceeds acceptance policy."
        )
    df_model = compute_log_returns(df_raw)

    print("\n--- Ham Veri Zaman Serisi Özeti ---")
    plot_raw_data_summary(df_raw)

    print("\n--- Temel İstatistikler (Ham Veri) ---")
    basic_stats = df_raw.describe()
    print(basic_stats.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- Korelasyon Matrisi (Ham Veri) ---")
    corr_matrix = df_raw.corr()
    print(corr_matrix.to_string(float_format=lambda x: f"{x:.6f}"))

    plot_correlation_heatmap(corr_matrix)

    preprocessed = prepare_leakage_safe_splits(df_model)
    train_df, val_df, test_df = preprocessed.train, preprocessed.val, preprocessed.test
    validate_split_integrity(train_df, val_df, test_df)
    df_stationary = pd.concat([train_df, val_df, test_df])
    adf_results = preprocessed.stationarity

    print("\n--- Betimsel İstatistikler (Birinci Farkı Alınmış Seriler) ---")
    diff_stats = df_stationary.describe().T[["mean", "std", "min", "max"]]
    diff_stats["skewness"] = df_stationary.skew()
    diff_stats.columns = ["Ortalama (Mean)", "Standart Sapma (Std)", "Min", "Max", "Çarpıklık (Skewness)"]
    print(diff_stats.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Modül 2-3: ARIMAX + MLP + Hibrit + Benchmark ===")
    y_test, predictions, arimax_model, arimax_meta = run_single_window_models(train_df, val_df, test_df)
    arimax_coef_table_str = str(arimax_model.summary().tables[1])
    predictions_df = pd.DataFrame(predictions).reindex(y_test.index)
    residuals_df = pd.DataFrame(
        {"XGBoost": y_test - predictions_df["XGBoost"], "Hibrit ARIMAX-MLP": y_test - predictions_df["Hibrit ARIMAX-MLP"]},
        index=y_test.index,
    )

    print("\n=== Modül 4: Analiz ve Çıktılar ===")
    metrics_df = calculate_metrics(y_test, predictions)
    print("\nTest Seti Performans Tablosu (RMSE / MAE):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    hybrid_residuals = residuals_df["Hibrit ARIMAX-MLP"].dropna()
    ljung_box_pvalue = float(acorr_ljungbox(hybrid_residuals, lags=[10], return_df=True)["lb_pvalue"].iloc[0])
    arch_lm_pvalue = float(het_arch(hybrid_residuals)[1])
    diagnostics_df = pd.DataFrame(
        [{"Test": "Ljung-Box (lag=10)", "p-değeri": ljung_box_pvalue}, {"Test": "ARCH-LM", "p-değeri": arch_lm_pvalue}]
    )
    print("\nResidual Tanı Testleri (Hibrit Model):")
    print(diagnostics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    rolling_summary_df, rolling_window_results_df = run_rolling_backtest(df_raw=df_model, max_windows=4)
    success_summary = evaluate_success_criteria(metrics_df, rolling_window_results_df)
    acceptance_summary = {
        "official_metric": OFFICIAL_METRIC,
        "support_metric": SUPPORT_METRIC,
        "stable_rmse_upper": STABLE_RMSE_BAND_UPPER,
        "pass": bool(success_summary["pass"]),
    }
    diagnostic_baseline_df = run_diagnostic_baseline(df_model, seeds=REPEATED_SEEDS)
    module_breakdown_df = build_module_breakdown(y_test, predictions, rolling_summary_df)
    root_cause_report = find_first_breakpoint(module_breakdown_df, threshold=STABLE_RMSE_BAND_UPPER)
    ablation_df = run_ablation_experiments(y_test, predictions)
    print("\n=== Başarı Kriteri Kontrolü ===")
    print(
        f"Tek split iyileşme: {success_summary['single_split_improvement']:.2%} | "
        f"Rolling kazanma oranı: {success_summary['rolling_win_ratio']:.2%} | "
        f"PASS: {success_summary['pass']} | {success_summary['decision_reason']}"
    )
    print("\n=== Teşhis Özeti ===")
    print(root_cause_report)

    plot_module_visualizations(y_test=y_test, predictions_df=predictions_df, metrics_df=metrics_df, residuals_df=residuals_df)

    print("\n=== Modül 5: Karbon Stres Testi ===")
    # Stres testi, USD standardize edilmiş ham fiyat üzerinden üretilir.
    real_base_price = float(df_raw[TARGET_TICKER].iloc[-1])
    stress_table_df = run_stress_test(base_price=real_base_price)
    plot_stress_test_fan_chart(last_test_date=y_test.index[-1], base_price=real_base_price)
    print(stress_table_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    write_thesis_report(
        basic_stats=basic_stats,
        corr_matrix=corr_matrix,
        diff_stats=diff_stats,
        adf_results=adf_results,
        arimax_summary=str(arimax_model.summary()) + f"\n\nSelected exog: {arimax_meta['exog_name']}",
        arimax_coef_table_str=arimax_coef_table_str,
        diagnostics_df=diagnostics_df,
        metrics_df=metrics_df,
        rolling_summary_df=rolling_summary_df,
        success_summary=success_summary,
        acceptance_summary=acceptance_summary,
        diagnostic_baseline_df=diagnostic_baseline_df,
        module_breakdown_df=module_breakdown_df,
        ablation_df=ablation_df,
        root_cause_report=root_cause_report,
        stress_table_df=stress_table_df,
        report_path=Path.cwd() / "Tez_Bulgulari_Raporu.txt",
    )

    return PipelineResult(
        metrics_table=metrics_df,
        predictions=predictions_df,
        residuals=residuals_df,
        diagnostics=diagnostics_df,
        stress_table=stress_table_df,
        rolling_metrics=rolling_summary_df,
        success_summary=success_summary,
        acceptance_summary=acceptance_summary,
        diagnostic_baseline=diagnostic_baseline_df,
        module_breakdown=module_breakdown_df,
        ablation_table=ablation_df,
        root_cause_report=root_cause_report,
    )


if __name__ == "__main__":
    run_pipeline()
