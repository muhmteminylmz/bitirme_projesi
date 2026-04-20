"""ARIMAX + MLP hybrid pipeline for corporate carbon stress testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
except Exception as exc:  # pragma: no cover - import guard for runtime environment
    raise ImportError(
        "TensorFlow is required for Module 4 (MLP). Install dependencies from requirements.txt"
    ) from exc


TARGET_TICKER = "EREGL.IS"
CARBON_TICKER = "KEA"
IRON_TICKER = "TIO=F"


@dataclass
class PipelineResult:
    arimax_rmse: float
    hybrid_rmse: float
    arimax_test_predictions: pd.Series
    hybrid_test_predictions: pd.Series


def fetch_market_data(period: str = "5y") -> pd.DataFrame:
    """Download target and exogenous market data from yfinance."""
    tickers = [TARGET_TICKER, CARBON_TICKER, IRON_TICKER]
    data = yf.download(tickers=tickers, period=period, interval="1d", progress=False)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = close.rename(
        columns={
            TARGET_TICKER: "Y_EREGL",
            CARBON_TICKER: "X1_KEA",
            IRON_TICKER: "X2_TIO",
        }
    )
    expected = ["Y_EREGL", "X1_KEA", "X2_TIO"]
    missing = [c for c in expected if c not in close.columns]
    if missing:
        raise ValueError(f"Missing downloaded columns: {missing}")

    return close[expected].sort_index()


def clean_and_scale_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Apply ffill/bfill only and MinMax scale to [0,1]."""
    cleaned = df.ffill().bfill()
    if cleaned.isna().any().any():
        raise ValueError("NaN values remain after ffill/bfill.")

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(cleaned)
    scaled_df = pd.DataFrame(scaled_values, index=cleaned.index, columns=cleaned.columns)
    return scaled_df, scaler


def enforce_stationarity(df: pd.DataFrame, alpha: float = 0.05) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """Run ADF tests and difference non-stationary series."""
    stationary = pd.DataFrame(index=df.index)
    was_stationary: Dict[str, bool] = {}

    for col in df.columns:
        series = df[col].dropna()
        p_value = adfuller(series)[1]
        is_stationary = p_value < alpha
        was_stationary[col] = is_stationary
        stationary[col] = df[col] if is_stationary else df[col].diff()

    stationary = stationary.dropna()
    return stationary, was_stationary


def train_test_split_time_series(df: pd.DataFrame, train_ratio: float = 0.8):
    """Chronological train/test split."""
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def fit_arimax(y_train: pd.Series, x2_train: pd.Series):
    """Fit ARIMAX model with X2 as exogenous variable."""
    model = SARIMAX(
        y_train,
        exog=x2_train,
        order=(1, 0, 1),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def build_residual_training_frame(x1_train: pd.Series, residuals: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Create MLP training set: [X1, lagged residual] -> residual."""
    frame = pd.DataFrame({"x1": x1_train, "residual": residuals})
    frame["residual_lag1"] = frame["residual"].shift(1)
    frame = frame.dropna()
    X = frame[["x1", "residual_lag1"]]
    y = frame["residual"]
    return X, y


def train_mlp(X_train: pd.DataFrame, y_train: pd.Series, epochs: int = 80, batch_size: int = 16):
    """Train TensorFlow/Keras MLP model."""
    tf.random.set_seed(42)
    np.random.seed(42)

    model = Sequential(
        [
            Dense(32, activation="relu", input_shape=(2,)),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train.values, y_train.values, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def forecast_mlp_residuals(mlp_model, x1_test: pd.Series, last_train_residual: float) -> pd.Series:
    """Recursive residual forecasting for test horizon."""
    preds = []
    lag_residual = float(last_train_residual)

    for idx, x1_value in x1_test.items():
        features = np.array([[x1_value, lag_residual]], dtype=float)
        pred = float(mlp_model.predict(features, verbose=0).flatten()[0])
        preds.append((idx, pred))
        lag_residual = pred

    return pd.Series({idx: val for idx, val in preds}, name="mlp_residual_pred")


def run_pipeline(period: str = "5y") -> PipelineResult:
    """Execute full 5-module hybrid pipeline."""
    print("\n=== Modül 1: Veri Çekme, Temizleme, Normalizasyon ve ADF Testi ===")
    raw_df = fetch_market_data(period=period)
    scaled_df, _ = clean_and_scale_data(raw_df)
    stationary_df, stationarity = enforce_stationarity(scaled_df)

    for col, is_stat in stationarity.items():
        print(f"{col} durağan mı? {'Evet' if is_stat else 'Hayır (diff uygulandı)'}")

    print("\n=== Modül 2: ARIMAX Eğitimi ve Test Tahmini ===")
    train_df, test_df = train_test_split_time_series(stationary_df, train_ratio=0.8)
    y_train, y_test = train_df["Y_EREGL"], test_df["Y_EREGL"]
    x1_train, x1_test = train_df["X1_KEA"], test_df["X1_KEA"]
    x2_train, x2_test = train_df["X2_TIO"], test_df["X2_TIO"]

    arimax_fit = fit_arimax(y_train, x2_train)
    arimax_test_pred = arimax_fit.get_forecast(steps=len(y_test), exog=x2_test).predicted_mean

    print("\n=== Modül 3: ARIMAX Residual Çıkarımı ===")
    fitted_train = arimax_fit.fittedvalues.reindex(y_train.index)
    residuals_train = (y_train - fitted_train).dropna()

    print("\n=== Modül 4: MLP ile Residual Modelleme ===")
    mlp_X_train, mlp_y_train = build_residual_training_frame(x1_train.reindex(residuals_train.index), residuals_train)
    mlp_model = train_mlp(mlp_X_train, mlp_y_train)

    print("\n=== Modül 5: Hibrit Birleşim ve RMSE Değerlendirme ===")
    mlp_residual_test_pred = forecast_mlp_residuals(
        mlp_model=mlp_model,
        x1_test=x1_test,
        last_train_residual=float(residuals_train.iloc[-1]),
    )

    hybrid_pred = arimax_test_pred.add(mlp_residual_test_pred, fill_value=0.0)
    arimax_rmse = float(np.sqrt(mean_squared_error(y_test, arimax_test_pred)))
    hybrid_rmse = float(np.sqrt(mean_squared_error(y_test, hybrid_pred.reindex(y_test.index))))

    print(f"ARIMAX RMSE : {arimax_rmse:.6f}")
    print(f"Hibrit RMSE : {hybrid_rmse:.6f}")

    return PipelineResult(
        arimax_rmse=arimax_rmse,
        hybrid_rmse=hybrid_rmse,
        arimax_test_predictions=arimax_test_pred,
        hybrid_test_predictions=hybrid_pred,
    )
