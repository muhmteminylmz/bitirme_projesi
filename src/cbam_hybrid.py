"""ARIMAX + MLP hybrid pipeline for corporate carbon stress testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

TARGET_TICKER = "EREGL.IS"
CARBON_TICKER = "KRBN"
IRON_TICKER = "TIO=F"
XGBOOST_PARAMS = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5}
MODEL_COLORS = {
    "Baseline": "#6c7a89",
    "ARIMAX": "#4c78a8",
    "XGBoost": "#72b7b2",
    "Hibrit ARIMAX-MLP": "#8b0000",
}
DEFAULT_MODEL_COLOR = "#808080"


@dataclass
class PipelineResult:
    metrics_table: pd.DataFrame
    predictions: pd.DataFrame
    residuals: pd.DataFrame


def fetch_market_data(period: str = "5y") -> pd.DataFrame:
    """Download target and exogenous market data from yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - runtime import guard
        raise ImportError(
            "yfinance is required for Module 1. Install dependencies from requirements.txt"
        ) from exc

    tickers = [TARGET_TICKER, CARBON_TICKER, IRON_TICKER]
    data = yf.download(tickers=tickers, period=period, interval="1d", progress=False)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = close.rename(
        columns={
            TARGET_TICKER: "Y_EREGL",
            CARBON_TICKER: "X1_KRBN",
            IRON_TICKER: "X2_TIO",
        }
    )
    expected = ["Y_EREGL", "X1_KRBN", "X2_TIO"]
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
    y_train_reset = y_train.reset_index(drop=True)
    x2_train_reset = x2_train.reset_index(drop=True)

    model = SARIMAX(
        y_train_reset,
        exog=x2_train_reset,
        order=(1, 1, 1),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def fit_xgboost_regressor(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit XGBoost benchmark with requested baseline parameters."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - runtime import guard
        raise ImportError(
            "xgboost is required for benchmark modeling. Install dependencies from requirements.txt"
        ) from exc

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=XGBOOST_PARAMS["n_estimators"],
        learning_rate=XGBOOST_PARAMS["learning_rate"],
        max_depth=XGBOOST_PARAMS["max_depth"],
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def build_residual_training_frame(x1_train: pd.Series, residuals: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Create MLP training set: [X1, lagged residual] -> residual."""
    frame = pd.DataFrame({"x1": x1_train, "residual": residuals})
    frame["residual_lag1"] = frame["residual"].shift(1)
    frame = frame.dropna()
    X = frame[["x1", "residual_lag1"]]
    y = frame["residual"]
    return X, y


def train_mlp(X_train: pd.DataFrame, y_train: pd.Series, epochs: int = 200, batch_size: int = 16):
    """Train TensorFlow/Keras MLP model."""
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, Input
    except ImportError as exc:  # pragma: no cover - runtime import guard
        raise ImportError(
            "TensorFlow is required for Module 4 (MLP). Install dependencies from requirements.txt"
        ) from exc

    tf.random.set_seed(42)
    np.random.seed(42)

    model = Sequential(
        [
            Input(shape=(2,)),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train.values,
        y_train.values,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )
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


def safe_mape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-8) -> float:
    """Calculate MAPE (%) with epsilon clipping to prevent division by zero near zero targets."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.clip(np.abs(y_true_arr), eps, None)
    return float(np.mean(np.abs((y_true_arr - y_pred_arr) / denominator)) * 100)


def calculate_metrics(y_true: pd.Series, predictions: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute RMSE, MAE, MAPE for all model predictions."""
    rows = []
    for model_name, pred in predictions.items():
        aligned_pred = pred.reindex(y_true.index)
        rows.append(
            {
                "Model": model_name,
                "RMSE": float(np.sqrt(mean_squared_error(y_true, aligned_pred))),
                "MAE": float(mean_absolute_error(y_true, aligned_pred)),
                "MAPE": safe_mape(y_true, aligned_pred),
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def plot_module_6_visualizations(
    y_test: pd.Series,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
) -> None:
    """Create academic-quality benchmark comparison visualizations (Module 6)."""
    print("\n=== Modül 6: Akademik Görselleştirme ===")
    sns.set_style("whitegrid")

    # Grafik 1: RMSE bar chart (ayrı figür)
    plt.figure(figsize=(12, 5))
    bar_colors = [MODEL_COLORS.get(model, DEFAULT_MODEL_COLOR) for model in metrics_df["Model"]]
    ax = plt.gca()
    sns.barplot(data=metrics_df, x="Model", y="RMSE", palette=bar_colors, ax=ax)
    ax.set_title("Grafik 1 - Model Performans Karşılaştırması (RMSE)", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", labelrotation=15)
    plt.tight_layout()
    plt.show()

    # Grafik 2: Test seti tahmin çizgileri (ayrı figür)
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    ax.plot(y_test.index, y_test.values, label="Gerçek Y", color="black", linewidth=2.6)
    for col in predictions_df.columns:
        ax.plot(y_test.index, predictions_df[col].values, label=col, linewidth=1.8)
    ax.set_title("Grafik 2 - Test Seti Üzerinde Zaman Serisi Tahminleri", fontsize=14)
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Y (Ölçeklenmiş/Durağanlaştırılmış)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Grafik 3: Residual dağılım karşılaştırması (Hibrit vs XGBoost, ayrı figür)
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    sns.kdeplot(
        residuals_df["Hibrit ARIMAX-MLP"],
        fill=True,
        alpha=0.35,
        label="Hibrit ARIMAX-MLP Residual",
        ax=ax,
        color="#8b0000",
    )
    sns.kdeplot(
        residuals_df["XGBoost"],
        fill=True,
        alpha=0.35,
        label="XGBoost Residual",
        ax=ax,
        color=MODEL_COLORS["XGBoost"],
    )
    ax.axvline(0, linestyle="--", color="black", linewidth=1.2)
    ax.set_title("Grafik 3 - Residual Dağılımı (Gerçek - Tahmin)", fontsize=14)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Yoğunluk")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Grafik 4 (Literatür): Gerçek vs Tahmin (Parity/Scatter)
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    min_val = float(min(y_test.min(), predictions_df.min().min()))
    max_val = float(max(y_test.max(), predictions_df.max().max()))
    for model_name, color in MODEL_COLORS.items():
        if model_name in predictions_df.columns:
            ax.scatter(
                y_test.values,
                predictions_df[model_name].values,
                alpha=0.7,
                s=35,
                label=model_name,
                color=color,
            )
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1.2, label="y=x")
    ax.set_title("Grafik 4 - Parity Plot (Gerçek vs Tahmin)", fontsize=14)
    ax.set_xlabel("Gerçek Y")
    ax.set_ylabel("Tahmin Y")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Grafik 5 (Literatür): Residual zaman serisi
    plt.figure(figsize=(14, 5))
    ax = plt.gca()
    ax.plot(residuals_df.index, residuals_df["Hibrit ARIMAX-MLP"], label="Hibrit Residual", color="#8b0000", linewidth=1.8)
    ax.plot(residuals_df.index, residuals_df["XGBoost"], label="XGBoost Residual", color=MODEL_COLORS["XGBoost"], linewidth=1.8, alpha=0.85)
    ax.axhline(0, linestyle="--", color="black", linewidth=1.2)
    ax.set_title("Grafik 5 - Residual Zaman Serisi Karşılaştırması", fontsize=14)
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Residual")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def run_pipeline(period: str = "5y") -> PipelineResult:
    """Execute full 6-module benchmark + hybrid pipeline."""
    print("\n=== Modül 1: Veri Çekme, Temizleme, Normalizasyon ve ADF Testi ===")
    raw_df = fetch_market_data(period=period)
    scaled_df, _ = clean_and_scale_data(raw_df)
    stationary_df, stationarity = enforce_stationarity(scaled_df)

    for col, is_stat in stationarity.items():
        print(f"{col} durağan mı? {'Evet' if is_stat else 'Hayır (diff uygulandı)'}")

    print("\n=== Modül 2: ARIMAX Eğitimi ve Test Tahmini ===")
    train_df, test_df = train_test_split_time_series(stationary_df, train_ratio=0.8)
    y_train, y_test = train_df["Y_EREGL"], test_df["Y_EREGL"]
    x1_train, x1_test = train_df["X1_KRBN"], test_df["X1_KRBN"]
    x2_train, x2_test = train_df["X2_TIO"], test_df["X2_TIO"]

    arimax_fit = fit_arimax(y_train, x2_train)
    arimax_test_pred = arimax_fit.get_forecast(
        steps=len(y_test),
        exog=x2_test.reset_index(drop=True),
    ).predicted_mean
    arimax_test_pred = pd.Series(arimax_test_pred.to_numpy(), index=y_test.index, name="ARIMAX")

    print("\n=== Modül 3: ARIMAX Residual Çıkarımı ===")
    fitted_train = pd.Series(arimax_fit.fittedvalues.to_numpy(), index=y_train.index, name="arimax_fitted")
    residuals_train = (y_train - fitted_train).dropna()

    print("\n=== Modül 4: MLP ile Residual Modelleme ===")
    mlp_X_train, mlp_y_train = build_residual_training_frame(x1_train.reindex(residuals_train.index), residuals_train)
    mlp_model = train_mlp(mlp_X_train, mlp_y_train)

    print("\n=== Modül 5: Benchmark + Hibrit Değerlendirme ===")
    # Baseline Linear Regression
    X_train_bench = train_df[["X1_KRBN", "X2_TIO"]]
    X_test_bench = test_df[["X1_KRBN", "X2_TIO"]]

    baseline_model = LinearRegression()
    baseline_model.fit(X_train_bench, y_train)
    baseline_pred = pd.Series(
        baseline_model.predict(X_test_bench),
        index=y_test.index,
        name="Baseline",
    )

    # XGBoost Regressor
    xgb_model = fit_xgboost_regressor(X_train_bench, y_train)
    xgb_pred = pd.Series(xgb_model.predict(X_test_bench), index=y_test.index, name="XGBoost")

    # Hybrid ARIMAX-MLP
    mlp_residual_test_pred = forecast_mlp_residuals(
        mlp_model=mlp_model,
        x1_test=x1_test,
        last_train_residual=float(residuals_train.iloc[-1]),
    )
    hybrid_pred = arimax_test_pred.add(mlp_residual_test_pred, fill_value=0.0)
    hybrid_pred.name = "Hibrit ARIMAX-MLP"

    predictions = {
        "Baseline": baseline_pred,
        "ARIMAX": arimax_test_pred,
        "XGBoost": xgb_pred,
        "Hibrit ARIMAX-MLP": hybrid_pred,
    }
    predictions_df = pd.DataFrame(predictions).reindex(y_test.index)
    residuals_df = pd.DataFrame(
        {
            "XGBoost": y_test - predictions_df["XGBoost"],
            "Hibrit ARIMAX-MLP": y_test - predictions_df["Hibrit ARIMAX-MLP"],
        },
        index=y_test.index,
    )

    metrics_df = calculate_metrics(y_test, predictions)
    print("\nTest Seti Performans Tablosu (RMSE / MAE / MAPE):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    plot_module_6_visualizations(
        y_test=y_test,
        predictions_df=predictions_df,
        metrics_df=metrics_df,
        residuals_df=residuals_df,
    )

    return PipelineResult(
        metrics_table=metrics_df,
        predictions=predictions_df,
        residuals=residuals_df,
    )
