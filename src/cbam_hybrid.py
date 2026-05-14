"""ARIMAX + MLP hybrid pipeline for corporate carbon stress testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yfinance as yf
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
sns.set_style("whitegrid")

TARGET_TICKER = "EREGL.IS"
CARBON_TICKER = "KEUA"
IRON_TICKER = "TIO=F"
FX_TICKER = "USDTRY=X"
XGBOOST_PARAMS = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5}
MLP_PARAMS = {"epochs": 200, "batch_size": 16, "validation_split": 0.2, "patience": 10}
MODEL_COLORS = {
    "Baseline": "#6c7a89",
    "ARIMAX": "#4c78a8",
    "XGBoost": "#72b7b2",
    "LSTM": "#f58518",
    "Hibrit ARIMAX-MLP": "#8b0000",
}
DEFAULT_MODEL_COLOR = "#808080"
STRESS_BANDS = {"S1 (+%30)": 0.30, "S2 (+%60)": 0.60, "S3 (+%100)": 1.00}
STABLE_RMSE_BAND_UPPER = 0.05
VALID_FEATURE_MODES = ("standard", "minimal")


@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    stationarity: Dict
    scaler: MinMaxScaler


@dataclass
class PipelineResult:
    metrics_table: pd.DataFrame
    predictions: pd.DataFrame
    residuals: pd.DataFrame
    diagnostics: pd.DataFrame
    stress_table: pd.DataFrame
    walk_forward_metrics: pd.DataFrame


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
    return close_df


def clean_and_scale_data(df: pd.DataFrame, fit_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    cleaned = df.copy().sort_index().ffill().bfill()
    scaler = MinMaxScaler()
    if fit_df is not None:
        fit_cleaned = fit_df.copy().sort_index().ffill().bfill()
        scaler.fit(fit_cleaned)
    else:
        scaler.fit(cleaned)
    scaled = pd.DataFrame(scaler.transform(cleaned), columns=cleaned.columns, index=cleaned.index)
    return scaled, scaler


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).iloc[1:]


def enforce_stationarity(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    stationary_df = df.copy()
    adf_results: Dict[str, Dict[str, float]] = {}

    def safe_adf_pvalue(series: pd.Series) -> float:
        s = series.dropna()
        if s.empty or s.nunique() <= 1:
            return 1.0
        return float(adfuller(s)[1])

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
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    """Apply ADF-based differencing determined on train; preserve val/test lengths."""

    def _adf_p(series: pd.Series) -> float:
        s = series.dropna()
        if s.empty or s.nunique() <= 1:
            return 1.0
        return float(adfuller(s)[1])

    stationarity: Dict[str, Dict] = {}
    train_out = train.copy()
    val_out = val.copy()
    test_out = test.copy()

    for col in train.columns:
        initial_p = _adf_p(train[col])
        if initial_p > 0.05:
            train_out[col] = train[col].diff()
            # Difference val using last train row as anchor so no rows are lost
            val_anchored = pd.concat([train[[col]].iloc[[-1]], val[[col]]]).diff().iloc[1:]
            val_out[col] = val_anchored.values
            # Difference test using last val row as anchor
            test_anchored = pd.concat([val[[col]].iloc[[-1]], test[[col]]]).diff().iloc[1:]
            test_out[col] = test_anchored.values
            stationarity[col] = {"differenced": True, "initial_p": initial_p}
        else:
            stationarity[col] = {"differenced": False, "initial_p": initial_p}

    return train_out.dropna(), val_out, test_out, stationarity


def train_test_split_time_series(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_size = int(n * train_ratio)
    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()


def train_val_test_split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def prepare_leakage_safe_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> DataSplits:
    """Split, scale (fit on train only), and apply stationarity policy – all leakage-safe."""
    train_raw, val_raw, test_raw = train_val_test_split_time_series(df, train_ratio, val_ratio, test_ratio)
    scaled_train, scaler = clean_and_scale_data(train_raw)
    scaled_val, _ = clean_and_scale_data(val_raw, fit_df=train_raw)
    scaled_test, _ = clean_and_scale_data(test_raw, fit_df=train_raw)
    train_s, val_s, test_s, stationarity = apply_stationarity_policy(scaled_train, scaled_val, scaled_test)
    return DataSplits(train=train_s, val=val_s, test=test_s, stationarity=stationarity, scaler=scaler)


def build_expanding_windows(
    df: pd.DataFrame,
    min_train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
    max_windows: int = None,
) -> list:
    """Return list of (train, val, test) tuples using an expanding-window (walk-forward) scheme."""
    windows = []
    train_end = min_train_size
    while True:
        val_end = train_end + val_size
        test_end = val_end + test_size
        if test_end > len(df):
            break
        windows.append((df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:test_end].copy()))
        if max_windows is not None and len(windows) >= max_windows:
            break
        train_end += step_size
    return windows


def validate_data_quality(df: pd.DataFrame) -> Dict:
    """Run basic data-quality checks on a raw price panel."""
    has_nan = bool(df.isna().any().any())
    has_nonpositive = bool((df <= 0).any().any())
    quality_pass = not has_nan and not has_nonpositive
    return {
        "quality_pass": quality_pass,
        "has_nan": has_nan,
        "has_nonpositive": has_nonpositive,
        "shape": df.shape,
    }


def validate_split_integrity(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise ValueError if any two splits share index values."""
    if not train.index.intersection(val.index).empty:
        raise ValueError("Train ve val indeksleri örtüşüyor.")
    if not train.index.intersection(test.index).empty:
        raise ValueError("Train ve test indeksleri örtüşüyor.")
    if not val.index.intersection(test.index).empty:
        raise ValueError("Val ve test indeksleri örtüşüyor.")


def fit_sarimax(y: pd.Series, exog: pd.Series) -> SARIMAXResultsWrapper:
    print("\nOptimal ARIMA parametreleri aranıyor...")
    auto_model = auto_arima(
        y,
        exogenous=exog,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        trace=True,
        error_action="ignore",
        max_p=5,
        max_q=5,
        d=0,
    )
    order = auto_model.order
    print(f"Seçilen ARIMA order: {order}")

    model = SARIMAX(y, exog=exog, order=order, enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(disp=False)
    print(fitted_model.summary())
    return fitted_model


def fit_xgboost_regressor(X: pd.DataFrame, y: pd.Series):
    from xgboost import XGBRegressor

    model = XGBRegressor(**XGBOOST_PARAMS, random_state=42)
    model.fit(X, y)
    return model


def build_residual_training_frame(
    x1_train: pd.Series,
    residuals_train: pd.Series,
    x2_train: pd.Series = None,
    feature_mode: str = "standard",
) -> Tuple[pd.DataFrame, pd.Series]:
    if feature_mode not in VALID_FEATURE_MODES:
        raise ValueError(f"Geçersiz feature_mode: '{feature_mode}'. Kabul edilenler: {VALID_FEATURE_MODES}")

    if feature_mode == "minimal":
        X = pd.DataFrame(
            {
                "x1": x1_train,
                "x1_lag1": x1_train.shift(1),
                "residual_lag1": residuals_train.shift(1),
            }
        )
        X = X.dropna()
        return X, residuals_train.loc[X.index]

    # "standard" feature set (includes x2 features when x2_train is provided)
    features: Dict = {
        "x1": x1_train,
        "x1_lag1": x1_train.shift(1),
        "x1_change_lag1": x1_train.diff().shift(1),
        "residual_lag1": residuals_train.shift(1),
        "residual_lag5": residuals_train.shift(5),
        "residual_roll_mean_3": residuals_train.rolling(3).mean(),
        "residual_roll_std_3": residuals_train.rolling(3).std(),
        # 7-day rolling features produce 6 leading NaN → drives dropna boundary
        "x1_roll_mean_7": x1_train.rolling(7).mean(),
        "x1_roll_std_7": x1_train.rolling(7).std(),
    }

    if x2_train is not None:
        features["x2"] = x2_train
        features["x2_lag1"] = x2_train.shift(1)
        features["x2_change_lag1"] = x2_train.diff().shift(1)
        features["x1_x2_interaction"] = x1_train * x2_train

    X = pd.DataFrame(features)
    X = X.dropna()
    y = residuals_train.loc[X.index]
    return X, y


def build_and_train_mlp(X_train: np.ndarray, y_train: np.ndarray):
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=300,
        batch_size=8,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )
    return model


def forecast_mlp_residuals(
    mlp_model,
    x1_train: pd.Series,
    residuals_train: pd.Series,
    x1_test: pd.Series,
    x2_train: pd.Series = None,
    x2_test: pd.Series = None,
) -> pd.Series:
    if len(x1_train) < 10:
        raise ValueError("MLP residual tahmini için x1_train en az 10 gözlem içermelidir.")
    if len(residuals_train) < 5:
        raise ValueError("MLP residual tahmini için residuals_train en az 5 gözlem içermelidir.")

    preds = []
    x1_history = x1_train.astype(float).tolist()
    residual_history = residuals_train.astype(float).tolist()
    x2_history = x2_train.astype(float).tolist() if x2_train is not None else None

    for i in range(len(x1_test)):
        curr_x1 = float(x1_test.iloc[i])
        x1_ctx7 = x1_history[-6:] + [curr_x1]

        features = [
            curr_x1,
            x1_history[-1],
            x1_history[-1] - x1_history[-2],
            residual_history[-1],
            residual_history[-5],
            float(np.mean(residual_history[-3:])),
            float(np.std(residual_history[-3:], ddof=1)) if len(residual_history) >= 3 else 0.0,
            float(np.mean(x1_ctx7)),
            float(np.std(x1_ctx7, ddof=1)) if len(x1_ctx7) >= 2 else 0.0,
        ]

        if x2_history is not None and x2_test is not None:
            curr_x2 = float(x2_test.iloc[i])
            features += [
                curr_x2,
                x2_history[-1],
                x2_history[-1] - x2_history[-2],
                curr_x1 * curr_x2,
            ]

        X_input = np.array([features], dtype=float)
        pred_res = float(mlp_model.predict(X_input, verbose=0).ravel()[0])
        preds.append(pred_res)

        x1_history.append(curr_x1)
        residual_history.append(pred_res)
        if x2_history is not None and x2_test is not None:
            x2_history.append(float(x2_test.iloc[i]))

    return pd.Series(preds, index=x1_test.index)


def create_lstm_dataset(series, window_size: int = 5):
    X = []
    y = []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_lstm_model(train_series: pd.Series):
    window_size = 5
    X_train, y_train = create_lstm_dataset(train_series.values, window_size)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential([LSTM(16, input_shape=(window_size, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    return model, window_size


def forecast_lstm(model, full_series: np.ndarray, train_size: int, window_size: int):
    predictions = []
    for i in range(train_size, len(full_series)):
        x_input = full_series[i - window_size : i]
        x_input = x_input.reshape(1, window_size, 1)
        pred = model.predict(x_input, verbose=0)[0][0]
        predictions.append(pred)
    return np.array(predictions)


def calculate_metrics(y_true: pd.Series, predictions_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    metrics = []
    for model_name, y_pred in predictions_dict.items():
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        metrics.append({"Model": model_name, "RMSE": rmse, "MAE": mae})
    return pd.DataFrame(metrics).sort_values(by="RMSE").reset_index(drop=True)


def train_hybrid_combiner(
    y_val: pd.Series,
    arimax_val: pd.Series,
    mlp_val: pd.Series,
):
    """Train a linear meta-learner that combines ARIMAX and MLP predictions."""
    X = pd.DataFrame({"arimax": arimax_val, "mlp": mlp_val})
    model = LinearRegression()
    model.fit(X, y_val)
    return model


def apply_hybrid_combiner(model, arimax_pred: pd.Series, mlp_pred: pd.Series) -> pd.Series:
    """Apply the trained linear combiner to new predictions."""
    X = pd.DataFrame({"arimax": arimax_pred, "mlp": mlp_pred})
    return pd.Series(model.predict(X), index=arimax_pred.index)


def evaluate_success_criteria(
    metrics_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    stable_rmse_upper: float = None,
) -> Dict:
    """Evaluate whether the hybrid model meets success criteria."""
    hybrid_rmse = float(metrics_df.loc[metrics_df["Model"] == "Hibrit ARIMAX-MLP", "RMSE"].iloc[0])
    baseline_rmse = float(metrics_df.loc[metrics_df["Model"] == "Baseline", "RMSE"].iloc[0])
    single_split_improvement = baseline_rmse - hybrid_rmse

    wins = 0
    total = 0
    for _window, grp in rolling_df.groupby("window"):
        h_rmse = grp.loc[grp["Model"] == "Hibrit ARIMAX-MLP", "RMSE"]
        b_rmse = grp.loc[grp["Model"] == "Baseline", "RMSE"]
        if h_rmse.empty or b_rmse.empty:
            continue
        wins += int(float(h_rmse.iloc[0]) < float(b_rmse.iloc[0]))
        total += 1

    rolling_win_ratio = wins / total if total > 0 else 0.0
    passed = (single_split_improvement > 0) and (rolling_win_ratio > 0)
    return {
        "single_split_improvement": single_split_improvement,
        "rolling_win_ratio": rolling_win_ratio,
        "pass": passed,
    }


def run_ablation_experiments(
    y_true: pd.Series,
    predictions: Dict[str, pd.Series],
) -> pd.DataFrame:
    """Return ablation table sorted by RMSE ascending."""
    rows = []
    for variant, preds in predictions.items():
        aligned_true, aligned_pred = y_true.align(preds, join="inner")
        rmse = float(np.sqrt(mean_squared_error(aligned_true, aligned_pred)))
        mae = float(mean_absolute_error(aligned_true, aligned_pred))
        rows.append({"Variant": variant, "RMSE": rmse, "MAE": mae})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def build_module_breakdown(
    y_true: pd.Series,
    predictions: Dict[str, pd.Series],
    rolling_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-module RMSE breakdown table."""
    rows = []
    for model_name, preds in predictions.items():
        aligned_true, aligned_pred = y_true.align(preds, join="inner")
        rmse = float(np.sqrt(mean_squared_error(aligned_true, aligned_pred)))
        rows.append({"module": model_name, "rmse": rmse})
    return pd.DataFrame(rows).reset_index(drop=True)


def find_first_breakpoint(breakdown: pd.DataFrame, threshold: float) -> str:
    """Return a Turkish-language report string for the first module exceeding threshold RMSE."""
    bad = breakdown[breakdown["rmse"] > threshold]
    if bad.empty:
        return f"İlk kırılma noktası bulunamadı (eşik={threshold:.4f})."
    first_row = bad.iloc[0]
    return f"İlk kırılma noktası: {first_row['module']} (RMSE={first_row['rmse']:.4f}, eşik={threshold:.4f})"


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

    plt.title("Şekil 4.4: Test Sonrası 30 Gün Karbon Stres Testi Yelpaze Grafiği (Gerçek Fiyat)", pad=15, fontsize=12, fontweight="bold")
    plt.xlabel("Tarih")
    plt.ylabel("Hisse Fiyatı (TL)")
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
    stress_table_df: pd.DataFrame,
    report_path: Path,
    walk_forward_metrics_df: pd.DataFrame = None,
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
        "2) ADF TEST SONUÇLARI",
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
            "TABLO 4.2: ARIMAX KATSAYI TABLOSU (AR, MA ve X2 / Demir Cevheri + Kur)",
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
            "6) STRES TESTİ SONUÇ TABLOSU (GERÇEK TL FİYATI ÜZERİNDEN)",
            "-" * 80,
            stress_table_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
            "",
        ]
    )

    if walk_forward_metrics_df is not None and not walk_forward_metrics_df.empty:
        lines.extend(
            [
                "7) WALK-FORWARD VALIDATION SONUÇLARI (KUR FARKI DÜZELTMESİ SONRASI)",
                "-" * 80,
                walk_forward_metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTez raporu oluşturuldu: {report_path}")


def walk_forward_validation(
    df_stationary: pd.DataFrame,
    target_col: str,
    carbon_col: str,
    iron_col: str,
    fx_col: str,
    initial_train_size: int,
    step: int = 60,
    max_windows: int = 5,
) -> pd.DataFrame:
    """Run walk-forward validation and return per-window metrics."""
    rows = []
    n = len(df_stationary)
    train_end = initial_train_size
    window_idx = 0

    while train_end < n and window_idx < max_windows:
        test_end = min(train_end + step, n)
        train = df_stationary.iloc[:train_end]
        test = df_stationary.iloc[train_end:test_end]

        if len(test) == 0:
            break

        y_tr = train[target_col]
        y_te = test[target_col]
        exog_cols = [iron_col, fx_col]
        exog_tr = train[exog_cols]
        exog_te = test[exog_cols]
        X_bench_tr = train[[carbon_col] + exog_cols]
        X_bench_te = test[[carbon_col] + exog_cols]

        try:
            # ARIMAX
            arimax_m = fit_sarimax(y_tr, exog=exog_tr)
            arimax_pred = pd.Series(
                arimax_m.predict(start=len(y_tr), end=len(y_tr) + len(y_te) - 1, exog=exog_te).values,
                index=y_te.index,
            )
            residuals_tr = pd.Series(arimax_m.resid, index=y_tr.index)

            # XGBoost
            xgb_m = fit_xgboost_regressor(X_bench_tr, y_tr)
            xgb_pred = pd.Series(xgb_m.predict(X_bench_te), index=y_te.index)

            # Hybrid ARIMAX-MLP
            X_mlp_tr, y_mlp_tr = build_residual_training_frame(
                train[carbon_col], residuals_tr, x2_train=train[fx_col]
            )
            mlp_m = build_and_train_mlp(X_mlp_tr.values, y_mlp_tr.values)
            mlp_resid_pred = forecast_mlp_residuals(
                mlp_m,
                x1_train=train[carbon_col],
                residuals_train=residuals_tr,
                x1_test=test[carbon_col],
                x2_train=train[fx_col],
                x2_test=test[fx_col],
            )
            hybrid_pred = arimax_pred.add(mlp_resid_pred, fill_value=0.0)

            for name, pred in [
                ("ARIMAX", arimax_pred),
                ("XGBoost", xgb_pred),
                ("Hibrit ARIMAX-MLP", hybrid_pred),
            ]:
                aligned_true, aligned_pred = y_te.align(pred, join="inner")
                rmse = float(np.sqrt(mean_squared_error(aligned_true, aligned_pred)))
                mae = float(mean_absolute_error(aligned_true, aligned_pred))
                rows.append({"window": window_idx + 1, "Model": name, "RMSE": rmse, "MAE": mae})

            print(f"Walk-forward pencere {window_idx + 1}/{max_windows} tamamlandı (test boyutu={len(test)}).")
        except Exception as exc:
            print(f"Walk-forward pencere {window_idx + 1} atlandı: {exc}")

        train_end += step
        window_idx += 1

    return pd.DataFrame(rows)


def plot_walk_forward_metrics(wf_metrics: pd.DataFrame) -> None:
    """Plot per-fold RMSE for walk-forward validation results."""
    if wf_metrics.empty:
        return
    pivot = wf_metrics.pivot_table(index="window", columns="Model", values="RMSE")
    plt.figure(figsize=(10, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=col, color=MODEL_COLORS.get(col, DEFAULT_MODEL_COLOR))
    plt.title("Şekil 4.5: Walk-Forward Validation – Pencere Bazında RMSE", pad=15, fontsize=12, fontweight="bold")
    plt.xlabel("Pencere")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Grafik_5_WalkForward_Fold_Metrics.png", dpi=300)
    plt.close()


def run_pipeline(interval: str = "1d") -> PipelineResult:
    print("=== Modül 1: Veri Çekme ve Ön İşleme ===")
    df_raw = fetch_yfinance_data(interval=interval)

    # --- Kur Farkı Düzeltmesi: EREGL.IS (TRY) → USD ---
    df_raw["EREGL.IS_USD"] = df_raw[TARGET_TICKER] / df_raw[FX_TICKER]
    usd_target = "EREGL.IS_USD"

    print("\n--- Veri kalitesi kontrolü ---")
    quality = validate_data_quality(df_raw)
    print(f"Kalite geçti: {quality['quality_pass']}")

    # Korelasyon ve betimsel istatistikler ham TRY fiyatı üzerinden
    analysis_cols = [TARGET_TICKER, CARBON_TICKER, IRON_TICKER, FX_TICKER]
    print("\n--- Temel İstatistikler (Ham Veri) ---")
    basic_stats = df_raw[analysis_cols].describe()
    print(basic_stats.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- Korelasyon Matrisi (Ham Veri) ---")
    corr_matrix = df_raw[analysis_cols].corr()
    print(corr_matrix.to_string(float_format=lambda x: f"{x:.6f}"))

    plot_correlation_heatmap(corr_matrix)

    # Model DataFrame: USD-normalized target + exogenous features
    df_model = df_raw[[usd_target, CARBON_TICKER, IRON_TICKER, FX_TICKER]].copy()
    df_model = df_model.rename(columns={usd_target: TARGET_TICKER})

    df_scaled, _ = clean_and_scale_data(df_model)
    df_stationary, adf_results = enforce_stationarity(df_scaled)

    print("\n--- Betimsel İstatistikler (Birinci Farkı Alınmış Seriler) ---")
    diff_stats = df_stationary.describe().T[["mean", "std", "min", "max"]]
    diff_stats["skewness"] = df_stationary.skew()
    diff_stats.columns = ["Ortalama (Mean)", "Standart Sapma (Std)", "Min", "Max", "Çarpıklık (Skewness)"]
    print(diff_stats.to_string(float_format=lambda x: f"{x:.6f}"))

    train_df, test_df = train_test_split_time_series(df_stationary)

    y_train = train_df[TARGET_TICKER]
    x1_train = train_df[CARBON_TICKER]
    x2_train = train_df[IRON_TICKER]
    fx_train = train_df[FX_TICKER]
    y_test = test_df[TARGET_TICKER]
    x1_test = test_df[CARBON_TICKER]
    x2_test = test_df[IRON_TICKER]
    fx_test = test_df[FX_TICKER]

    print("\n=== Modül 2: ARIMAX Eğitimi ve Benchmark'lar ===")
    # ARIMAX exog: demir cevheri + kur (USD etkisi ARIMAX'a da dahil)
    exog_train = train_df[[IRON_TICKER, FX_TICKER]]
    exog_test = test_df[[IRON_TICKER, FX_TICKER]]
    arimax_model = fit_sarimax(y_train, exog=exog_train)
    arimax_coef_table_str = str(arimax_model.summary().tables[1])
    arimax_test_pred = pd.Series(
        arimax_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=exog_test).values,
        index=y_test.index,
        name="ARIMAX",
    )

    residuals_train = pd.Series(arimax_model.resid, index=y_train.index)

    baseline_model = LinearRegression()
    X_train_bench = train_df[[CARBON_TICKER, IRON_TICKER, FX_TICKER]]
    X_test_bench = test_df[[CARBON_TICKER, IRON_TICKER, FX_TICKER]]
    baseline_model.fit(X_train_bench, y_train)
    baseline_pred = pd.Series(baseline_model.predict(X_test_bench), index=y_test.index, name="Baseline")

    xgb_model = fit_xgboost_regressor(X_train_bench, y_train)
    xgb_pred = pd.Series(xgb_model.predict(X_test_bench), index=y_test.index, name="XGBoost")

    print("\nLSTM benchmark eğitiliyor...")
    lstm_model, lstm_window = train_lstm_model(y_train)
    lstm_preds = forecast_lstm(lstm_model, df_stationary[TARGET_TICKER].values, len(y_train), lstm_window)
    lstm_pred = pd.Series(lstm_preds, index=y_test.index[: len(lstm_preds)], name="LSTM")

    print("\n=== Modül 3: MLP Eğitim ve Hibrit Birleştirme ===")
    # MLP residual features: karbon (x1) + kur (x2) – kur farkı etkisini yakalar
    X_mlp_train, y_mlp_train = build_residual_training_frame(x1_train, residuals_train, x2_train=fx_train)
    mlp_model = build_and_train_mlp(X_mlp_train.values, y_mlp_train.values)

    mlp_residual_test_pred = forecast_mlp_residuals(
        mlp_model=mlp_model,
        x1_train=x1_train,
        residuals_train=residuals_train,
        x1_test=x1_test,
        x2_train=fx_train,
        x2_test=fx_test,
    )
    hybrid_pred = arimax_test_pred.add(mlp_residual_test_pred, fill_value=0.0)
    hybrid_pred.name = "Hibrit ARIMAX-MLP"

    predictions = {
        "Baseline": baseline_pred,
        "ARIMAX": arimax_test_pred,
        "XGBoost": xgb_pred,
        "LSTM": lstm_pred,
        "Hibrit ARIMAX-MLP": hybrid_pred,
    }
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

    plot_module_visualizations(y_test=y_test, predictions_df=predictions_df, metrics_df=metrics_df, residuals_df=residuals_df)

    print("\n=== Modül 5: Karbon Stres Testi ===")
    # Stres testi için ham TL fiyatı kullanıldı (gerçekçi TL bazlı senaryo)
    real_base_price = float(df_raw[TARGET_TICKER].iloc[-1])
    stress_table_df = run_stress_test(base_price=real_base_price)
    plot_stress_test_fan_chart(last_test_date=y_test.index[-1], base_price=real_base_price)
    print(stress_table_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Modül 6: Walk-Forward Validation (Kur Farkı Düzeltmesi Sonrası) ===")
    initial_train_size = int(len(df_stationary) * 0.6)
    wf_metrics = walk_forward_validation(
        df_stationary=df_stationary,
        target_col=TARGET_TICKER,
        carbon_col=CARBON_TICKER,
        iron_col=IRON_TICKER,
        fx_col=FX_TICKER,
        initial_train_size=initial_train_size,
        step=60,
        max_windows=5,
    )
    if not wf_metrics.empty:
        print("\nWalk-Forward Ortalama Metrikler:")
        print(wf_metrics.groupby("Model")[["RMSE", "MAE"]].mean().to_string(float_format=lambda x: f"{x:.6f}"))
    plot_walk_forward_metrics(wf_metrics)

    write_thesis_report(
        basic_stats=basic_stats,
        corr_matrix=corr_matrix,
        diff_stats=diff_stats,
        adf_results=adf_results,
        arimax_summary=str(arimax_model.summary()),
        arimax_coef_table_str=arimax_coef_table_str,
        diagnostics_df=diagnostics_df,
        metrics_df=metrics_df,
        stress_table_df=stress_table_df,
        report_path=Path.cwd() / "Tez_Bulgulari_Raporu.txt",
        walk_forward_metrics_df=wf_metrics,
    )

    return PipelineResult(
        metrics_table=metrics_df,
        predictions=predictions_df,
        residuals=residuals_df,
        diagnostics=diagnostics_df,
        stress_table=stress_table_df,
        walk_forward_metrics=wf_metrics,
    )


if __name__ == "__main__":
    run_pipeline()
