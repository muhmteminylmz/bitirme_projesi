"""ARIMAX + MLP hybrid pipeline for corporate carbon stress testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yfinance as yf
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
sns.set_style("whitegrid")

TARGET_TICKER = "EREGL.IS"
CARBON_TICKER = "KEUA"
IRON_TICKER = "TIO=F"
FX_TICKER = "USDTRY=X"
TARGET_USD_TICKER = f"{TARGET_TICKER}_USD"
XGBOOST_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3}
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
LSTM_WINDOW_SIZE = 20
LSTM_EPOCHS = 150
STABLE_RMSE_BAND_UPPER = 0.60
MIN_HYBRID_IMPROVEMENT = 0.01
MIN_ROLLING_WIN_RATIO = 0.50


@dataclass
class RecoveryConfig:
    use_usd_target: bool = False
    include_fx_feature: bool = False
    use_learned_hybrid_combiner: bool = True


@dataclass
class PipelineResult:
    metrics_table: pd.DataFrame
    predictions: pd.DataFrame
    residuals: pd.DataFrame
    diagnostics: pd.DataFrame
    stress_table: pd.DataFrame


@dataclass
class LeakageSafeSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    stationarity: Dict[str, Dict[str, float]]


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


def clean_and_scale_data(df: pd.DataFrame, fit_df: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    cleaned = df.copy().sort_index().ffill().bfill()
    fit_cleaned = fit_df.copy().sort_index().ffill().bfill() if fit_df is not None else cleaned
    scaler = MinMaxScaler()
    scaler.fit(fit_cleaned)
    scaled = pd.DataFrame(scaler.transform(cleaned), columns=cleaned.columns, index=cleaned.index)
    return scaled, scaler


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(df / df.shift(1))
    return returns.dropna(how="any")


def convert_target_to_usd(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET_USD_TICKER] = out[TARGET_TICKER] / out[FX_TICKER]
    return out


def apply_stationarity_policy_train_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    train_stationary = train_df.copy()
    test_stationary = test_df.copy()
    adf_results: Dict[str, Dict[str, float]] = {}

    def safe_adf_pvalue(series: pd.Series) -> float:
        s = series.dropna()
        if s.empty or s.nunique() <= 1:
            return 1.0
        return float(adfuller(s)[1])

    for col in train_df.columns:
        initial_p = safe_adf_pvalue(train_df[col])
        if initial_p > 0.05:
            train_stationary[col] = train_df[col].diff()
            bridge = pd.concat([train_df[col].iloc[[-1]], test_df[col]])
            test_stationary[col] = bridge.diff().iloc[1:]
            final_p = safe_adf_pvalue(train_stationary[col])
            adf_results[col] = {"initial_p": initial_p, "differenced": True, "final_p": final_p}
        else:
            adf_results[col] = {"initial_p": initial_p, "differenced": False, "final_p": initial_p}

    train_stationary = train_stationary.dropna(how="any")
    test_stationary = test_stationary.dropna(how="any")
    aligned_test_index = test_df.index.intersection(test_stationary.index)
    test_stationary = test_stationary.loc[aligned_test_index]
    return train_stationary, test_stationary, adf_results


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


def train_test_split_time_series(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_size = int(n * train_ratio)
    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()


def train_val_test_split_time_series(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio toplamı 1.0 olmalıdır.")
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def apply_stationarity_policy(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    train_stationary = train_df.copy()
    val_stationary = val_df.copy()
    test_stationary = test_df.copy()
    adf_results: Dict[str, Dict[str, float]] = {}

    def safe_adf_pvalue(series: pd.Series) -> float:
        s = series.dropna()
        if s.empty or s.nunique() <= 1:
            return 1.0
        return float(adfuller(s)[1])

    for col in train_df.columns:
        initial_p = safe_adf_pvalue(train_df[col])
        if initial_p > 0.05:
            train_stationary[col] = train_df[col].diff()
            val_bridge = pd.concat([train_df[col].iloc[[-1]], val_df[col]])
            val_stationary[col] = val_bridge.diff().iloc[1:]
            test_bridge = pd.concat([val_df[col].iloc[[-1]], test_df[col]])
            test_stationary[col] = test_bridge.diff().iloc[1:]
            final_p = safe_adf_pvalue(train_stationary[col])
            adf_results[col] = {"initial_p": initial_p, "differenced": True, "final_p": final_p}
        else:
            adf_results[col] = {"initial_p": initial_p, "differenced": False, "final_p": initial_p}

    train_stationary = train_stationary.dropna(how="any")
    val_stationary = val_stationary.dropna(how="any")
    test_stationary = test_stationary.dropna(how="any")
    val_stationary = val_stationary.loc[val_df.index.intersection(val_stationary.index)]
    test_stationary = test_stationary.loc[test_df.index.intersection(test_stationary.index)]
    return train_stationary, val_stationary, test_stationary, adf_results


def validate_split_integrity(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/val/test split boş olamaz.")
    if not train_df.index.is_monotonic_increasing or not val_df.index.is_monotonic_increasing or not test_df.index.is_monotonic_increasing:
        raise ValueError("Tarih indeksleri artan sırada olmalıdır.")
    if train_df.index.max() >= val_df.index.min() or val_df.index.max() >= test_df.index.min():
        raise ValueError("Train/val/test aralıkları çakışıyor.")


def prepare_leakage_safe_splits(
    df_prices: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15
) -> LeakageSafeSplits:
    returns = compute_log_returns(df_prices)
    train_raw, val_raw, test_raw = train_val_test_split_time_series(
        returns, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )
    train_stationary, val_stationary, test_stationary, stationarity = apply_stationarity_policy(train_raw, val_raw, test_raw)
    validate_split_integrity(train_stationary, val_stationary, test_stationary)
    combined = pd.concat([train_stationary, val_stationary, test_stationary]).sort_index()
    combined_scaled, _ = clean_and_scale_data(combined, fit_df=train_stationary)
    train = combined_scaled.loc[train_stationary.index]
    val = combined_scaled.loc[val_stationary.index]
    test = combined_scaled.loc[test_stationary.index]
    return LeakageSafeSplits(train=train, val=val, test=test, stationarity=stationarity)


def build_expanding_windows(
    df: pd.DataFrame, min_train_size: int, val_size: int, test_size: int, step_size: int, max_windows: int = 5
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    windows: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    n = len(df)
    start_train_size = min_train_size
    while start_train_size + val_size + test_size <= n and len(windows) < max_windows:
        train = df.iloc[:start_train_size].copy()
        val = df.iloc[start_train_size : start_train_size + val_size].copy()
        test = df.iloc[start_train_size + val_size : start_train_size + val_size + test_size].copy()
        validate_split_integrity(train, val, test)
        windows.append((train, val, test))
        start_train_size += step_size
    return windows


def validate_data_quality(df_prices: pd.DataFrame) -> Dict[str, float | bool]:
    missing_ratio = float(df_prices.isna().mean().mean())
    monotonic_index = bool(df_prices.index.is_monotonic_increasing)
    returns = compute_log_returns(df_prices.ffill().bfill())
    jump_ratio = float((returns.abs() > returns.abs().quantile(0.99)).mean().mean()) if not returns.empty else 0.0
    quality_pass = bool(missing_ratio < 0.01 and monotonic_index and jump_ratio < 0.05)
    return {
        "missing_ratio": missing_ratio,
        "monotonic_index": monotonic_index,
        "jump_ratio": jump_ratio,
        "quality_pass": quality_pass,
    }


def train_hybrid_combiner(y_true: pd.Series, arimax_pred: pd.Series, mlp_residual_pred: pd.Series) -> Ridge:
    aligned = pd.concat(
        [y_true.rename("y"), arimax_pred.rename("arimax"), mlp_residual_pred.rename("mlp_residual")], axis=1
    ).dropna()
    X = aligned[["arimax", "mlp_residual"]]
    y = aligned["y"]
    model = Ridge(alpha=0.1)
    model.fit(X, y)
    return model


def apply_hybrid_combiner(model: Ridge, arimax_pred: pd.Series, mlp_residual_pred: pd.Series) -> pd.Series:
    features = pd.concat([arimax_pred.rename("arimax"), mlp_residual_pred.rename("mlp_residual")], axis=1).dropna()
    preds = model.predict(features)
    return pd.Series(preds, index=features.index, name="Hibrit ARIMAX-MLP")


def evaluate_success_criteria(
    metrics_df: pd.DataFrame, rolling_df: pd.DataFrame, stable_rmse_upper: float = STABLE_RMSE_BAND_UPPER
) -> Dict[str, float | bool]:
    hybrid_row = metrics_df[metrics_df["Model"] == "Hibrit ARIMAX-MLP"].iloc[0]
    others = metrics_df[metrics_df["Model"] != "Hibrit ARIMAX-MLP"]
    second_best_rmse = float(others["RMSE"].min()) if not others.empty else float(hybrid_row["RMSE"])
    hybrid_rmse = float(hybrid_row["RMSE"])
    single_split_improvement = float(second_best_rmse - hybrid_rmse)

    rolling_wins = 0
    total_windows = 0
    if not rolling_df.empty:
        for _, g in rolling_df.groupby("window"):
            if "Hibrit ARIMAX-MLP" not in set(g["Model"]):
                continue
            total_windows += 1
            h_rmse = float(g[g["Model"] == "Hibrit ARIMAX-MLP"]["RMSE"].iloc[0])
            best_rmse = float(g["RMSE"].min())
            if np.isclose(h_rmse, best_rmse) or h_rmse <= best_rmse:
                rolling_wins += 1
    rolling_win_ratio = float(rolling_wins / total_windows) if total_windows else 0.0

    single_split_ok = single_split_improvement >= MIN_HYBRID_IMPROVEMENT
    rolling_ok = rolling_win_ratio >= MIN_ROLLING_WIN_RATIO
    stable_ok = hybrid_rmse <= stable_rmse_upper
    overall_pass = bool(single_split_ok and rolling_ok and stable_ok)

    if single_split_ok and not rolling_ok:
        overall_pass = False

    return {
        "single_split_improvement": single_split_improvement,
        "rolling_win_ratio": rolling_win_ratio,
        "single_split_ok": single_split_ok,
        "rolling_ok": rolling_ok,
        "stable_ok": stable_ok,
        "pass": overall_pass,
    }


def run_ablation_experiments(y_true: pd.Series, predictions: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for variant, pred in predictions.items():
        aligned = pd.concat([y_true.rename("y"), pred.rename("p")], axis=1).dropna()
        if aligned.empty:
            continue
        rmse = float(np.sqrt(mean_squared_error(aligned["y"], aligned["p"])))
        mae = float(mean_absolute_error(aligned["y"], aligned["p"]))
        rows.append({"Variant": variant, "RMSE": rmse, "MAE": mae})
    return pd.DataFrame(rows).sort_values(by="RMSE").reset_index(drop=True)


def build_module_breakdown(
    y_true: pd.Series, predictions: Dict[str, pd.Series], rolling_summary: pd.DataFrame
) -> pd.DataFrame:
    breakdown_rows = []
    for module_name, pred in predictions.items():
        aligned = pd.concat([y_true.rename("y"), pred.rename("p")], axis=1).dropna()
        if aligned.empty:
            continue
        rmse = float(np.sqrt(mean_squared_error(aligned["y"], aligned["p"])))
        breakdown_rows.append({"module": module_name, "rmse": rmse, "source": "single_split"})
    if not rolling_summary.empty and {"Model", "RMSE_mean"}.issubset(set(rolling_summary.columns)):
        for _, row in rolling_summary.iterrows():
            breakdown_rows.append({"module": f"{row['Model']} (rolling)", "rmse": float(row["RMSE_mean"]), "source": "rolling"})
    return pd.DataFrame(breakdown_rows).sort_values(by="rmse").reset_index(drop=True)


def find_first_breakpoint(breakdown: pd.DataFrame, threshold: float = 0.20) -> str:
    over = breakdown[breakdown["rmse"] > threshold]
    if over.empty:
        return f"İlk kırılma noktası bulunamadı (threshold={threshold:.3f})."
    first = over.iloc[0]
    return f"İlk kırılma noktası: {first['module']} (rmse={first['rmse']:.6f})"

def build_xgboost_features(
    target: pd.Series, features_df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5]
) -> pd.DataFrame:
    """Build enriched feature matrix for XGBoost with target/feature lags and rolling statistics."""
    X = features_df.copy()
    for lag in lags:
        X[f"target_lag{lag}"] = target.shift(lag)
    for col in features_df.columns:
        X[f"{col}_lag1"] = features_df[col].shift(1)
        X[f"{col}_lag2"] = features_df[col].shift(2)
        X[f"{col}_change1"] = features_df[col].diff()
    X["target_roll5_mean"] = target.rolling(5).mean()
    X["target_roll5_std"] = target.rolling(5).std()
    X["target_roll10_mean"] = target.rolling(10).mean()
    X["target_momentum5"] = target - target.shift(5)
    return X.dropna()


def fit_sarimax(y: pd.Series, exog: pd.DataFrame) -> SARIMAXResultsWrapper:
    print("\nOptimal ARIMA parametreleri aranıyor...")
    auto_model = auto_arima(
        y,
        exogenous=exog,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        trace=True,
        error_action="ignore",
        max_p=7,
        max_q=7,
        d=0,
        information_criterion="aic",
    )
    order = auto_model.order
    print(f"Seçilen ARIMA order: {order}")

    model = SARIMAX(y, exog=exog, order=order, enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(disp=False)
    print(fitted_model.summary())
    return fitted_model


def forecast_arimax_val_test(
    arimax_model: SARIMAXResultsWrapper,
    exog_val: pd.DataFrame,
    exog_test: pd.DataFrame,
    val_index: pd.Index,
    test_index: pd.Index,
) -> Tuple[pd.Series, pd.Series]:
    exog_horizon = pd.concat([exog_val, exog_test])
    horizon_pred = arimax_model.get_forecast(steps=len(exog_horizon), exog=exog_horizon).predicted_mean
    arimax_val_pred = pd.Series(horizon_pred.iloc[: len(exog_val)].values, index=val_index, name="ARIMAX_VAL")
    arimax_test_pred = pd.Series(horizon_pred.iloc[len(exog_val) :].values, index=test_index, name="ARIMAX")
    return arimax_val_pred, arimax_test_pred


def fit_xgboost_regressor(X: pd.DataFrame, y: pd.Series):
    from xgboost import XGBRegressor

    model = XGBRegressor(**XGBOOST_PARAMS, random_state=42)
    model.fit(X, y)
    return model


def build_residual_training_frame(
    x1_train: pd.Series,
    residuals_train: pd.Series,
    x2_train: pd.Series | None = None,
    feature_mode: str = "extended",
) -> Tuple[pd.DataFrame, pd.Series]:
    if feature_mode not in {"extended", "legacy"}:
        raise ValueError("feature_mode 'extended' veya 'legacy' olmalıdır.")

    if feature_mode == "legacy":
        X = pd.DataFrame(
            {
                "x1": x1_train,
                "x1_lag1": x1_train.shift(1),
                "x1_lag2": x1_train.shift(2),
                "x1_lag3": x1_train.shift(3),
                "residual_lag1": residuals_train.shift(1),
                "residual_lag2": residuals_train.shift(2),
                "residual_lag3": residuals_train.shift(3),
                "rolling_mean_5": x1_train.rolling(5).mean(),
                "rolling_std_5": x1_train.rolling(5).std(),
                "rolling_mean_10": x1_train.rolling(10).mean(),
                "rolling_std_10": x1_train.rolling(10).std(),
            }
        )
    else:
        x2 = x2_train.reindex(x1_train.index) if x2_train is not None else x1_train.rename("x2_proxy")
        X = pd.DataFrame(
            {
                "x1": x1_train,
                "x2": x2,
                "x1_x2_interaction": x1_train * x2,
                "x1_lag1": x1_train.shift(1),
                "x2_lag1": x2.shift(1),
                "x1_change_lag1": x1_train.diff().shift(1),
                "x2_change_lag1": x2.diff().shift(1),
                "residual_lag1": residuals_train.shift(1),
                "residual_lag5": residuals_train.shift(5),
                "residual_lag6": residuals_train.shift(6),
                "residual_roll_mean_3": residuals_train.rolling(3).mean(),
                "residual_roll_std_3": residuals_train.rolling(3).std(),
            }
        )
    X = X.dropna()
    y = residuals_train.reindex(X.index)
    return X, y


def build_and_train_mlp(X_train: np.ndarray, y_train: np.ndarray):
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=500,
        batch_size=8,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )
    return model


def forecast_mlp_residuals(mlp_model, x1_train: pd.Series, residuals_train: pd.Series, x1_test: pd.Series) -> pd.Series:
    if len(x1_train) < 10:
        raise ValueError("MLP residual tahmini için x1_train en az 10 gözlem içermelidir.")
    if len(residuals_train) < 3:
        raise ValueError("MLP residual tahmini için residuals_train en az 3 gözlem içermelidir.")

    preds = []
    x1_history = x1_train.astype(float).tolist()
    residual_history = residuals_train.astype(float).tolist()

    for i in range(len(x1_test)):
        curr_x1 = x1_test.iloc[i]

        x1_context = x1_history + [float(curr_x1)]
        X_input = np.array(
            [
                [
                    float(curr_x1),
                    x1_history[-1],
                    x1_history[-2],
                    x1_history[-3],
                    residual_history[-1],
                    residual_history[-2],
                    residual_history[-3],
                    float(np.mean(x1_context[-5:])),
                    float(np.std(x1_context[-5:], ddof=1)),
                    float(np.mean(x1_context[-10:])),
                    float(np.std(x1_context[-10:], ddof=1)),
                ]
            ],
            dtype=float,
        )
        pred_res = float(mlp_model.predict(X_input, verbose=0).ravel()[0])
        preds.append(pred_res)

        x1_history.append(float(curr_x1))
        residual_history.append(pred_res)

    return pd.Series(preds, index=x1_test.index)


def forecast_mlp_residuals_extended(
    mlp_model,
    x1_train: pd.Series,
    x2_train: pd.Series,
    residuals_train: pd.Series,
    x1_test: pd.Series,
    x2_test: pd.Series,
) -> pd.Series:
    """Forecast MLP residuals using extended features (Carbon x1 + Iron x2 + interactions)."""
    if len(x1_train) < 10:
        raise ValueError("MLP residual tahmini için x1_train en az 10 gözlem içermelidir.")
    if len(residuals_train) < 6:
        raise ValueError("MLP residual tahmini (extended) için residuals_train en az 6 gözlem içermelidir.")

    preds = []
    x1_history = x1_train.astype(float).tolist()
    x2_history = x2_train.reindex(x1_train.index).astype(float).tolist()
    residual_history = residuals_train.astype(float).tolist()

    for i in range(len(x1_test)):
        curr_x1 = float(x1_test.iloc[i])
        curr_x2 = float(x2_test.iloc[i])
        x1_lag1 = x1_history[-1]
        x2_lag1 = x2_history[-1]
        x1_change_lag1 = x1_history[-1] - x1_history[-2] if len(x1_history) >= 2 else 0.0
        x2_change_lag1 = x2_history[-1] - x2_history[-2] if len(x2_history) >= 2 else 0.0
        res_lag1 = residual_history[-1]
        res_lag5 = residual_history[-5] if len(residual_history) >= 5 else residual_history[0]
        res_lag6 = residual_history[-6] if len(residual_history) >= 6 else residual_history[0]
        res_roll3 = residual_history[-3:]
        res_roll_mean3 = float(np.mean(res_roll3))
        res_roll_std3 = float(np.std(res_roll3, ddof=1)) if len(res_roll3) >= 2 else 0.0

        X_input = np.array(
            [[
                curr_x1, curr_x2, curr_x1 * curr_x2,
                x1_lag1, x2_lag1,
                x1_change_lag1, x2_change_lag1,
                res_lag1, res_lag5, res_lag6,
                res_roll_mean3, res_roll_std3,
            ]],
            dtype=float,
        )
        pred_res = float(mlp_model.predict(X_input, verbose=0).ravel()[0])
        preds.append(pred_res)

        x1_history.append(curr_x1)
        x2_history.append(curr_x2)
        residual_history.append(pred_res)

    return pd.Series(preds, index=x1_test.index)


def create_lstm_dataset(series, window_size: int = 5):
    X = []
    y = []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_lstm_model(train_series: pd.Series):
    window_size = LSTM_WINDOW_SIZE
    X_train, y_train = create_lstm_dataset(train_series.values, window_size)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,
    )
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


def run_carbon_stress_test(
    arimax_model: SARIMAXResultsWrapper,
    exog_test: pd.DataFrame,
    baseline_forecast: pd.Series,
    train_size: int,
) -> pd.DataFrame:
    rows = []
    for scenario, shock in STRESS_BANDS.items():
        shocked_exog = exog_test.copy()
        shocked_exog[CARBON_TICKER] = shocked_exog[CARBON_TICKER] * (1 + shock)
        stressed_forecast = pd.Series(
            arimax_model.predict(
                start=train_size,
                end=train_size + len(exog_test) - 1,
                exog=shocked_exog,
            ).values,
            index=baseline_forecast.index,
        )
        rows.append(
            {
                "Senaryo": scenario,
                "Karbon Şoku": f"+%{int(shock * 100)}",
                "Baz Tahmin Ortalaması": float(baseline_forecast.mean()),
                "Stres Tahmin Ortalaması": float(stressed_forecast.mean()),
                "Ortalama Etki (%)": float((stressed_forecast.mean() / baseline_forecast.mean() - 1.0) * 100.0),
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
            "TABLO 4.2: ARIMAX KATSAYI TABLOSU (AR, MA ve Exogenous: Karbon + Demir + Kur)",
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
            "6) STRES TESTİ SONUÇ TABLOSU (Karbon Şoku Exogenous Üzerinden ARIMAX Re-Forecast)",
            "-" * 80,
            stress_table_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTez raporu oluşturuldu: {report_path}")


def run_pipeline(interval: str = "1d", config: RecoveryConfig | None = None) -> PipelineResult:
    cfg = config or RecoveryConfig()
    print("=== Modül 1: Veri Çekme ve Ön İşleme ===")
    df_raw = fetch_yfinance_data(interval=interval)
    df_raw_usd = convert_target_to_usd(df_raw) if cfg.use_usd_target else df_raw.copy()
    target_col = TARGET_USD_TICKER if cfg.use_usd_target else TARGET_TICKER
    feature_cols = [CARBON_TICKER, IRON_TICKER]
    if cfg.include_fx_feature:
        feature_cols.append(FX_TICKER)
    model_input_raw = df_raw_usd[[target_col, *feature_cols]].copy()

    print("\n--- Temel İstatistikler (Ham Veri) ---")
    basic_stats = model_input_raw.describe()
    print(basic_stats.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- Korelasyon Matrisi (Ham Veri) ---")
    corr_matrix = model_input_raw.corr()
    print(corr_matrix.to_string(float_format=lambda x: f"{x:.6f}"))

    plot_correlation_heatmap(corr_matrix)

    df_returns = compute_log_returns(model_input_raw)
    train_returns, val_returns, test_returns = train_val_test_split_time_series(df_returns)
    train_stationary, val_stationary, test_stationary, adf_results = apply_stationarity_policy(
        train_returns, val_returns, test_returns
    )

    combined_stationary = pd.concat([train_stationary, val_stationary, test_stationary]).sort_index()
    combined_scaled, _ = clean_and_scale_data(combined_stationary, fit_df=train_stationary)
    train_df = combined_scaled.loc[train_stationary.index]
    val_df = combined_scaled.loc[val_stationary.index]
    test_df = combined_scaled.loc[test_stationary.index]

    print("\n--- Betimsel İstatistikler (Birinci Farkı Alınmış Seriler) ---")
    diff_stats = combined_stationary.describe().T[["mean", "std", "min", "max"]]
    diff_stats["skewness"] = combined_stationary.skew()
    diff_stats.columns = ["Ortalama (Mean)", "Standart Sapma (Std)", "Min", "Max", "Çarpıklık (Skewness)"]
    print(diff_stats.to_string(float_format=lambda x: f"{x:.6f}"))

    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]
    x1_train = train_df[CARBON_TICKER]
    x1_val = val_df[CARBON_TICKER]
    x1_test = test_df[CARBON_TICKER]
    x2_train = train_df[IRON_TICKER]
    x2_val = val_df[IRON_TICKER]
    x2_test = test_df[IRON_TICKER]
    if cfg.include_fx_feature:
        fx_train = train_df[FX_TICKER]
        fx_val = val_df[FX_TICKER]
        fx_test = test_df[FX_TICKER]
        arimax_exog_train = pd.DataFrame({CARBON_TICKER: x1_train, IRON_TICKER: x2_train, FX_TICKER: fx_train})
        arimax_exog_val = pd.DataFrame({CARBON_TICKER: x1_val, IRON_TICKER: x2_val, FX_TICKER: fx_val})
        arimax_exog_test = pd.DataFrame({CARBON_TICKER: x1_test, IRON_TICKER: x2_test, FX_TICKER: fx_test})
        bench_cols = [CARBON_TICKER, IRON_TICKER, FX_TICKER]
    else:
        arimax_exog_train = pd.DataFrame({CARBON_TICKER: x1_train, IRON_TICKER: x2_train})
        arimax_exog_val = pd.DataFrame({CARBON_TICKER: x1_val, IRON_TICKER: x2_val})
        arimax_exog_test = pd.DataFrame({CARBON_TICKER: x1_test, IRON_TICKER: x2_test})
        bench_cols = [CARBON_TICKER, IRON_TICKER]

    print("\n=== Modül 2: ARIMAX Eğitimi ve Benchmark'lar ===")
    arimax_model = fit_sarimax(y_train, exog=arimax_exog_train)
    arimax_coef_table_str = str(arimax_model.summary().tables[1])
    arimax_val_pred, arimax_test_pred = forecast_arimax_val_test(
        arimax_model=arimax_model,
        exog_val=arimax_exog_val,
        exog_test=arimax_exog_test,
        val_index=y_val.index,
        test_index=y_test.index,
    )

    residuals_train = pd.Series(arimax_model.resid, index=y_train.index)

    baseline_model = LinearRegression()
    X_train_bench = train_df[bench_cols]
    X_test_bench = test_df[bench_cols]
    baseline_model.fit(X_train_bench, y_train)
    baseline_pred = pd.Series(baseline_model.predict(X_test_bench), index=y_test.index, name="Baseline")

    full_target_xgb = pd.concat([y_train, y_val, y_test])
    full_features_xgb = pd.concat([train_df[bench_cols], val_df[bench_cols], test_df[bench_cols]])
    full_xgb_X = build_xgboost_features(full_target_xgb, full_features_xgb)
    xgb_X_train = full_xgb_X.loc[full_xgb_X.index.isin(y_train.index)]
    xgb_X_test = full_xgb_X.loc[full_xgb_X.index.isin(y_test.index)]
    xgb_model = fit_xgboost_regressor(xgb_X_train, y_train.loc[xgb_X_train.index])
    xgb_pred = pd.Series(xgb_model.predict(xgb_X_test), index=xgb_X_test.index, name="XGBoost")

    print("\nLSTM benchmark eğitiliyor...")

    lstm_train_series = train_df[target_col]

    lstm_model, lstm_window = train_lstm_model(lstm_train_series)

    lstm_input_series = pd.concat(
    [train_df[target_col], val_df[target_col], test_df[target_col]]
    ).values

    lstm_preds = forecast_lstm(
    lstm_model,
    lstm_input_series,
    len(y_train) + len(y_val),
    lstm_window,
    )

    lstm_pred = pd.Series(
    lstm_preds,
    index=y_test.index[: len(lstm_preds)],
    name="LSTM",
    )

    print("\n=== Modül 3: MLP Eğitim ve Hibrit Birleştirme ===")
    X_mlp_train, y_mlp_train = build_residual_training_frame(x1_train, residuals_train, x2_train=x2_train, feature_mode="extended")
    mlp_model = build_and_train_mlp(X_mlp_train.values, y_mlp_train.values)

    mlp_residual_val_pred = forecast_mlp_residuals_extended(
        mlp_model=mlp_model,
        x1_train=x1_train,
        x2_train=x2_train,
        residuals_train=residuals_train,
        x1_test=x1_val,
        x2_test=x2_val,
    )
    mlp_residual_test_pred = forecast_mlp_residuals_extended(
        mlp_model=mlp_model,
        x1_train=pd.concat([x1_train, x1_val]),
        x2_train=pd.concat([x2_train, x2_val]),
        residuals_train=pd.concat([residuals_train, mlp_residual_val_pred]),
        x1_test=x1_test,
        x2_test=x2_test,
    )
    if cfg.use_learned_hybrid_combiner:
        combiner = train_hybrid_combiner(y_val, arimax_val_pred, mlp_residual_val_pred)
        hybrid_pred = apply_hybrid_combiner(combiner, arimax_test_pred, mlp_residual_test_pred)
    else:
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
    stress_table_df = run_carbon_stress_test(
        arimax_model=arimax_model,
        exog_test=arimax_exog_test,
        baseline_forecast=arimax_test_pred,
        train_size=len(y_train),
    )
    print(stress_table_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

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
    )

    return PipelineResult(
        metrics_table=metrics_df,
        predictions=predictions_df,
        residuals=residuals_df,
        diagnostics=diagnostics_df,
        stress_table=stress_table_df,
    )


if __name__ == "__main__":
    run_pipeline()
