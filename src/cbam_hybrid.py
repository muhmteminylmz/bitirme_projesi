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
LSTM_WINDOW_SIZE = 20
LSTM_EPOCHS = 80


@dataclass
class PipelineResult:
    metrics_table: pd.DataFrame
    predictions: pd.DataFrame
    residuals: pd.DataFrame
    diagnostics: pd.DataFrame
    stress_table: pd.DataFrame


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


def build_residual_training_frame(x1_train: pd.Series, residuals_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
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
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=16, verbose=0)
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


def run_pipeline(interval: str = "1d") -> PipelineResult:
    print("=== Modül 1: Veri Çekme ve Ön İşleme ===")
    df_raw = fetch_yfinance_data(interval=interval)
    df_raw_usd = convert_target_to_usd(df_raw)
    model_input_raw = df_raw_usd[[TARGET_USD_TICKER, CARBON_TICKER, IRON_TICKER, FX_TICKER]].copy()

    print("\n--- Temel İstatistikler (Ham Veri) ---")
    basic_stats = model_input_raw.describe()
    print(basic_stats.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- Korelasyon Matrisi (Ham Veri) ---")
    corr_matrix = model_input_raw.corr()
    print(corr_matrix.to_string(float_format=lambda x: f"{x:.6f}"))

    plot_correlation_heatmap(corr_matrix)

    df_returns = compute_log_returns(model_input_raw)
    train_returns, test_returns = train_test_split_time_series(df_returns)
    train_stationary, test_stationary, adf_results = apply_stationarity_policy_train_test(train_returns, test_returns)

    combined_stationary = pd.concat([train_stationary, test_stationary]).sort_index()
    combined_scaled, _ = clean_and_scale_data(combined_stationary, fit_df=train_stationary)
    train_df = combined_scaled.loc[train_stationary.index]
    test_df = combined_scaled.loc[test_stationary.index]

    print("\n--- Betimsel İstatistikler (Birinci Farkı Alınmış Seriler) ---")
    diff_stats = combined_stationary.describe().T[["mean", "std", "min", "max"]]
    diff_stats["skewness"] = combined_stationary.skew()
    diff_stats.columns = ["Ortalama (Mean)", "Standart Sapma (Std)", "Min", "Max", "Çarpıklık (Skewness)"]
    print(diff_stats.to_string(float_format=lambda x: f"{x:.6f}"))

    y_train = train_df[TARGET_USD_TICKER]
    x1_train = train_df[CARBON_TICKER]
    x2_train = train_df[IRON_TICKER]
    fx_train = train_df[FX_TICKER]
    y_test = test_df[TARGET_USD_TICKER]
    x1_test = test_df[CARBON_TICKER]
    x2_test = test_df[IRON_TICKER]
    fx_test = test_df[FX_TICKER]
    arimax_exog_train = pd.DataFrame({CARBON_TICKER: x1_train, IRON_TICKER: x2_train, FX_TICKER: fx_train})
    arimax_exog_test = pd.DataFrame({CARBON_TICKER: x1_test, IRON_TICKER: x2_test, FX_TICKER: fx_test})

    print("\n=== Modül 2: ARIMAX Eğitimi ve Benchmark'lar ===")
    arimax_model = fit_sarimax(y_train, exog=arimax_exog_train)
    arimax_coef_table_str = str(arimax_model.summary().tables[1])
    arimax_test_pred = pd.Series(
        arimax_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=arimax_exog_test).values,
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
    lstm_input_series = pd.concat([train_df[TARGET_USD_TICKER], test_df[TARGET_USD_TICKER]]).values
    lstm_preds = forecast_lstm(lstm_model, lstm_input_series, len(y_train), lstm_window)
    lstm_pred = pd.Series(lstm_preds, index=y_test.index[: len(lstm_preds)], name="LSTM")

    print("\n=== Modül 3: MLP Eğitim ve Hibrit Birleştirme ===")
    X_mlp_train, y_mlp_train = build_residual_training_frame(x1_train, residuals_train)
    mlp_model = build_and_train_mlp(X_mlp_train.values, y_mlp_train.values)

    mlp_residual_test_pred = forecast_mlp_residuals(
        mlp_model=mlp_model,
        x1_train=x1_train,
        residuals_train=residuals_train,
        x1_test=x1_test,
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
