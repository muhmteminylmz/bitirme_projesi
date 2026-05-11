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
from sklearn.linear_model import LinearRegression
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
XGBOOST_PARAMS = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5}
MLP_PARAMS = {"epochs": 200, "batch_size": 16, "validation_split": 0.2, "patience": 10}
MODEL_COLORS = {
    "Baseline": "#6c7a89",
    "ARIMAX": "#4c78a8",
    "XGBoost": "#72b7b2",
    "Hibrit ARIMAX-MLP": "#8b0000",
}
DEFAULT_MODEL_COLOR = "#808080"
STRESS_BANDS = {"S1 (+%30)": 0.30, "S2 (+%60)": 0.60, "S3 (+%100)": 1.00}


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
    tickers = [TARGET_TICKER, CARBON_TICKER, IRON_TICKER]
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


def clean_and_scale_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    cleaned = df.copy().sort_index().ffill().bfill()
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(cleaned), columns=cleaned.columns, index=cleaned.index)
    return scaled, scaler


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


def fit_sarimax(y: pd.Series, exog: pd.Series) -> SARIMAXResultsWrapper:
    print("\nARIMAX modeli eğitiliyor...")
    model = SARIMAX(y, exog=exog, order=(1, 0, 1), enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(disp=False)
    print(fitted_model.summary())
    return fitted_model


def fit_xgboost_regressor(X: pd.DataFrame, y: pd.Series):
    from xgboost import XGBRegressor

    model = XGBRegressor(**XGBOOST_PARAMS, random_state=42)
    model.fit(X, y)
    return model


def build_residual_training_frame(x1_train: pd.Series, residuals_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    # İYİLEŞTİRME: MLP'ye daha fazla geçmiş (lag) verisi vererek performansını artırıyoruz.
    X = pd.DataFrame({
        "x1": x1_train,
        "x1_lag1": x1_train.shift(1),
        "residual_lag1": residuals_train.shift(1),
        "residual_lag2": residuals_train.shift(2)
    })
    X = X.dropna()
    y = residuals_train.loc[X.index]
    return X, y


def build_and_train_mlp(X_train: np.ndarray, y_train: np.ndarray):
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=MLP_PARAMS["patience"], restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=MLP_PARAMS["epochs"],
        batch_size=MLP_PARAMS["batch_size"],
        validation_split=MLP_PARAMS["validation_split"],
        callbacks=[early_stopping],
        verbose=0,
    )
    return model


def forecast_mlp_residuals(mlp_model, x1_test: pd.Series, train_x1_last: float, train_res_last1: float, train_res_last2: float) -> pd.Series:
    preds = []
    curr_res1 = train_res_last1
    curr_res2 = train_res_last2
    curr_x1_lag = train_x1_last

    for i in range(len(x1_test)):
        curr_x1 = x1_test.iloc[i]
        X_input = np.array([[curr_x1, curr_x1_lag, curr_res1, curr_res2]], dtype=float)
        pred_res = float(mlp_model.predict(X_input, verbose=0).ravel()[0])
        preds.append(pred_res)
        
        # Gelecek adım için lag'leri kaydırıyoruz
        curr_res2 = curr_res1
        curr_res1 = pred_res
        curr_x1_lag = curr_x1

    return pd.Series(preds, index=x1_test.index)


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

    plt.figure(figsize=(8, 6))
    bar_colors = [MODEL_COLORS.get(m, DEFAULT_MODEL_COLOR) for m in metrics_df["Model"]]
    ax = sns.barplot(data=metrics_df, x="Model", y="MAE", hue="Model", palette=bar_colors, legend=False)
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
    plt.title("Şekil 4.5: Modellerin Test Kümesi MAE Karşılaştırması", pad=15, fontsize=12, fontweight="bold")
    plt.ylabel("MAE Değeri")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("Grafik_5_MAE_Bar.png")
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
            "TABLO 4.2: ARIMAX KATSAYI TABLOSU (AR, MA ve X2 / Demir Cevheri)",
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

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTez raporu oluşturuldu: {report_path}")


def run_pipeline(interval: str = "1d") -> PipelineResult:
    print("=== Modül 1: Veri Çekme ve Ön İşleme ===")
    df_raw = fetch_yfinance_data(interval=interval)

    print("\n--- Temel İstatistikler (Ham Veri) ---")
    basic_stats = df_raw.describe()
    print(basic_stats.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- Korelasyon Matrisi (Ham Veri) ---")
    corr_matrix = df_raw.corr()
    print(corr_matrix.to_string(float_format=lambda x: f"{x:.6f}"))

    plot_correlation_heatmap(corr_matrix)

    df_scaled, _ = clean_and_scale_data(df_raw)
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
    y_test = test_df[TARGET_TICKER]
    x1_test = test_df[CARBON_TICKER]
    x2_test = test_df[IRON_TICKER]

    print("\n=== Modül 2: ARIMAX Eğitimi ve Benchmark'lar ===")
    arimax_model = fit_sarimax(y_train, exog=x2_train)
    arimax_coef_table_str = str(arimax_model.summary().tables[1])
    arimax_test_pred = pd.Series(
        arimax_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=x2_test).values,
        index=y_test.index,
        name="ARIMAX",
    )

    residuals_train = pd.Series(arimax_model.resid, index=y_train.index)

    baseline_model = LinearRegression()
    X_train_bench = train_df[[CARBON_TICKER, IRON_TICKER]]
    X_test_bench = test_df[[CARBON_TICKER, IRON_TICKER]]
    baseline_model.fit(X_train_bench, y_train)
    baseline_pred = pd.Series(baseline_model.predict(X_test_bench), index=y_test.index, name="Baseline")

    xgb_model = fit_xgboost_regressor(X_train_bench, y_train)
    xgb_pred = pd.Series(xgb_model.predict(X_test_bench), index=y_test.index, name="XGBoost")

    print("\n=== Modül 3: MLP Eğitim ve Hibrit Birleştirme ===")
    X_mlp_train, y_mlp_train = build_residual_training_frame(x1_train, residuals_train)
    mlp_model = build_and_train_mlp(X_mlp_train.values, y_mlp_train.values)

    # İYİLEŞTİRME: MLP test verisi için geçmiş 2 günün hata payı da sağlanıyor.
    mlp_residual_test_pred = forecast_mlp_residuals(
        mlp_model=mlp_model, 
        x1_test=x1_test, 
        train_x1_last=float(x1_train.iloc[-1]),
        train_res_last1=float(residuals_train.iloc[-1]),
        train_res_last2=float(residuals_train.iloc[-2])
    )
    hybrid_pred = arimax_test_pred.add(mlp_residual_test_pred, fill_value=0.0)
    hybrid_pred.name = "Hibrit ARIMAX-MLP"

    predictions = {"Baseline": baseline_pred, "ARIMAX": arimax_test_pred, "XGBoost": xgb_pred, "Hibrit ARIMAX-MLP": hybrid_pred}
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
    # ÇÖZÜM: 0.000 Hatasını önlemek ve gerçekçi sonuç vermek için ham TL fiyatı kullanıldı.
    real_base_price = float(df_raw[TARGET_TICKER].iloc[-1])
    stress_table_df = run_stress_test(base_price=real_base_price)
    plot_stress_test_fan_chart(last_test_date=y_test.index[-1], base_price=real_base_price)
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
