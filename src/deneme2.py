"""ARIMAX + MLP PURE HYBRID PIPELINE (Champion ML Logic + Full Academic Reporting)."""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List

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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
sns.set_style("whitegrid")

# --- KÜRESEL DEĞİŞKENLER ---
TARGET_TICKER = "EREGL.IS"
EXOGENOUS_MAP = {
    "EREGL.IS": ["KEUA", "TIO=F", "USDTRY=X"],
    "ISDMR.IS": ["KEUA", "TIO=F", "USDTRY=X"],
    "KRDMD.IS": ["KEUA", "SLX", "USDTRY=X"],
    "TUPRS.IS": ["KEUA", "BZ=F", "USDTRY=X"],
    "AKCNS.IS": ["KEUA", "NG=F", "USDTRY=X"],
}
CARBON_TICKER = "KEUA"
FX_TICKER = "USDTRY=X"
ROBUST_TEST_TICKERS = ["ISDMR", "KRDMD", "TUPRS"]

MODEL_COLORS = {
    "Baseline (OLS)": "#6c7a89",
    "XGBoost": "#72b7b2",
    "LSTM": "#f58518",
    "ARIMAX": "#4c78a8",
    "Hibrit ARIMAX-MLP": "#8b0000",
}

STRESS_BANDS = {"S1 (+%30)": 0.30, "S2 (+%60)": 0.60, "S3 (+%100)": 1.00}


# ==========================================
# 1. VERİ ÇEKME VE HAZIRLIK
# ==========================================
def fetch_and_clean_data(target_ticker: str = TARGET_TICKER) -> pd.DataFrame:

    exog_cols = EXOGENOUS_MAP[target_ticker]

    tickers = [target_ticker] + exog_cols
    raw = yf.download(
        tickers=tickers,
        start="2023-10-01",
        end="2026-05-17",
        interval="1d",
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"]
    else:
        close_obj = raw["Close"] if "Close" in raw.columns else raw
        df = (
            close_obj.to_frame(name=target_ticker)
            if isinstance(close_obj, pd.Series)
            else close_obj
        )
        if "Close" in df.columns:
            df = df.rename(columns={"Close": target_ticker})
    missing_cols = [col for col in tickers if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Eksik veri sütunları: {missing_cols}")
    return df.reindex(columns=tickers).ffill().bfill().dropna()


def run_single_ticker_all_models_test(target_ticker: str) -> pd.DataFrame:

    df_raw = fetch_and_clean_data(target_ticker=target_ticker)
    df_returns = get_log_returns(df_raw)

    train_df, val_df, test_df = strict_data_split(df_returns)

    y_all = df_returns[target_ticker]
    exog_cols = EXOGENOUS_MAP[target_ticker]
    exog_all = df_returns[exog_cols]
    carbon_col = exog_cols[0]
    commodity_col = exog_cols[1]
    fx_col = exog_cols[2]
    

    y_train = train_df[target_ticker]
    exog_train = train_df[exog_all.columns]

    # =========================================================
    # 1. BASELINE OLS
    # =========================================================
    baseline = LinearRegression().fit(exog_train, y_train)

    pred_base_test = pd.Series(
        baseline.predict(test_df[exog_all.columns]), index=test_df.index
    )

    # =========================================================
    # 2. XGBOOST
    # =========================================================
    xgb = XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
    )

    xgb.fit(exog_train, y_train)

    pred_xgb_test = pd.Series(
        xgb.predict(test_df[exog_all.columns]), index=test_df.index
    )

    # =========================================================
    # 3. LSTM
    # =========================================================
    pred_lstm_test = run_lstm_benchmark(
        y_train, val_df[target_ticker], test_df[target_ticker]
    )

    # =========================================================
    # 4. ARIMAX
    # =========================================================
    arimax_model = SARIMAX(y_train.values, exog=exog_train.values, order=(1, 0, 1)).fit(
        maxiter=1000, disp=False
    )

    pred_in = arimax_model.predict(start=0, end=len(y_train) - 1)

    exog_oos = exog_all.iloc[len(y_train) :].values

    pred_oos = arimax_model.predict(
        start=len(y_train), end=len(y_all) - 1, exog=exog_oos
    )

    pred_arimax_all = pd.Series(np.concatenate([pred_in, pred_oos]), index=y_all.index)

    pred_arimax_test = pd.Series(
        pred_arimax_all.loc[test_df.index].values, index=test_df.index
    )

    # =========================================================
    # 5. HIBRIT MLP
    # =========================================================
    true_residuals = y_all - pred_arimax_all

    mlp_features = pd.DataFrame(index=y_all.index)

    # CARBON
    mlp_features["carbon_t"] = exog_all[carbon_col]
    mlp_features["carbon_t_minus_1"] = exog_all[carbon_col].shift(1)

    # FX
    mlp_features["fx_t"] = exog_all[fx_col]
    mlp_features["fx_t_minus_1"] = exog_all[fx_col].shift(1)

    # RESIDUAL LAGS
    mlp_features["e_t_minus_1"] = true_residuals.shift(1)
    mlp_features["e_t_minus_2"] = true_residuals.shift(2)
    mlp_df = pd.concat(
        [mlp_features, true_residuals.rename("Target_Residual")], axis=1
    ).dropna()

    mlp_train = mlp_df.loc[mlp_df.index.isin(train_df.index)]
    mlp_val = mlp_df.loc[mlp_df.index.isin(val_df.index)]
    mlp_test = mlp_df.loc[mlp_df.index.isin(test_df.index)]

    X_mlp_train = mlp_train.drop(columns="Target_Residual")
    y_mlp_train = mlp_train["Target_Residual"]

    X_mlp_val = mlp_val.drop(columns="Target_Residual")
    y_mlp_val = mlp_val["Target_Residual"]

    X_mlp_test = mlp_test.drop(columns="Target_Residual")

    scaler_X = MinMaxScaler()

    X_mlp_train_sc = scaler_X.fit_transform(X_mlp_train)
    X_mlp_val_sc = scaler_X.transform(X_mlp_val)
    X_mlp_test_sc = scaler_X.transform(X_mlp_test)

    tf.keras.utils.set_random_seed(7)

    model_mlp = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_mlp_train_sc.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )

    model_mlp.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.Huber(delta=0.05),
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    model_mlp.fit(
        X_mlp_train_sc,
        y_mlp_train.values,
        validation_data=(X_mlp_val_sc, y_mlp_val.values),
        epochs=200,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=0,
    )

    mlp_pred_resid = pd.Series(
        model_mlp.predict(X_mlp_test_sc, verbose=0).ravel(), index=mlp_test.index
    )

    pred_hybrid_test = pred_arimax_test.loc[mlp_test.index] + mlp_pred_resid

    # =========================================================
    # METRICS
    # =========================================================
    idx = pred_hybrid_test.index

    y_test_final = test_df[target_ticker].loc[idx]

    predictions = {
        "Baseline (OLS)": pred_base_test.loc[idx],
        "XGBoost": pred_xgb_test.loc[idx],
        "LSTM": pred_lstm_test.loc[idx],
        "ARIMAX": pred_arimax_test.loc[idx],
        "Hibrit ARIMAX-MLP": pred_hybrid_test,
    }

    rows = []

    for model_name, preds in predictions.items():

        rmse = np.sqrt(mean_squared_error(y_test_final, preds))
        mae = mean_absolute_error(y_test_final, preds)

        rows.append(
            {
                "Hisse": target_ticker.replace(".IS", ""),
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "Test Gözlem": len(y_test_final),
            }
        )

    return pd.DataFrame(rows)


def run_robust_test_for_stocks(stocks: List[str]) -> pd.DataFrame:

    all_results = []

    for stock in stocks:

        ticker = stock if "." in stock else f"{stock}.IS"

        try:
            result_df = run_single_ticker_all_models_test(ticker)

            result_df["Durum"] = "Başarılı"

            all_results.append(result_df)

        except Exception as exc:

            error_df = pd.DataFrame(
                [
                    {
                        "Hisse": stock,
                        "Model": "Tümü",
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "Test Gözlem": 0,
                        "Durum": f"Hata: {exc}",
                    }
                ]
            )

            all_results.append(error_df)

    final_df = pd.concat(all_results, ignore_index=True)

    final_df = final_df.sort_values(
    by=["Hisse", "RMSE"],
    ascending=[True, True]
    ).reset_index(drop=True)

    return final_df


def get_log_returns(df: pd.DataFrame):
    return np.log(df / df.shift(1)).dropna()


def strict_data_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.80)
    val_end = train_end + int(n * 0.10)
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


# ==========================================
# 2. GÖRSELLEŞTİRME VE RAPORLAMA MODÜLLERİ
# ==========================================
def plot_raw_time_series(df_raw: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Ana başlık (y parametresine gerek kalmadı, tight_layout halledecek)
    fig.suptitle(
        "Ham Veri Zaman Serisi Özeti\n(Ekim 2023 - Mayıs 2026)",
        fontsize=14,
        fontweight="bold",
    )

    # EREGL.IS - Mavi
    axes[0].plot(df_raw.index, df_raw["EREGL.IS"], color="#1f77b4", linewidth=1.5)
    axes[0].set_title("EREGL.IS - Kapanış Fiyatı (TL)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Fiyat")

    # KEUA - Yeşil
    axes[1].plot(df_raw.index, df_raw["KEUA"], color="#2ca02c", linewidth=1.5)
    axes[1].set_title("KEUA - Karbon Fonu Fiyatı", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Fiyat")

    # TIO=F - Kırmızı
    axes[2].plot(df_raw.index, df_raw["TIO=F"], color="#d62728", linewidth=1.5)
    axes[2].set_title(
        "TIO=F - Demir Cevheri Vadeli İşlem Fiyatı", fontsize=10, fontweight="bold"
    )
    axes[2].set_ylabel("Fiyat")

    # USDTRY=X - Mor
    axes[3].plot(df_raw.index, df_raw["USDTRY=X"], color="#9467bd", linewidth=1.5)
    axes[3].set_title("USDTRY=X - USD/TRY Kuru", fontsize=10, fontweight="bold")
    axes[3].set_ylabel("Kur")
    axes[3].set_xlabel("Tarih")

    # KESİN ÇÖZÜM BURASI:
    # rect=[sol, alt, sağ, üst] -> Üst limiti 0.88 yaparak tepeye devasa bir nefes alma boşluğu bırakıyoruz.
    # h_pad=2.0 ise 4 grafiğin kendi aralarındaki dikey boşluğu açıyor.
    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2.0)

    plt.savefig("Grafik_0a_Ham_Veri_Zaman_Serisi.png")
    plt.close()
    print("Ham veri grafiği oluşturuldu: Grafik_0a_Ham_Veri_Zaman_Serisi.png")


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
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
        "Temel Değişkenler Korelasyon Isı Haritası\n(Ham Kapanış Fiyatları)",
        pad=15,
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("Grafik_0_Korelasyon_Heatmap.png")
    plt.close()


def plot_stress_test_fan_chart(last_test_date: pd.Timestamp, base_price: float):
    dates = pd.bdate_range(last_test_date + pd.offsets.BDay(1), periods=30)
    base = np.full(len(dates), base_price)

    plt.figure(figsize=(12, 6))
    plt.plot(
        dates,
        base,
        color="black",
        linestyle=":",
        linewidth=2.2,
        label="Baz Senaryo Fiyatı",
    )

    colors = {"S1 (+%30)": "#ffb703", "S2 (+%60)": "#fb8500", "S3 (+%100)": "#d00000"}
    prev_upper, prev_lower = base.copy(), base.copy()

    for scenario, shock in STRESS_BANDS.items():
        upper = base * (1 + shock)
        lower = base * (1 - shock)
        plt.plot(dates, upper, color=colors[scenario], linewidth=1.4, alpha=0.9)
        plt.plot(dates, lower, color=colors[scenario], linewidth=1.4, alpha=0.9)
        plt.fill_between(
            dates,
            prev_upper,
            upper,
            color=colors[scenario],
            alpha=0.12,
            label=f"{scenario} Üst Bant",
        )
        plt.fill_between(
            dates,
            lower,
            prev_lower,
            color=colors[scenario],
            alpha=0.12,
            label=f"{scenario} Alt Bant",
        )
        prev_upper, prev_lower = upper, lower

    plt.title(
        "Test Sonrası 30 Gün Karbon Stres Testi Yelpaze Grafiği (Gerçek Fiyat)",
        pad=15,
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Tarih")
    plt.ylabel("Hisse Fiyatı (TL)")
    plt.legend(loc="upper left", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("Grafik_4_Stres_Testi_Fan.png")
    plt.close()


def run_stress_test(base_price: float) -> pd.DataFrame:
    rows = [
        {
            "Senaryo": sc,
            "Şok Oranı": f"+%{int(sh * 100)}",
            "Alt Bant Fiyatı": base_price * (1 - sh),
            "Baz Senaryo Fiyatı": base_price,
            "Üst Bant Fiyatı": base_price * (1 + sh),
        }
        for sc, sh in STRESS_BANDS.items()
    ]
    return pd.DataFrame(rows)


def plot_all_visualizations(
    y_test_final, predictions_df, metrics_df, hybrid_resid, xgb_resid
):
    # 1. RMSE Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=metrics_df,
        x="Model",
        y="RMSE",
        palette=[MODEL_COLORS.get(x, "#333") for x in metrics_df["Model"]],
    )
    ax = plt.gca()
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
    plt.title(
        "Modellerin Test Kümesi RMSE Karşılaştırması",
        pad=15,
        fontsize=12,
        fontweight="bold",
    )
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("Grafik_1_RMSE_Bar.png")
    plt.close()

    # 2. Prediction Line Chart
    plt.figure(figsize=(12, 6))
    plt.plot(
        y_test_final.index,
        y_test_final.values,
        color="black",
        label="Gerçek EREGL.IS",
        linewidth=2.2,
        linestyle="--",
    )
    for model_col in predictions_df.columns:
        lw = 2.5 if "Hibrit" in model_col else 1.2
        alpha = 1.0 if "Hibrit" in model_col else 0.75
        plt.plot(
            predictions_df.index,
            predictions_df[model_col],
            label=model_col,
            color=MODEL_COLORS.get(model_col, "#808080"),
            linewidth=lw,
            alpha=alpha,
        )
    plt.title(
        "Zaman Serisi Tahmin Performansı (Gerçek vs. Modeller)",
        pad=15,
        fontsize=12,
        fontweight="bold",
    )
    plt.ylabel("Log Getiri")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("Grafik_2_Prediction_Line.png")
    plt.close()

    # 3. Residual KDE
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=xgb_resid,
        label="XGBoost Hata Dağılımı",
        color=MODEL_COLORS["XGBoost"],
        fill=True,
        alpha=0.3,
    )
    sns.kdeplot(
        data=hybrid_resid,
        label="Hibrit Model Hata Dağılımı",
        color=MODEL_COLORS["Hibrit ARIMAX-MLP"],
        fill=True,
        alpha=0.5,
    )
    plt.title(
        "Hata Dağılımı Çekirdek Yoğunluk Tahmini (KDE)",
        pad=15,
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Tahmin Hatası (Gerçek - Tahmin)")
    plt.ylabel("Yoğunluk (Density)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Grafik_3_Residual_KDE.png")
    plt.close()


def write_thesis_report(
    raw_stats,
    raw_corr,
    ret_stats,
    adf_res,
    arimax_summary,
    arimax_coef,
    diag_df,
    metrics_df,
    stress_df,
):
    lines = [
        "TEZ BULGULARI RAPORU",
        "=" * 80,
        "",
        "TABLO 4.1: BETİMSEL İSTATİSTİKLER (LOG GETİRİ)",
        "-" * 80,
        ret_stats.to_string(float_format=lambda x: f"{x:.6f}"),
        "",
        "=" * 80,
        "",
        "1) TEMEL İSTATİSTİKLER VE KORELASYON ANALİZİ (HAM VERİ)",
        "-" * 80,
        "1a) Temel İstatistikler (Ham Kapanış Fiyatları):",
        raw_stats.to_string(float_format=lambda x: f"{x:.4f}"),
        "",
        "1b) Korelasyon Matrisi (Ham Kapanış Fiyatları):",
        raw_corr.to_string(float_format=lambda x: f"{x:.6f}"),
        "",
        "2) ADF TEST SONUÇLARI (GETİRİ SERİSİ)",
        "-" * 80,
    ]
    for col, pval in adf_res.items():
        lines.append(
            f"{col}: ADF p-değeri={pval:.6f} -> {'DURAĞAN' if pval < 0.05 else 'DURAĞAN DEĞİL'}"
        )

    lines.extend(
        [
            "",
            "3) ARIMAX MODEL ÖZETİ",
            "-" * 80,
            arimax_summary,
            "",
            "TABLO 4.2: ARIMAX KATSAYI TABLOSU",
            "-" * 80,
            arimax_coef,
            "",
            "4) RESIDUAL TANI TESTLERİ (LJUNG-BOX / ARCH-LM - HİBRİT MODEL)",
            "-" * 80,
            diag_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
            "",
            "5) TEST KÜMESİ PERFORMANS METRİKLERİ (RMSE / MAE)",
            "-" * 80,
            metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
            "",
            "6) STRES TESTİ SONUÇ TABLOSU (GERÇEK TL FİYATI ÜZERİNDEN)",
            "-" * 80,
            stress_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
            "",
        ]
    )

    Path("Tez_Bulgulari_Raporu.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\nTez raporu başarıyla oluşturuldu: Tez_Bulgulari_Raporu.txt")


# ==========================================
# 3. BENCHMARK: LSTM
# ==========================================
def run_lstm_benchmark(y_train, y_val, y_test, window_size=5):
    def create_dataset(series):
        X, y = [], []
        for i in range(window_size, len(series)):
            X.append(series[i - window_size : i])
            y.append(series[i])
        return np.array(X).reshape(-1, window_size, 1), np.array(y)

    X_tr, y_tr = create_dataset(y_train.values)
    X_val, y_val_arr = create_dataset(
        pd.concat([y_train.iloc[-window_size:], y_val]).values
    )
    X_te, _ = create_dataset(pd.concat([y_val.iloc[-window_size:], y_test]).values)

    tf.keras.utils.set_random_seed(42)
    model = Sequential([LSTM(32, input_shape=(window_size, 1)), Dropout(0.2), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val_arr),
        epochs=100,
        batch_size=16,
        callbacks=[es],
        verbose=0,
    )

    return pd.Series(model.predict(X_te, verbose=0).ravel(), index=y_test.index)


# ==========================================
# 4. ANA ÇALIŞTIRMA BORU HATTI
# ==========================================
def run_pipeline():
    print("=== 1. Veri Hazırlama ===")
    df_raw = fetch_and_clean_data()

    plot_raw_time_series(df_raw)

    df_returns = get_log_returns(df_raw)

    # Veri Raporu İçin İstatistikler
    raw_stats = df_raw.describe()
    raw_corr = df_raw.corr()
    plot_correlation_heatmap(raw_corr)

    ret_stats = df_returns.describe().T[["mean", "std", "min", "max"]]
    ret_stats["skewness"] = df_returns.skew()
    ret_stats.columns = [
        "Ortalama (Mean)",
        "Standart Sapma (Std)",
        "Min",
        "Max",
        "Çarpıklık (Skewness)",
    ]

    adf_res = {col: adfuller(df_returns[col].dropna())[1] for col in df_returns.columns}

    train_df, val_df, test_df = strict_data_split(df_returns)
    y_all = df_returns[TARGET_TICKER]

    exog_cols = EXOGENOUS_MAP[TARGET_TICKER]

    CARBON_TICKER = exog_cols[0]
    COMMODITY_TICKER = exog_cols[1]
    FX_TICKER = exog_cols[2]

    exog_all = df_returns[exog_cols]

    y_train = train_df[TARGET_TICKER]
    exog_train = train_df[exog_all.columns]

    print("\n=== 2. Benchmark Modeller Eğitiliyor ===")
    baseline = LinearRegression().fit(exog_train, y_train)
    pred_base_test = pd.Series(
        baseline.predict(test_df[exog_all.columns]),
        index=test_df.index,
        name="Baseline (OLS)",
    )

    xgb = XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
    )
    xgb.fit(exog_train, y_train)
    pred_xgb_test = pd.Series(
        xgb.predict(test_df[exog_all.columns]), index=test_df.index, name="XGBoost"
    )

    pred_lstm_test = run_lstm_benchmark(
        y_train, val_df[TARGET_TICKER], test_df[TARGET_TICKER]
    )
    pred_lstm_test.name = "LSTM"

    print("ARIMAX(1,0,1) Modeli Eğitiliyor...")
    arimax_model = SARIMAX(y_train.values, exog=exog_train.values, order=(1, 0, 1)).fit(
        maxiter=1000, disp=False
    )
    arimax_summary = str(arimax_model.summary())
    arimax_coef = str(arimax_model.summary().tables[1])

    pred_in = arimax_model.predict(start=0, end=len(y_train) - 1)
    exog_oos = exog_all.iloc[len(y_train) :].values
    pred_oos = arimax_model.predict(
        start=len(y_train), end=len(y_all) - 1, exog=exog_oos
    )

    pred_arimax_all = pd.Series(np.concatenate([pred_in, pred_oos]), index=y_all.index)
    pred_arimax_test = pd.Series(
        pred_arimax_all.loc[test_df.index].values, index=test_df.index, name="ARIMAX"
    )

    print("\n=== 3. Hibrit MLP Feature Engineering ===")
    true_residuals = y_all - pred_arimax_all

    mlp_features = pd.DataFrame(index=y_all.index)
    mlp_features["X1_t"] = exog_all[CARBON_TICKER]
    mlp_features["X1_t_minus_1"] = exog_all[CARBON_TICKER].shift(1)
    mlp_features["X3_t"] = exog_all[FX_TICKER]
    mlp_features["X3_t_minus_1"] = exog_all[FX_TICKER].shift(1)
    mlp_features["e_t_minus_1"] = true_residuals.shift(1)
    mlp_features["e_t_minus_2"] = true_residuals.shift(2)

    mlp_df = pd.concat(
        [mlp_features, true_residuals.rename("Target_Residual")], axis=1
    ).dropna()

    mlp_train = mlp_df.loc[mlp_df.index.isin(train_df.index)]
    mlp_val = mlp_df.loc[mlp_df.index.isin(val_df.index)]
    mlp_test = mlp_df.loc[mlp_df.index.isin(test_df.index)]

    X_mlp_train, y_mlp_train = (
        mlp_train.drop(columns="Target_Residual"),
        mlp_train["Target_Residual"],
    )
    X_mlp_val, y_mlp_val = (
        mlp_val.drop(columns="Target_Residual"),
        mlp_val["Target_Residual"],
    )
    X_mlp_test = mlp_test.drop(columns="Target_Residual")

    scaler_X = MinMaxScaler()
    X_mlp_train_sc = scaler_X.fit_transform(X_mlp_train)
    X_mlp_val_sc = scaler_X.transform(X_mlp_val)
    X_mlp_test_sc = scaler_X.transform(X_mlp_test)

    print("\n=== 4. Yapay Sinir Ağı (MLP) Eğitiliyor ===")
    tf.keras.utils.set_random_seed(7)

    model_mlp = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_mlp_train_sc.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )

    model_mlp.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.Huber(delta=0.05),
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    model_mlp.fit(
        X_mlp_train_sc,
        y_mlp_train.values,
        validation_data=(X_mlp_val_sc, y_mlp_val.values),
        epochs=200,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=0,
    )

    mlp_pred_resid = pd.Series(
        model_mlp.predict(X_mlp_test_sc, verbose=0).ravel(), index=mlp_test.index
    )
    pred_hybrid_test = pred_arimax_test.loc[mlp_test.index] + mlp_pred_resid
    pred_hybrid_test.name = "Hibrit ARIMAX-MLP"

    print("\n=== 5. Analiz, Raporlama ve Stres Testi ===")
    idx = pred_hybrid_test.index
    y_test_final = test_df[TARGET_TICKER].loc[idx]

    predictions_df = pd.DataFrame(
        {
            pred_base_test.name: pred_base_test.loc[idx],
            pred_xgb_test.name: pred_xgb_test.loc[idx],
            pred_lstm_test.name: pred_lstm_test.loc[idx],
            pred_arimax_test.name: pred_arimax_test.loc[idx],
            pred_hybrid_test.name: pred_hybrid_test,
        }
    )

    metrics = []
    for col in predictions_df.columns:
        p = predictions_df[col]
        rmse = float(np.sqrt(mean_squared_error(y_test_final, p)))
        mae = float(mean_absolute_error(y_test_final, p))
        metrics.append({"Model": col, "RMSE": rmse, "MAE": mae})

    metrics_df = pd.DataFrame(metrics).sort_values(by="RMSE").reset_index(drop=True)
    print("\nTest Seti Performans Tablosu:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Diagnostik Testler
    hybrid_resid = y_test_final - pred_hybrid_test
    xgb_resid = y_test_final - predictions_df["XGBoost"]

    lb_pval = float(
        acorr_ljungbox(hybrid_resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    )
    arch_pval = float(het_arch(hybrid_resid.dropna())[1])
    diag_df = pd.DataFrame(
        [
            {"Test": "Ljung-Box (lag=10)", "p-değeri": lb_pval},
            {"Test": "ARCH-LM", "p-değeri": arch_pval},
        ]
    )

    # Stres Testi (Gerçek Fiyat üzerinden)
    real_base_price = float(df_raw[TARGET_TICKER].iloc[-1])
    stress_df = run_stress_test(real_base_price)

    # Görseller ve Rapor
    plot_stress_test_fan_chart(y_test_final.index[-1], real_base_price)
    plot_all_visualizations(
        y_test_final, predictions_df, metrics_df, hybrid_resid, xgb_resid
    )

    write_thesis_report(
        raw_stats,
        raw_corr,
        ret_stats,
        adf_res,
        arimax_summary,
        arimax_coef,
        diag_df,
        metrics_df,
        stress_df,
    )

    print("\n=== 6. Robust Test (AKCNS, TUPRS, KRDMD) ===")
    robust_df = run_robust_test_for_stocks(ROBUST_TEST_TICKERS)
    print(robust_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    Path("Robust_Test_Sonuclari.txt").write_text(
        robust_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        encoding="utf-8",
    )
    print("Robust test raporu oluşturuldu: Robust_Test_Sonuclari.txt")


if __name__ == "__main__":
    run_pipeline()
