"""ARIMAX + MLP PURE HYBRID PIPELINE (Champion ML Logic + Full Academic Reporting)."""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from math import erf, sqrt
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
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
CARBON_TICKER = "KEUA"
IRON_TICKER = "TIO=F"
FX_TICKER = "USDTRY=X"

MODEL_COLORS = {
    "Baseline (OLS)": "#6c7a89", 
    "XGBoost": "#72b7b2",
    "LSTM": "#f58518",
    "ARIMAX": "#4c78a8", 
    "Hibrit ARIMAX-MLP": "#8b0000"
}

STRESS_BANDS = {"S1 (+%30)": 0.30, "S2 (+%60)": 0.60, "S3 (+%100)": 1.00}

# ==========================================
# 1. VERİ ÇEKME VE HAZIRLIK
# ==========================================
def fetch_and_clean_data() -> pd.DataFrame:
    tickers = [TARGET_TICKER, CARBON_TICKER, IRON_TICKER, FX_TICKER]
    raw = yf.download(tickers=tickers, start="2023-10-01", end="2026-05-17", interval="1d", progress=False)
    df = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw.rename(columns={"Close": TARGET_TICKER})[tickers]
    return df.reindex(columns=tickers).ffill().bfill().dropna()

def get_log_returns(df: pd.DataFrame):
    return np.log(df / df.shift(1)).dropna()

def strict_data_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.80)
    val_end = train_end + int(n * 0.10)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

# ==========================================
# 2. GÖRSELLEŞTİRME VE RAPORLAMA MODÜLLERİ
# ==========================================
def plot_raw_time_series(df_raw: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Ana başlık (y parametresine gerek kalmadı, tight_layout halledecek)
    fig.suptitle("Ham Veri Zaman Serisi Özeti\n(Ekim 2023 - Mayıs 2026)", 
                 fontsize=14, fontweight="bold")

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
    axes[2].set_title("TIO=F - Demir Cevheri Vadeli İşlem Fiyatı", fontsize=10, fontweight="bold")
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
    sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Temel Değişkenler Korelasyon Isı Haritası\n(Ham Kapanış Fiyatları)", pad=15, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Grafik_0_Korelasyon_Heatmap.png")
    plt.close()

def plot_stress_test_fan_chart(last_test_date: pd.Timestamp, base_price: float):
    dates = pd.bdate_range(last_test_date + pd.offsets.BDay(1), periods=30)
    base = np.full(len(dates), base_price)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, base, color="black", linestyle=":", linewidth=2.2, label="Baz Senaryo Fiyatı")

    colors = {"S1 (+%30)": "#ffb703", "S2 (+%60)": "#fb8500", "S3 (+%100)": "#d00000"}
    prev_upper, prev_lower = base.copy(), base.copy()

    for scenario, shock in STRESS_BANDS.items():
        upper = base * (1 + shock)
        lower = base * (1 - shock)
        plt.plot(dates, upper, color=colors[scenario], linewidth=1.4, alpha=0.9)
        plt.plot(dates, lower, color=colors[scenario], linewidth=1.4, alpha=0.9)
        plt.fill_between(dates, prev_upper, upper, color=colors[scenario], alpha=0.12, label=f"{scenario} Üst Bant")
        plt.fill_between(dates, lower, prev_lower, color=colors[scenario], alpha=0.12, label=f"{scenario} Alt Bant")
        prev_upper, prev_lower = upper, lower

    plt.title("Test Sonrası 30 Gün Karbon Stres Testi Yelpaze Grafiği (Gerçek Fiyat)", pad=15, fontsize=12, fontweight="bold")
    plt.xlabel("Tarih")
    plt.ylabel("Hisse Fiyatı (TL)")
    plt.legend(loc="upper left", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("Grafik_4_Stres_Testi_Fan.png")
    plt.close()

def run_stress_test(base_price: float) -> pd.DataFrame:
    rows = [{"Senaryo": sc, "Şok Oranı": f"+%{int(sh * 100)}", "Alt Bant Fiyatı": base_price * (1 - sh), 
             "Baz Senaryo Fiyatı": base_price, "Üst Bant Fiyatı": base_price * (1 + sh)} for sc, sh in STRESS_BANDS.items()]
    return pd.DataFrame(rows)

def plot_all_visualizations(y_test_final, predictions_df, metrics_df, hybrid_resid, xgb_resid):
    # 1. RMSE Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x="Model", y="RMSE", palette=[MODEL_COLORS.get(x, "#333") for x in metrics_df["Model"]])
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.6f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    plt.title("Modellerin Test Kümesi RMSE Karşılaştırması", pad=15, fontsize=12, fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("Grafik_1_RMSE_Bar.png")
    plt.close()

    # 2. Prediction Line Chart
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_final.index, y_test_final.values, color="black", label="Gerçek EREGL.IS", linewidth=2.2, linestyle="--")
    for model_col in predictions_df.columns:
        lw = 2.5 if "Hibrit" in model_col else 1.2
        alpha = 1.0 if "Hibrit" in model_col else 0.75
        plt.plot(predictions_df.index, predictions_df[model_col], label=model_col, 
                 color=MODEL_COLORS.get(model_col, "#808080"), linewidth=lw, alpha=alpha)
    plt.title("Zaman Serisi Tahmin Performansı (Gerçek vs. Modeller)", pad=15, fontsize=12, fontweight="bold")
    plt.ylabel("Log Getiri")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("Grafik_2_Prediction_Line.png")
    plt.close()

    # 3. Residual KDE
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=xgb_resid, label="XGBoost Hata Dağılımı", color=MODEL_COLORS["XGBoost"], fill=True, alpha=0.3)
    sns.kdeplot(data=hybrid_resid, label="Hibrit Model Hata Dağılımı", color=MODEL_COLORS["Hibrit ARIMAX-MLP"], fill=True, alpha=0.5)
    plt.title("Hata Dağılımı Çekirdek Yoğunluk Tahmini (KDE)", pad=15, fontsize=12, fontweight="bold")
    plt.xlabel("Tahmin Hatası (Gerçek - Tahmin)")
    plt.ylabel("Yoğunluk (Density)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Grafik_3_Residual_KDE.png")
    plt.close()

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def diebold_mariano_test(
    y_true: pd.Series,
    pred_model_a: pd.Series,
    pred_model_b: pd.Series,
    loss_power: int = 2,
) -> Tuple[float, float, str]:
    """One-step-ahead Diebold-Mariano testi (h=1) döndürür: (DM istatistiği, p-value, yorum)."""
    idx = y_true.index.intersection(pred_model_a.index).intersection(pred_model_b.index)
    y = y_true.loc[idx]
    e1 = (y - pred_model_a.loc[idx]).astype(float)
    e2 = (y - pred_model_b.loc[idx]).astype(float)

    loss_diff = np.abs(e1) ** loss_power - np.abs(e2) ** loss_power
    n = len(loss_diff)
    if n < 2:
        return np.nan, np.nan, "Yetersiz gözlem"

    d_bar = float(loss_diff.mean())
    d_var = float(loss_diff.var(ddof=1))
    if np.isclose(d_var, 0.0):
        return np.nan, 1.0, "Anlamlı fark yok (varyans≈0)"

    dm_stat = d_bar / np.sqrt(d_var / n)
    p_value = 2.0 * (1.0 - _normal_cdf(abs(dm_stat)))
    interpretation = "istatistiksel olarak anlamlı" if p_value < 0.05 else "istatistiksel olarak anlamsız"
    return float(dm_stat), float(p_value), interpretation


def run_dm_tests(
    y_true: pd.Series,
    predictions_df: pd.DataFrame,
    hybrid_model_name: str = "Hibrit ARIMAX-MLP",
) -> pd.DataFrame:
    """Hibrit modelin diğer tüm modellere karşı DM test sonuçlarını tablo halinde döndürür."""
    rows = []
    for model_name in predictions_df.columns:
        if model_name == hybrid_model_name:
            continue
        dm_stat, p_value, interpretation = diebold_mariano_test(
            y_true=y_true,
            pred_model_a=predictions_df[hybrid_model_name],
            pred_model_b=predictions_df[model_name],
        )
        rows.append(
            {
                "Karşılaştırma": f"{hybrid_model_name} vs {model_name}",
                "DM İstatistiği": dm_stat,
                "p-value": p_value,
                "Yorum": interpretation,
            }
        )
    return pd.DataFrame(rows)


def split_fold_train_val(train_val_df: pd.DataFrame, val_ratio: float = 0.2, min_val_size: int = 20):
    """Walk-forward fold içindeki train kısmını leakage-safe train/val olarak ayırır."""
    if len(train_val_df) < (min_val_size * 2):
        split_idx = max(int(len(train_val_df) * (1 - val_ratio)), 1)
    else:
        split_idx = len(train_val_df) - min_val_size
    split_idx = min(max(split_idx, 1), len(train_val_df) - 1)
    return train_val_df.iloc[:split_idx].copy(), train_val_df.iloc[split_idx:].copy()


def train_and_predict_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    return_arimax_details: bool = False,
):
    """Tek split için tüm benchmark + hibrit modeli yeniden eğitir ve test tahminlerini döndürür."""
    y_train = train_df[TARGET_TICKER]
    y_val = val_df[TARGET_TICKER]
    y_test = test_df[TARGET_TICKER]

    exog_cols = [CARBON_TICKER, IRON_TICKER, FX_TICKER]
    exog_train = train_df[exog_cols]
    exog_val = val_df[exog_cols]
    exog_test = test_df[exog_cols]

    # Baseline (OLS)
    baseline = LinearRegression().fit(exog_train, y_train)
    pred_base_test = pd.Series(baseline.predict(exog_test), index=test_df.index, name="Baseline (OLS)")

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    xgb.fit(exog_train, y_train)
    pred_xgb_test = pd.Series(xgb.predict(exog_test), index=test_df.index, name="XGBoost")

    # LSTM
    pred_lstm_test = run_lstm_benchmark(y_train, y_val, y_test)
    pred_lstm_test.name = "LSTM"

    # ARIMAX
    arimax_model = SARIMAX(y_train.values, exog=exog_train.values, order=(1, 0, 1)).fit(maxiter=1000, disp=False)
    arimax_summary = str(arimax_model.summary())
    arimax_coef = str(arimax_model.summary().tables[1])

    fold_all = pd.concat([train_df, val_df, test_df]).sort_index()
    y_all = fold_all[TARGET_TICKER]
    exog_all = fold_all[exog_cols]

    pred_in = arimax_model.predict(start=0, end=len(y_train) - 1)
    exog_oos = exog_all.iloc[len(y_train):].values
    pred_oos = arimax_model.predict(start=len(y_train), end=len(y_all) - 1, exog=exog_oos)
    pred_arimax_all = pd.Series(np.concatenate([pred_in, pred_oos]), index=y_all.index)
    pred_arimax_test = pd.Series(pred_arimax_all.loc[test_df.index].values, index=test_df.index, name="ARIMAX")

    # Hibrit MLP residual modeli (yalnızca geçmiş bilgi ile)
    true_residuals = y_all - pred_arimax_all
    mlp_features = pd.DataFrame(index=y_all.index)
    mlp_features["X1_t"] = exog_all[CARBON_TICKER]
    mlp_features["X1_t_minus_1"] = exog_all[CARBON_TICKER].shift(1)
    mlp_features["X3_t"] = exog_all[FX_TICKER]
    mlp_features["X3_t_minus_1"] = exog_all[FX_TICKER].shift(1)
    mlp_features["e_t_minus_1"] = true_residuals.shift(1)
    mlp_features["e_t_minus_2"] = true_residuals.shift(2)
    mlp_df = pd.concat([mlp_features, true_residuals.rename("Target_Residual")], axis=1).dropna()

    mlp_train = mlp_df.loc[mlp_df.index.isin(train_df.index)]
    mlp_val = mlp_df.loc[mlp_df.index.isin(val_df.index)]
    mlp_test = mlp_df.loc[mlp_df.index.isin(test_df.index)]

    X_mlp_train, y_mlp_train = mlp_train.drop(columns="Target_Residual"), mlp_train["Target_Residual"]
    X_mlp_val, y_mlp_val = mlp_val.drop(columns="Target_Residual"), mlp_val["Target_Residual"]
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
    model_mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.Huber(delta=0.05))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model_mlp.fit(
        X_mlp_train_sc,
        y_mlp_train.values,
        validation_data=(X_mlp_val_sc, y_mlp_val.values),
        epochs=200,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=0,
    )

    mlp_pred_resid = pd.Series(model_mlp.predict(X_mlp_test_sc, verbose=0).ravel(), index=mlp_test.index)
    pred_hybrid_test = pred_arimax_test.loc[mlp_test.index] + mlp_pred_resid
    pred_hybrid_test.name = "Hibrit ARIMAX-MLP"

    idx = pred_hybrid_test.index
    y_test_final = y_test.loc[idx]
    predictions_df = pd.DataFrame(
        {
            pred_base_test.name: pred_base_test.loc[idx],
            pred_xgb_test.name: pred_xgb_test.loc[idx],
            pred_lstm_test.name: pred_lstm_test.loc[idx],
            pred_arimax_test.name: pred_arimax_test.loc[idx],
            pred_hybrid_test.name: pred_hybrid_test,
        }
    )
    artifacts = {
        "y_test_final": y_test_final,
        "predictions_df": predictions_df,
        "hybrid_resid": y_test_final - pred_hybrid_test,
        "xgb_resid": y_test_final - predictions_df["XGBoost"],
    }
    if return_arimax_details:
        artifacts["arimax_summary"] = arimax_summary
        artifacts["arimax_coef"] = arimax_coef
    return artifacts


def walk_forward_cross_validation(df_returns: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """TimeSeriesSplit ile walk-forward CV: her fold'da tüm modelleri yeniden eğitir."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model_order = ["Baseline (OLS)", "XGBoost", "LSTM", "ARIMAX", "Hibrit ARIMAX-MLP"]
    fold_rows: List[Dict[str, float]] = []

    for fold_id, (train_val_idx, test_idx) in enumerate(tscv.split(df_returns), start=1):
        fold_train_val = df_returns.iloc[train_val_idx].copy()
        fold_test = df_returns.iloc[test_idx].copy()
        fold_train, fold_val = split_fold_train_val(fold_train_val, val_ratio=0.2, min_val_size=max(10, len(fold_test)))

        fold_artifacts = train_and_predict_all_models(fold_train, fold_val, fold_test)
        y_fold = fold_artifacts["y_test_final"]
        pred_fold = fold_artifacts["predictions_df"]

        row: Dict[str, float] = {"Fold": fold_id}
        for model_name in model_order:
            row[model_name] = float(np.sqrt(mean_squared_error(y_fold, pred_fold[model_name])))
        fold_rows.append(row)

    fold_rmse_df = pd.DataFrame(fold_rows)
    return fold_rmse_df


def summarize_walk_forward_rmse(fold_rmse_df: pd.DataFrame) -> pd.DataFrame:
    """Fold RMSE sonuçlarından akademik özet tablo üretir (ortalama ± std)."""
    model_cols = [c for c in fold_rmse_df.columns if c != "Fold"]
    rows = []
    for model_name in model_cols:
        rows.append(
            {
                "Model": model_name,
                "RMSE Mean": float(fold_rmse_df[model_name].mean()),
                "RMSE Std": float(fold_rmse_df[model_name].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values(by="RMSE Mean").reset_index(drop=True)


def bootstrap_rmse_confidence_interval(
    y_true: pd.Series,
    y_pred: pd.Series,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
):
    """Hibrit RMSE için bootstrap %95 güven aralığı hesaplar."""
    idx = y_true.index.intersection(y_pred.index)
    y = y_true.loc[idx].values
    p = y_pred.loc[idx].values
    n = len(y)
    rng = np.random.default_rng(seed=random_state)

    rmse_samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        rmse_samples[i] = float(np.sqrt(mean_squared_error(y[sample_idx], p[sample_idx])))

    alpha = 1.0 - confidence_level
    low = float(np.quantile(rmse_samples, alpha / 2.0))
    high = float(np.quantile(rmse_samples, 1.0 - alpha / 2.0))
    point_rmse = float(np.sqrt(mean_squared_error(y, p)))
    return point_rmse, low, high, rmse_samples


def plot_fold_rmse_comparison(fold_rmse_df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    model_cols = [c for c in fold_rmse_df.columns if c != "Fold"]
    for model_name in model_cols:
        plt.plot(
            fold_rmse_df["Fold"],
            fold_rmse_df[model_name],
            marker="o",
            linewidth=1.8,
            label=model_name,
            color=MODEL_COLORS.get(model_name, None),
        )
    plt.title("Walk-Forward CV Fold Bazlı RMSE Karşılaştırması", pad=15, fontsize=12, fontweight="bold")
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("Grafik_5_Fold_Bazli_RMSE.png")
    plt.close()


def plot_bootstrap_rmse_histogram(rmse_samples: np.ndarray):
    plt.figure(figsize=(10, 6))
    sns.histplot(rmse_samples, bins=30, kde=True, color="#8b0000")
    plt.title("Bootstrap RMSE Dağılımı (Hibrit ARIMAX-MLP)", pad=15, fontsize=12, fontweight="bold")
    plt.xlabel("RMSE")
    plt.ylabel("Frekans")
    plt.tight_layout()
    plt.savefig("Grafik_6_Bootstrap_RMSE_Histogram.png")
    plt.close()


def plot_dm_pvalue_bar(dm_df: pd.DataFrame):
    if dm_df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(data=dm_df, x="Karşılaştırma", y="p-value", palette="viridis")
    plt.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="p = 0.05")
    plt.title("Diebold-Mariano Testi p-value Karşılaştırması", pad=15, fontsize=12, fontweight="bold")
    plt.ylabel("p-value")
    plt.xlabel("Model Karşılaştırması")
    plt.xticks(rotation=15)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("Grafik_7_DM_pvalue_Bar.png")
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
    dm_results_df: Optional[pd.DataFrame] = None,
    fold_rmse_df: Optional[pd.DataFrame] = None,
    cv_summary_df: Optional[pd.DataFrame] = None,
    bootstrap_text: Optional[str] = None,
):
    lines = [
        "TEZ BULGULARI RAPORU", "=" * 80, "",
        "TABLO 4.1: BETİMSEL İSTATİSTİKLER (LOG GETİRİ)", "-" * 80,
        ret_stats.to_string(float_format=lambda x: f"{x:.6f}"), "", "=" * 80, "",
        "1) TEMEL İSTATİSTİKLER VE KORELASYON ANALİZİ (HAM VERİ)", "-" * 80,
        "1a) Temel İstatistikler (Ham Kapanış Fiyatları):", raw_stats.to_string(float_format=lambda x: f"{x:.4f}"), "",
        "1b) Korelasyon Matrisi (Ham Kapanış Fiyatları):", raw_corr.to_string(float_format=lambda x: f"{x:.6f}"), "",
        "2) ADF TEST SONUÇLARI (GETİRİ SERİSİ)", "-" * 80
    ]
    for col, pval in adf_res.items():
        lines.append(f"{col}: ADF p-değeri={pval:.6f} -> {'DURAĞAN' if pval < 0.05 else 'DURAĞAN DEĞİL'}")
        
    lines.extend([
        "", "3) ARIMAX MODEL ÖZETİ", "-" * 80, arimax_summary, "",
        "TABLO 4.2: ARIMAX KATSAYI TABLOSU", "-" * 80, arimax_coef, "",
        "4) RESIDUAL TANI TESTLERİ (LJUNG-BOX / ARCH-LM - HİBRİT MODEL)", "-" * 80,
        diag_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"), "",
        "5) TEST KÜMESİ PERFORMANS METRİKLERİ (RMSE / MAE)", "-" * 80,
        metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"), "",
        "6) STRES TESTİ SONUÇ TABLOSU (GERÇEK TL FİYATI ÜZERİNDEN)", "-" * 80,
        stress_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"), ""
    ])

    if dm_results_df is not None and not dm_results_df.empty:
        lines.extend(
            [
                "=" * 80,
                "7) DIEBOLD-MARIANO TEST SONUÇLARI (HİBRİT vs DİĞER MODELLER)",
                "-" * 80,
                dm_results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if pd.notna(x) else "NaN"),
                "",
                "Akademik Yorum: p < 0.05 olan karşılaştırmalarda hibrit model ile rakip model tahmin hataları arasında istatistiksel olarak anlamlı fark vardır.",
                "",
            ]
        )

    if fold_rmse_df is not None and not fold_rmse_df.empty:
        lines.extend(
            [
                "=" * 80,
                "8) WALK-FORWARD CROSS-VALIDATION (TimeSeriesSplit, 5 split)",
                "-" * 80,
                "Fold Bazlı RMSE Tablosu:",
                fold_rmse_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
                "",
            ]
        )

    if cv_summary_df is not None and not cv_summary_df.empty:
        lines.extend(
            [
                "Ortalama RMSE ± Std Özeti:",
                cv_summary_df.assign(
                    **{
                        "RMSE Mean ± Std": cv_summary_df.apply(
                            lambda r: f"{r['RMSE Mean']:.6f} ± {r['RMSE Std']:.6f}", axis=1
                        )
                    }
                )[["Model", "RMSE Mean ± Std"]].to_string(index=False),
                "",
                "Akademik Yorum: Ortalama RMSE ve standart sapma birlikte değerlendirilerek hem doğruluk hem de kararlılık raporlanmıştır.",
                "",
            ]
        )

    if bootstrap_text is not None:
        lines.extend(
            [
                "=" * 80,
                "9) BOOTSTRAP CONFIDENCE INTERVAL (HİBRİT RMSE, 1000 ÖRNEKLEME)",
                "-" * 80,
                bootstrap_text,
                "",
                "Akademik Yorum: Bootstrap güven aralığı tahmin performansının örnekleme belirsizliğine duyarlılığını gösterir.",
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
            X.append(series[i - window_size: i])
            y.append(series[i])
        return np.array(X).reshape(-1, window_size, 1), np.array(y)

    X_tr, y_tr = create_dataset(y_train.values)
    X_val, y_val_arr = create_dataset(pd.concat([y_train.iloc[-window_size:], y_val]).values)
    X_te, _ = create_dataset(pd.concat([y_val.iloc[-window_size:], y_test]).values)

    tf.keras.utils.set_random_seed(42)
    model = Sequential([LSTM(32, input_shape=(window_size, 1)), Dropout(0.2), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val_arr), epochs=100, batch_size=16, callbacks=[es], verbose=0)
    
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
    ret_stats.columns = ["Ortalama (Mean)", "Standart Sapma (Std)", "Min", "Max", "Çarpıklık (Skewness)"]
    
    adf_res = {col: adfuller(df_returns[col].dropna())[1] for col in df_returns.columns}
    
    train_df, val_df, test_df = strict_data_split(df_returns)
    print("\n=== 2. Benchmark Modeller Eğitiliyor ===")
    split_artifacts = train_and_predict_all_models(train_df, val_df, test_df, return_arimax_details=True)
    y_test_final = split_artifacts["y_test_final"]
    predictions_df = split_artifacts["predictions_df"]
    hybrid_resid = split_artifacts["hybrid_resid"]
    xgb_resid = split_artifacts["xgb_resid"]
    arimax_summary = split_artifacts["arimax_summary"]
    arimax_coef = split_artifacts["arimax_coef"]

    print("\n=== 5. Analiz, Raporlama ve Stres Testi ===")
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
    lb_pval = float(acorr_ljungbox(hybrid_resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0])
    arch_pval = float(het_arch(hybrid_resid.dropna())[1])
    diag_df = pd.DataFrame([{"Test": "Ljung-Box (lag=10)", "p-değeri": lb_pval}, {"Test": "ARCH-LM", "p-değeri": arch_pval}])

    # A) Diebold-Mariano Testi: Hibrit vs diğer modeller
    dm_results_df = run_dm_tests(y_test_final, predictions_df, hybrid_model_name="Hibrit ARIMAX-MLP")
    print("\nDiebold-Mariano Test Sonuçları:")
    print(dm_results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if pd.notna(x) else "NaN"))

    # B) Walk-Forward Cross Validation (5 split)
    print("\n=== 6. Walk-Forward Cross Validation (TimeSeriesSplit=5) ===")
    fold_rmse_df = walk_forward_cross_validation(df_returns, n_splits=5)
    cv_summary_df = summarize_walk_forward_rmse(fold_rmse_df)
    print("\nFold Bazlı RMSE Tablosu:")
    print(fold_rmse_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nAkademik RMSE Özeti (Ortalama ± Std):")
    cv_display_df = cv_summary_df.copy()
    cv_display_df["RMSE Mean ± Std"] = cv_display_df.apply(
        lambda r: f"{r['RMSE Mean']:.6f} ± {r['RMSE Std']:.6f}", axis=1
    )
    print(cv_display_df[["Model", "RMSE Mean ± Std"]].to_string(index=False))

    # C) Bootstrap Confidence Interval (%95) - Hibrit RMSE
    hybrid_name = "Hibrit ARIMAX-MLP"
    point_rmse, ci_low, ci_high, rmse_samples = bootstrap_rmse_confidence_interval(
        y_test_final,
        predictions_df[hybrid_name],
        n_bootstrap=1000,
        confidence_level=0.95,
        random_state=42,
    )
    bootstrap_text = f"RMSE = {point_rmse:.6f}\n95% CI = [{ci_low:.6f}, {ci_high:.6f}]"
    print("\nBootstrap Sonucu:")
    print(bootstrap_text)
    
    # Stres Testi (Gerçek Fiyat üzerinden)
    real_base_price = float(df_raw[TARGET_TICKER].iloc[-1])
    stress_df = run_stress_test(real_base_price)
    
    # D) Görseller
    plot_stress_test_fan_chart(y_test_final.index[-1], real_base_price)
    plot_all_visualizations(y_test_final, predictions_df, metrics_df, hybrid_resid, xgb_resid)
    plot_fold_rmse_comparison(fold_rmse_df)
    plot_bootstrap_rmse_histogram(rmse_samples)
    plot_dm_pvalue_bar(dm_results_df)
    
    # E) Tez raporu entegrasyonu
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
        dm_results_df=dm_results_df,
        fold_rmse_df=fold_rmse_df,
        cv_summary_df=cv_summary_df,
        bootstrap_text=bootstrap_text,
    )

if __name__ == "__main__":
    run_pipeline()
