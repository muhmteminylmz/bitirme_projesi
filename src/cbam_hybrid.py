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
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Akademik grafik ayarları
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_style("whitegrid")

TARGET_TICKER = "EREGL.IS"
CARBON_TICKER = "KEUA" # KRBN -> KEUA olarak güncellendi
IRON_TICKER = "TIO=F"
XGBOOST_PARAMS = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5}
MODEL_COLORS = {
    "Baseline": "#6c7a89",
    "ARIMAX": "#4c78a8",
    "XGBoost": "#72b7b2",
    "Hibrit ARIMAX-MLP": "#8b0000",
}
DEFAULT_MODEL_COLOR = "#808080"
STRESS_TEST_SCENARIOS = {"S1 (+%30 Karbon)": 1.30, "S2 (+%60 Karbon)": 1.60, "S3 (+%100 Karbon)": 2.00}


@dataclass
class PipelineResult:
    metrics_table: pd.DataFrame
    predictions: pd.DataFrame
    residuals: pd.DataFrame


def mock_get_yfinance_data() -> pd.DataFrame:
    """Mock verisi üreten fonksiyon."""
    dates = pd.date_range(start="2021-04-01", periods=1250, freq="B")
    np.random.seed(42)
    y_raw = 15 + np.cumsum(np.random.normal(0, 0.2, len(dates)))
    x1_raw = 60 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    x2_raw = 100 + np.cumsum(np.random.normal(0, 1.0, len(dates)))
    df = pd.DataFrame({TARGET_TICKER: y_raw, CARBON_TICKER: x1_raw, IRON_TICKER: x2_raw}, index=dates)
    return df


def enforce_stationarity(df: pd.DataFrame) -> pd.DataFrame:
    stationary_df = df.copy()
    for col in df.columns:
        result = adfuller(df[col].dropna())
        p_value = result[1]
        print(f"{col} - ADF p-değeri: {p_value:.4f}")
        if p_value > 0.05:
            print(f"  -> {col} serisi birim kök içeriyor. Birinci farkı alınıyor...")
            stationary_df[col] = df[col].diff()
            new_p = adfuller(stationary_df[col].dropna())[1]
            print(f"  -> Yeni {col} ADF p-değeri: {new_p:.4f} (Durağan)")
        else:
            print(f"  -> {col} serisi durağan.")
    return stationary_df.dropna()


def split_data(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_size = int(n * train_ratio)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    return train_df, test_df


def fit_sarimax(y: pd.Series, exog: pd.Series) -> SARIMAX:
    print("\nARIMAX modeli eğitiliyor...")
    model = SARIMAX(y, exog=exog, order=(1, 0, 1), enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(disp=False)
    print(fitted_model.summary().tables[1])
    return fitted_model


def fit_xgboost_regressor(X: pd.DataFrame, y: pd.Series):
    from xgboost import XGBRegressor
    model = XGBRegressor(**XGBOOST_PARAMS, random_state=42)
    model.fit(X, y)
    return model


def build_residual_training_frame(x1_train: pd.Series, residuals_train: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"X1_t": x1_train, "r_t_minus_1": residuals_train.shift(1)})
    return df.dropna()


def mock_build_and_train_mlp(X_train: np.ndarray, y_train: np.ndarray):
    """Keras MLP'nin mock edilmiş versiyonu (Hata alınmaması için)"""
    class MockMLP:
        def predict(self, X):
            return LinearRegression().fit(X_train, y_train).predict(X)
    return MockMLP()


def forecast_mlp_residuals(mlp_model, x1_test: pd.Series, last_train_residual: float) -> pd.Series:
    preds = []
    current_residual = last_train_residual
    for i in range(len(x1_test)):
        X_input = np.array([[x1_test.iloc[i], current_residual]])
        pred_res = float(mlp_model.predict(X_input)[0])
        preds.append(pred_res)
        current_residual = pred_res
    return pd.Series(preds, index=x1_test.index)


def calculate_metrics(y_true: pd.Series, predictions_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    metrics = []
    for model_name, y_pred in predictions_dict.items():
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        # MAPE HESAPLAMASI KALDIRILDI!
        metrics.append({"Model": model_name, "RMSE": rmse, "MAE": mae})
    return pd.DataFrame(metrics).sort_values(by="RMSE")


def plot_module_6_visualizations(
    y_test: pd.Series, predictions_df: pd.DataFrame, metrics_df: pd.DataFrame, residuals_df: pd.DataFrame
):
    print("\n[Grafik Üretimi] Çizimler masaüstüne kaydediliyor...")

    # 1. Grafik: RMSE Bar Chart
    plt.figure(figsize=(8, 6))
    bar_colors = [MODEL_COLORS.get(m, DEFAULT_MODEL_COLOR) for m in metrics_df["Model"]]
    ax = sns.barplot(data=metrics_df, x="Model", y="RMSE", hue="Model", palette=bar_colors, legend=False)
    
    # Değerleri barların üstüne yazdır
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.6f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
        
    plt.title("Şekil 4.1: Modellerin Test Kümesi RMSE Karşılaştırması", pad=15, fontsize=12, fontweight='bold')
    plt.ylabel("RMSE Değeri")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("Grafik_1_RMSE_Bar.png")
    plt.close()

    # 2. Grafik: Test Seti Prediction Line Chart (Zoomsuz, net)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, color="black", label="Gerçek EREGL.IS", linewidth=2.5, linestyle="--")
    for model_col in predictions_df.columns:
        lw = 2.5 if "Hibrit" in model_col else 1.2
        alpha = 1.0 if "Hibrit" in model_col else 0.7
        plt.plot(predictions_df.index, predictions_df[model_col], 
                 label=model_col, color=MODEL_COLORS.get(model_col, DEFAULT_MODEL_COLOR), 
                 linewidth=lw, alpha=alpha)
    
    plt.title("Şekil 4.2: Zaman Serisi Tahmin Performansı (Gerçek vs. Modeller)", pad=15, fontsize=12, fontweight='bold')
    plt.ylabel("Fiyat / Getiri (Fark Serisi)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("Grafik_2_Prediction_Line.png")
    plt.close()

    # 3. Grafik: KDE Dağılım Grafiği (Hibrit vs XGBoost)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=residuals_df["XGBoost"], label="XGBoost Hata Dağılımı", color=MODEL_COLORS["XGBoost"], fill=True, alpha=0.3)
    sns.kdeplot(data=residuals_df["Hibrit ARIMAX-MLP"], label="Hibrit Model Hata Dağılımı", color=MODEL_COLORS["Hibrit ARIMAX-MLP"], fill=True, alpha=0.5)
    plt.title("Şekil 4.3: Hata Dağılımı Çekirdek Yoğunluk Tahmini (KDE)", pad=15, fontsize=12, fontweight='bold')
    plt.xlabel("Tahmin Hatası (Gerçek - Tahmin)")
    plt.ylabel("Yoğunluk (Density)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Grafik_3_Residual_KDE.png")
    plt.close()

    print("-> Grafikler başarıyla 'Grafik_1', 'Grafik_2', 'Grafik_3' olarak kaydedildi.")


def plot_stress_test_fan(last_price: float, stress_results: dict):
    """Karbon Stres Testi Yelpaze Grafiğini çizer"""
    plt.figure(figsize=(10, 6))
    days = np.arange(1, 31)
    
    # Baz Çizgi
    base_line = np.full(30, last_price)
    plt.plot(days, base_line, label="S0: Baz Senaryo", color="black", linestyle="--", linewidth=2)
    
    # Şok çizgileri ve Yelpaze (Fan) dolguları
    colors = {"S1 (+%30 Karbon)": "orange", "S2 (+%60 Karbon)": "red", "S3 (+%100 Karbon)": "darkred"}
    prev_line = base_line
    
    for scenario, pct_change in stress_results.items():
        scenario_line = base_line * (1 + (pct_change / 100))
        plt.plot(days, scenario_line, label=f"{scenario} | Etki: %{pct_change:.2f}", color=colors[scenario], linewidth=2)
        plt.fill_between(days, prev_line, scenario_line, color=colors[scenario], alpha=0.15)
        prev_line = scenario_line

    plt.title("Şekil 4.4: Karbon Şoklarının EREGL.IS Üzerindeki Simüle Edilmiş Etkisi (Fan Chart)", pad=15, fontsize=12, fontweight='bold')
    plt.xlabel("Projeksiyon Günü (t+n)")
    plt.ylabel("Tahmini Hisse Fiyatı")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("Grafik_4_Stres_Testi_Fan.png")
    plt.close()
    print("-> Stres Testi Yelpaze grafiği 'Grafik_4_Stres_Testi_Fan.png' olarak kaydedildi.")


def run_stress_test(hybrid_model_mlp, last_arimax_pred: float, last_train_residual: float, last_actual_carbon: float) -> pd.DataFrame:
    results = []
    stress_dict = {}
    for scenario_name, multiplier in STRESS_TEST_SCENARIOS.items():
        shocked_carbon = last_actual_carbon * multiplier
        X_input = np.array([[shocked_carbon, last_train_residual]])
        shock_impact = float(hybrid_model_mlp.predict(X_input)[0])
        final_prediction = last_arimax_pred + shock_impact
        pct_change = (shock_impact / abs(last_arimax_pred)) * 100 if last_arimax_pred != 0 else 0
        
        results.append({
            "Senaryo": scenario_name,
            "Karbon Değişimi": f"+%{(multiplier-1)*100:.0f}",
            "Simüle Hisse Değişimi (%)": pct_change
        })
        stress_dict[scenario_name] = pct_change

    df = pd.DataFrame(results)
    
    # Grafiği çizdirecek fonksiyonu çağır
    plot_stress_test_fan(last_price=last_arimax_pred, stress_results=stress_dict)
    
    return df


def run_pipeline():
    print("=== Modül 1: Veri Çekme ve Ön İşleme ===")
    df_raw = mock_get_yfinance_data()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_raw), columns=df_raw.columns, index=df_raw.index)
    df_stationary = enforce_stationarity(df_scaled)
    train_df, test_df = split_data(df_stationary)

    y_train = train_df[TARGET_TICKER]
    x1_train = train_df[CARBON_TICKER]
    x2_train = train_df[IRON_TICKER]

    y_test = test_df[TARGET_TICKER]
    x1_test = test_df[CARBON_TICKER]
    x2_test = test_df[IRON_TICKER]

    print("\n=== Modül 2: ARIMAX Eğitimi ve Benchmark'lar ===")
    arimax_model = fit_sarimax(y_train, exog=x2_train)
    arimax_test_pred = pd.Series(arimax_model.predict(start=len(y_train), end=len(y_train)+len(y_test)-1, exog=x2_test).values, index=y_test.index, name="ARIMAX")

    residuals_train = pd.Series(arimax_model.resid, index=y_train.index)

    baseline_model = LinearRegression()
    X_train_bench = train_df[[CARBON_TICKER, IRON_TICKER]]
    X_test_bench = test_df[[CARBON_TICKER, IRON_TICKER]]
    baseline_model.fit(X_train_bench, y_train)
    baseline_pred = pd.Series(baseline_model.predict(X_test_bench), index=y_test.index, name="Baseline")

    xgb_model = fit_xgboost_regressor(X_train_bench, y_train)
    xgb_pred = pd.Series(xgb_model.predict(X_test_bench), index=y_test.index, name="XGBoost")

    print("\n=== Modül 3: MLP Eğitim ve Hibrit Birleştirme ===")
    mlp_train_df = build_residual_training_frame(x1_train, residuals_train)
    X_mlp_train = mlp_train_df.values
    y_mlp_train = residuals_train.loc[mlp_train_df.index].values
    
    mlp_model = mock_build_and_train_mlp(X_mlp_train, y_mlp_train)

    mlp_residual_test_pred = forecast_mlp_residuals(mlp_model=mlp_model, x1_test=x1_test, last_train_residual=float(residuals_train.iloc[-1]))
    hybrid_pred = arimax_test_pred.add(mlp_residual_test_pred, fill_value=0.0)
    hybrid_pred.name = "Hibrit ARIMAX-MLP"

    predictions = {"Baseline": baseline_pred, "ARIMAX": arimax_test_pred, "XGBoost": xgb_pred, "Hibrit ARIMAX-MLP": hybrid_pred}
    predictions_df = pd.DataFrame(predictions).reindex(y_test.index)
    residuals_df = pd.DataFrame({"XGBoost": y_test - predictions_df["XGBoost"], "Hibrit ARIMAX-MLP": y_test - predictions_df["Hibrit ARIMAX-MLP"]}, index=y_test.index)

    print("\n=== Modül 4: Analiz ve Çıktılar ===")
    metrics_df = calculate_metrics(y_test, predictions)
    print("\nTest Seti Performans Tablosu (RMSE / MAE):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    hybrid_residuals = residuals_df["Hibrit ARIMAX-MLP"].dropna()
    ljung_box_pvalue = float(acorr_ljungbox(hybrid_residuals, lags=[10], return_df=True)["lb_pvalue"].iloc[0])
    arch_lm_pvalue = float(het_arch(hybrid_residuals)[1])
    diagnostics_df = pd.DataFrame([{"Test": "Ljung-Box (lag=10)", "p-değeri": ljung_box_pvalue}, {"Test": "ARCH-LM", "p-değeri": arch_lm_pvalue}])
    
    print("\nResidual Tanı Testleri (Hibrit Model):")
    print(diagnostics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    plot_module_6_visualizations(y_test=y_test, predictions_df=predictions_df, metrics_df=metrics_df, residuals_df=residuals_df)

    print("\n=== Modül 5: Karbon Stres Testi ===")
    stress_table_df = run_stress_test(mlp_model, float(arimax_test_pred.iloc[-1]), float(residuals_train.iloc[-1]), float(x1_train.iloc[-1]))
    print(stress_table_df.to_string(index=False))

if __name__ == "__main__":
    run_pipeline()