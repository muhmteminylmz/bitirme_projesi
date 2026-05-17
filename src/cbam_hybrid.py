"""ARIMAX + Ridge Stacking Hybrid Pipeline (Strict 80-10-10 Split)."""

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yfinance as yf
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

plt.rcParams["figure.dpi"] = 300
sns.set_style("whitegrid")

# --- KÜRESEL DEĞİŞKENLER ---
TARGET_TICKER = "EREGL.IS"
CARBON_TICKER = "KEUA"
IRON_TICKER = "TIO=F"
FX_TICKER = "USDTRY=X"

# Renk haritasında Hibrit ismini Ridge olarak güncelledik
MODEL_COLORS = {"Baseline": "#6c7a89", "ARIMAX": "#4c78a8", "XGBoost": "#72b7b2", "LSTM": "#f58518", "Hibrit ARIMAX-Ridge": "#8b0000"}

# ==========================================
# 1. VERİ ÇEKME VE DURAĞANLIK
# ==========================================
def fetch_and_clean_data() -> pd.DataFrame:
    tickers = [TARGET_TICKER, CARBON_TICKER, IRON_TICKER, FX_TICKER]
    raw = yf.download(tickers=tickers, start="2021-04-01", end=None, interval="1d", progress=False)
    df = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw.rename(columns={"Close": TARGET_TICKER})[tickers]
    return df.reindex(columns=tickers).ffill().bfill().dropna()

def get_log_returns(df: pd.DataFrame):
    returns = np.log(df / df.shift(1)).dropna()
    print("\n--- ADF Durağanlık Testleri (Log Getiriler) ---")
    for col in returns.columns:
        p_val = adfuller(returns[col].dropna())[1]
        print(f"{col} - ADF p-değeri: {p_val:.6f} -> {'Durağan' if p_val < 0.05 else 'Durağan Değil'}")
    return returns

# ==========================================
# KESİN %80 - %10 - %10 VERİ BÖLÜNLEMESİ
# ==========================================
def strict_data_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.80)
    val_end = train_end + int(n * 0.10)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"\n--- Veri Seti Bölünmesi (Sıfır Sızıntı) ---")
    print(f"Toplam Veri: {n} gün")
    print(f"Eğitim (Train) Seti : {len(train_df)} gün (%80)")
    print(f"Doğrulama (Val) Seti: {len(val_df)} gün (%10)")
    print(f"Test Seti           : {len(test_df)} gün (%10)")
    
    return train_df, val_df, test_df

# ==========================================
# 2. BENCHMARK MODELLER
# ==========================================
def run_lstm(y_train, y_val, y_test, window_size=20):
    print("LSTM Benchmark Eğitiliyor...")
    def create_dataset(series):
        X, y = [], []
        for i in range(window_size, len(series)):
            X.append(series[i - window_size: i])
            y.append(series[i])
        return np.array(X).reshape(-1, window_size, 1), np.array(y)

    X_train_lstm, y_train_lstm = create_dataset(y_train.values)
    val_bridge = pd.concat([y_train.iloc[-window_size:], y_val])
    X_val_lstm, y_val_lstm = create_dataset(val_bridge.values)
    test_bridge = pd.concat([y_val.iloc[-window_size:], y_test])
    X_test_lstm, _ = create_dataset(test_bridge.values)

    tf.keras.utils.set_random_seed(42)
    model = Sequential([
        LSTM(32, input_shape=(window_size, 1)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    
    model.fit(X_train_lstm, y_train_lstm, validation_data=(X_val_lstm, y_val_lstm), epochs=100, batch_size=16, callbacks=[early_stopping], verbose=0)
    
    preds = model.predict(X_test_lstm, verbose=0).ravel()
    return pd.Series(preds, index=y_test.index)

# ==========================================
# 3. DOĞRUSAL HİBRİT (RIDGE STACKING) MODÜLÜ
# ==========================================
def train_hybrid_ridge(arimax_model, exog_train, exog_test, y_train, y_test):
    print("Hibrit Model (Ridge Meta-Learner) Eğitiliyor...")
    
    # ARIMAX'ın zekası (Tahminleri) çıkarılıyor
    arimax_train_pred = pd.Series(arimax_model.fittedvalues.values, index=y_train.index)
    arimax_test_pred = pd.Series(arimax_model.forecast(steps=len(exog_test), exog=exog_test).values, index=exog_test.index)
    
    # Meta-Learner için Özellik Matrisi: Dışsal Değişkenler + ARIMAX Tahmini
    X_train_meta = exog_train.copy()
    X_train_meta['ARIMAX_Pred'] = arimax_train_pred
    
    X_test_meta = exog_test.copy()
    X_test_meta['ARIMAX_Pred'] = arimax_test_pred
    
    # Standartlaştırma (Sıfır Sızıntı)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_meta)
    X_test_scaled = scaler_X.transform(X_test_meta)
    
    # Ridge Regresyonu: L2 cezası sayesinde Baseline gibi şımarmaz, ağırlıkları kusursuz dağıtır
    # alpha=10.0 ile gürültüyü baskılıyoruz
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train.values)
    
    hybrid_pred = pd.Series(ridge_model.predict(X_test_scaled), index=y_test.index, name="Hibrit ARIMAX-Ridge")
    
    return hybrid_pred, arimax_test_pred

# ==========================================
# 4. ANA ÇALIŞTIRMA BORU HATTI
# ==========================================
def run_pipeline():
    print("=== 1. Veri Hazırlama ===")
    df_raw = fetch_and_clean_data()
    df_returns = get_log_returns(df_raw)
    
    train_df, val_df, test_df = strict_data_split(df_returns)
    
    y_train, y_val, y_test = train_df[TARGET_TICKER], val_df[TARGET_TICKER], test_df[TARGET_TICKER]
    exog_cols = [CARBON_TICKER, IRON_TICKER, FX_TICKER]
    exog_train, exog_val, exog_test = train_df[exog_cols], val_df[exog_cols], test_df[exog_cols]

    print("\n=== 2. Modellerin Eğitimi ===")
    
    # Baseline: Sadece dışsal değişkenleri bilir
    baseline = LinearRegression().fit(exog_train, y_train)
    pred_baseline = pd.Series(baseline.predict(exog_test), index=y_test.index, name="Baseline")
    
    xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
    xgb.fit(exog_train, y_train, eval_set=[(exog_val, y_val)], early_stopping_rounds=20, verbose=False)
    pred_xgb = pd.Series(xgb.predict(exog_test), index=y_test.index, name="XGBoost")
    
    pred_lstm = run_lstm(y_train, y_val, y_test)
    pred_lstm.name = "LSTM"
    
    print("ARIMAX Eğitiliyor...")
    auto_model = auto_arima(y_train, exogenous=exog_train, stepwise=True, trace=False, error_action="ignore")
    arimax_model = SARIMAX(y_train, exog=exog_train, order=auto_model.order).fit(disp=False)
    
    # ŞAH MAT: Ridge Hibrit Modeli (Eğitim verisi üzerinden birleştirme)
    pred_hybrid, pred_arimax = train_hybrid_ridge(arimax_model, exog_train, exog_test, y_train, y_test)
    pred_arimax.name = "ARIMAX"

    print("\n=== 3. Performans Metrikleri (SADECE TEST SETİ) ===")
    predictions = [pred_baseline, pred_xgb, pred_lstm, pred_arimax, pred_hybrid]
    
    metrics = []
    for p in predictions:
        rmse = float(np.sqrt(mean_squared_error(y_test, p)))
        mae = float(mean_absolute_error(y_test, p))
        
        y_true_sign = np.sign(y_test)
        y_pred_sign = np.sign(p)
        da_score = float(np.mean(y_true_sign == y_pred_sign) * 100)
        
        metrics.append({"Model": p.name, "RMSE": rmse, "MAE": mae, "Yön Doğruluğu (%)": da_score})
    
    metrics_df = pd.DataFrame(metrics).sort_values(by="RMSE").reset_index(drop=True)
    print("\nTest Seti Performans Tablosu:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    
    # --- Grafikler ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x="Model", y="RMSE", palette=[MODEL_COLORS.get(x, "#333") for x in metrics_df["Model"]])
    
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.6f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
                    
    plt.title("Şekil 4.1: Modellerin RMSE Karşılaştırması (Ridge Stacking)")
    plt.tight_layout()
    plt.savefig("Grafik_1_Basit_RMSE.png")
    print("\nİşlem Tamamlandı. Grafikler hazır.")

if __name__ == "__main__":
    run_pipeline()