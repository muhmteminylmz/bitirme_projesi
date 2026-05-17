# =========================================================
# EREGL Forecasting Project - FULL DATA INSPECTION PIPELINE
# =========================================================
# Amaç:
# Veriyi modele başlamadan önce detaylı incelemek.
#
# İçerik:
# 1. Veri indirme
# 2. Temizleme
# 3. Missing value analizi
# 4. Descriptive statistics
# 5. Correlation
# 6. Lag correlation
# 7. Stationarity (ADF)
# 8. Volatility
# 9. Outlier kontrolü
# 10. Granger causality
# 11. ACF/PACF
# 12. Sana uygun özet çıktı üretimi
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# =========================================================
# TICKERS
# =========================================================

TARGET_TICKER = "EREGL.IS"
CARBON_TICKER = "KEUA"
IRON_TICKER = "TIO=F"
FX_TICKER = "USDTRY=X"

START = "2021-04-01"
END = "2026-05-30"
INTERVAL = "1d"

# =========================================================
# DOWNLOAD DATA
# =========================================================

tickers = {
    "EREGL": TARGET_TICKER,
    "CARBON": CARBON_TICKER,
    "IRON": IRON_TICKER,
    "USDTRY": FX_TICKER,
}

all_data = {}

for name, ticker in tickers.items():

    print(f"\nDownloading {ticker} ...")

    temp = yf.download(
        ticker,
        start=START,
        end=END,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False
    )

    # MultiIndex flatten
    temp.columns = [col[0] if isinstance(col, tuple) else col for col in temp.columns]

    temp = temp[["Open", "High", "Low", "Close", "Volume"]]

    temp.columns = [f"{name}_{col}" for col in temp.columns]

    all_data[name] = temp

# =========================================================
# MERGE
# =========================================================

df = pd.concat(all_data.values(), axis=1)

print("\n======================")
print("RAW SHAPE")
print("======================")
print(df.shape)

# =========================================================
# MISSING VALUES
# =========================================================

print("\n======================")
print("MISSING VALUES")
print("======================")
print(df.isna().sum())

missing_ratio = (df.isna().sum() / len(df)) * 100

print("\nMissing Ratio (%)")
print(missing_ratio)

# forward fill
df = df.ffill()

# =========================================================
# BASIC INFO
# =========================================================

print("\n======================")
print("DATA INFO")
print("======================")
print(df.info())

# =========================================================
# CREATE RETURNS
# =========================================================

close_cols = [c for c in df.columns if "Close" in c]

returns = pd.DataFrame(index=df.index)

for col in close_cols:
    returns[col + "_RET"] = np.log(df[col] / df[col].shift(1))

returns = returns.dropna()

# =========================================================
# DESCRIPTIVE STATISTICS
# =========================================================

print("\n======================")
print("DESCRIPTIVE STATS")
print("======================")

stats = returns.describe().T

stats["skew"] = returns.skew()
stats["kurtosis"] = returns.kurtosis()

print(stats)

# =========================================================
# PRICE PLOTS
# =========================================================

plt.figure(figsize=(16,8))

for col in close_cols:
    plt.plot(df.index, df[col], label=col)

plt.title("Close Prices")
plt.legend()
plt.show()

# =========================================================
# RETURN PLOTS
# =========================================================

plt.figure(figsize=(16,8))

for col in returns.columns:
    plt.plot(returns.index, returns[col], label=col)

plt.title("Log Returns")
plt.legend()
plt.show()

# =========================================================
# CORRELATION MATRIX
# =========================================================

corr = returns.corr()

print("\n======================")
print("CORRELATION MATRIX")
print("======================")
print(corr)

print(returns["EREGL_Close_RET"].autocorr(lag=1))

from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(
    returns["EREGL_Close_RET"],
    lags=[5,10,20],
    return_df=True
)

print(lb)

plt.figure(figsize=(10,8))


sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Return Correlation Matrix")
plt.show()

# =========================================================
# ROLLING VOLATILITY
# =========================================================

rolling_vol = returns.rolling(20).std()

plt.figure(figsize=(16,8))

for col in rolling_vol.columns:
    plt.plot(rolling_vol.index, rolling_vol[col], label=col)

plt.title("20-Day Rolling Volatility")
plt.legend()
plt.show()

# =========================================================
# ADF TEST
# =========================================================

print("\n======================")
print("ADF TEST")
print("======================")

def adf_test(series, name):

    result = adfuller(series.dropna())

    print(f"\n{name}")
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"P-value       : {result[1]:.6f}")

    if result[1] < 0.05:
        print("Stationary ✅")
    else:
        print("Not Stationary ❌")

for col in returns.columns:
    adf_test(returns[col], col)

# =========================================================
# OUTLIER CHECK
# =========================================================

print("\n======================")
print("OUTLIER CHECK")
print("======================")

z_scores = (returns - returns.mean()) / returns.std()

outliers = (np.abs(z_scores) > 3).sum()

print(outliers)

# =========================================================
# LAG CORRELATION
# =========================================================

print("\n======================")
print("LAG CORRELATION")
print("======================")

target = "EREGL_Close_RET"

for feature in [
    "CARBON_Close_RET",
    "IRON_Close_RET",
    "USDTRY_Close_RET"
]:

    print(f"\n{feature} vs {target}")

    lag_corrs = {}

    for lag in range(1, 15):

        corr = returns[target].corr(
            returns[feature].shift(lag)
        )

        lag_corrs[lag] = corr

    lag_df = pd.DataFrame({
        "Lag": lag_corrs.keys(),
        "Correlation": lag_corrs.values()
    })

    print(lag_df)

# =========================================================
# GRANGER CAUSALITY
# =========================================================

print("\n======================")
print("GRANGER CAUSALITY")
print("======================")

target = returns["EREGL_Close_RET"]

for feature in [
    "CARBON_Close_RET",
    "IRON_Close_RET",
    "USDTRY_Close_RET"
]:

    print(f"\nTesting: {feature} -> EREGL")

    test_df = returns[
        [target.name, feature]
    ].dropna()

    try:

        results = grangercausalitytests(
            test_df,
            maxlag=5,
            verbose=False
        )

        for lag in range(1, 6):

            p_value = results[lag][0]["ssr_ftest"][1]

            print(f"Lag {lag} p-value: {p_value:.6f}")

    except:
        print("Granger test failed.")

# =========================================================
# ACF & PACF
# =========================================================

target_series = returns["EREGL_Close_RET"].dropna()

fig, ax = plt.subplots(1, 2, figsize=(16,5))

plot_acf(target_series, lags=40, ax=ax[0])
plot_pacf(target_series, lags=40, ax=ax[1])

ax[0].set_title("ACF")
ax[1].set_title("PACF")

plt.show()

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n===================================================")
print("FINAL SUMMARY")
print("===================================================")

print(f"""
1. Dataset Shape:
{df.shape}

2. Date Range:
{df.index.min()} --> {df.index.max()}

3. Variables:
{list(df.columns)}

4. Return Variables:
{list(returns.columns)}

5. Most Correlated With EREGL:
{corr["EREGL_Close_RET"].sort_values(ascending=False)}

6. Stationarity:
ADF p-values mostly < 0.05 ise returns stationary.

7. Things To Look For:
- Strong lag correlations?
- Significant granger causality?
- Volatility clustering?
- Heavy outliers?
- ACF decay structure?

8. Model Strategy Suggestions:
- ARIMA -> if strong autocorrelation
- ARIMAX -> if exogenous vars significant
- XGBoost -> nonlinear lag effects
- LSTM -> temporal nonlinear patterns
- Hybrid -> residual structures remain
""")

print("\nDONE ✅")