import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Load or generate time series dataset
path = sys.argv[1] if len(sys.argv) > 1 else "DDoS_dataset.csv"
if not os.path.exists(path):
    np.random.seed(42)
    time = pd.date_range("2026-01-01 00:00", periods=100, freq="min")
    traffic = 50 + 12 * np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 3, 100)
    traffic[30:35] += 130  # DDoS Spike 1
    traffic[75:78] += 160  # DDoS Spike 2
    pd.DataFrame({"Timestamp": time, "Packet_Count": np.round(traffic)}).to_csv(path, index=False)

# 2. Load dataset and set datetime index
df = pd.read_csv(path)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# 3. Perform Time Series Decomposition (Observed, Trend, Seasonal, Residual)
result = seasonal_decompose(df['Packet_Count'], model='additive', period=20)

# 4. Output summary components
print("=" * 65)
print("Experiment 1: Time Series Decomposition (Trend & Seasonal Components)")
print("=" * 65)
print(f"Total Data Points  : {len(df)}")
print(f"Average Packet Count: {df['Packet_Count'].mean():.2f}")
print(f"Seasonal Variation  : [{result.seasonal.min():.2f}, {result.seasonal.max():.2f}]")
print(f"Residual Std Dev    : {result.resid.dropna().std():.2f}\n")

# 5. Plot and save decomposition components
fig = result.plot()
fig.set_size_inches(10, 6)
plt.suptitle("Time Series Decomposition: Trend, Seasonal & Residual Components", fontsize=11)
plt.tight_layout()
plt.savefig("time_series_decomposition.png")
print("[+] Decomposition plot saved to 'time_series_decomposition.png'")
