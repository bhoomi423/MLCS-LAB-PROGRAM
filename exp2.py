import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Generate sample dataset if file is not provided
path = sys.argv[1] if len(sys.argv) > 1 else "DDoS_dataset.csv"
if not os.path.exists(path):
    np.random.seed(42)
    time = pd.date_range("2026-01-01 00:00", periods=100, freq="min")
    traffic = 50 + 12 * np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 3, 100)
    traffic[30:35] += 130  # DDoS Spike 1
    traffic[75:78] += 160  # DDoS Spike 2
    pd.DataFrame({"Timestamp": time, "Packet_Count": np.round(traffic)}).to_csv(path, index=False)

# 2. Load & preprocess network traffic data
df = pd.read_csv(path)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
series = df['Packet_Count'].astype(float)

# 3. Fit ARIMA model for network traffic forecasting
model = ARIMA(series, order=(2, 1, 2)).fit()
df['Predicted'] = model.predict(start=0, end=len(series) - 1)
df['Residual'] = series - df['Predicted']

# 4. Anomaly detection for potential DDoS attacks
threshold = df['Residual'].mean() + 2.5 * df['Residual'].std()
attacks = df[df['Residual'] > threshold]

# 5. Output results and evaluation metrics
print("=" * 70)
print("Network Traffic Forecasting and Potential DDoS Attack Detection using ARIMA")
print("=" * 70)
print(f"MAE  : {mean_absolute_error(series[1:], df['Predicted'][1:]):.2f}")
print(f"RMSE : {np.sqrt(mean_squared_error(series[1:], df['Predicted'][1:])):.2f}\n")
print(f"[!] Detected {len(attacks)} potential DDoS attack instances:")
print(attacks[['Timestamp', 'Packet_Count', 'Predicted', 'Residual']].to_string(index=False))

# 6. Plot actual traffic vs ARIMA baseline and detected DDoS spikes
plt.figure(figsize=(10, 5))
plt.plot(df['Timestamp'], series, label="Actual Traffic", color="#1f77b4")
plt.plot(df['Timestamp'], df['Predicted'], label="ARIMA Forecast Baseline", color="#ff7f0e", linestyle="--")
plt.scatter(attacks['Timestamp'], attacks['Packet_Count'], color="red", label="Potential DDoS Attack", zorder=5)
plt.title("Network Traffic Forecasting and Potential DDoS Attack Detection using ARIMA")
plt.xlabel("Timestamp"); plt.ylabel("Packet Count"); plt.legend(); plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig("ddos_forecast_plot.png")
print("\n[+] Plot saved to 'ddos_forecast_plot.png'")
