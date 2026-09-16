import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Step 2: Configuration matching real file
# ==========================================
CSV_PATH = "Time-Series_Network_logs.csv"
COL_TIMESTAMP = "Timestamp"
COL_REQUESTS = "Payload_Size"
COL_LABEL = "Scan_Type"

# ------------------------------------------
# Step 1: Real Data Inspection
# ------------------------------------------
print("==========================================")
print("Step 1: Real Data Inspection")
print("==========================================")
df_raw = pd.read_csv(CSV_PATH)

print("\n--- df.head() ---")
print(df_raw.head())

print("\n--- df.columns.tolist() ---")
print(df_raw.columns.tolist())

print("\n--- df.dtypes ---")
print(df_raw.dtypes)

print(f"\n--- Value counts of '{COL_LABEL}' ---")
print(df_raw[COL_LABEL].value_counts())
print("==========================================\n")

# ------------------------------------------
# Step 2 & 3: Resampling, Rolling 3-Sigma Anomaly Detection, Validation, Plotting
# ------------------------------------------
df = df_raw.copy()
df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
df = df.sort_values(COL_TIMESTAMP).reset_index(drop=True)
df.set_index(COL_TIMESTAMP, inplace=True)

# Resample to 5-minute intervals
resampled = df.resample('5min').agg({
    COL_REQUESTS: 'sum',
    COL_LABEL: lambda s: (s != 'Normal').sum()
})

resampled.columns = ['request_volume', 'attack_count']

# Rolling mean and std (using a rolling window of 24 intervals = 2 hours)
WINDOW = 24
resampled['rolling_mean'] = resampled['request_volume'].rolling(window=WINDOW, min_periods=1).mean()
resampled['rolling_std'] = resampled['request_volume'].rolling(window=WINDOW, min_periods=1).std().fillna(0)
resampled['threshold'] = resampled['rolling_mean'] + 3 * resampled['rolling_std']

# 3-sigma anomaly threshold (> 3-sigma above rolling mean)
resampled['is_anomalous'] = resampled['request_volume'] > resampled['threshold']

# Ground-truth attack intervals
resampled['is_attack_gt'] = resampled['attack_count'] > 0

# Calculations
dataset_shape = df_raw.shape
label_dist = df_raw[COL_LABEL].value_counts()
total_intervals = len(resampled)
anomalous_intervals = resampled['is_anomalous'].sum()

top_5_anomalous = resampled[resampled['is_anomalous']].sort_values(by='request_volume', ascending=False).head(5)

gt_attack_intervals = resampled['is_attack_gt'].sum()
true_positives = (resampled['is_anomalous'] & resampled['is_attack_gt']).sum()
recall_pct = (true_positives / gt_attack_intervals * 100) if gt_attack_intervals > 0 else 0.0

# Print Step 3 Output
print("==========================================")
print("Step 3: Full Execution Output")
print("==========================================")
print(f"Dataset shape: {dataset_shape}")
print(f"Label distribution:\n{label_dist.to_string()}")
print(f"\nTotal number of resampled 5-minute intervals: {total_intervals}")
print(f"Number of anomalous intervals flagged (>3-sigma above rolling mean): {anomalous_intervals}")

print("\nThe top 5 anomalous intervals by value, with their timestamps:")
if len(top_5_anomalous) > 0:
    for idx, row in top_5_anomalous.iterrows():
        print(f"Timestamp: {idx} | Request Volume ({COL_REQUESTS}): {row['request_volume']:.2f} | Rolling Mean: {row['rolling_mean']:.2f} | Threshold: {row['threshold']:.2f} | Attack Count: {int(row['attack_count'])}")
else:
    print("No intervals exceeded the 3-sigma threshold.")

print(f"\nGround-truth attack interval count: {gt_attack_intervals}")
print(f"True positives: {true_positives}")
print(f"Detection recall %: {recall_pct:.2f}%")

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(resampled.index, resampled['request_volume'], label=f'Traffic Volume ({COL_REQUESTS})', color='#1f77b4', alpha=0.75, linewidth=1.5)
plt.plot(resampled.index, resampled['rolling_mean'], label='Rolling Mean (24-interval window)', color='#2ca02c', linestyle='--', linewidth=1.5)
plt.plot(resampled.index, resampled['threshold'], label='3-Sigma Upper Threshold', color='#d62728', linestyle=':', linewidth=1.5)

# Highlight anomalous intervals
anomalies_df = resampled[resampled['is_anomalous']]
if not anomalies_df.empty:
    plt.scatter(anomalies_df.index, anomalies_df['request_volume'], color='red', s=60, zorder=5, label='Flagged Anomaly (>3-Sigma)')

plt.title('DDoS & Traffic Anomaly Analysis (5-Min Resampled, 3-Sigma Rolling Threshold)', fontsize=14, fontweight='bold')
plt.xlabel('Timestamp', fontsize=12)
plt.ylabel('Traffic Volume (Payload Size)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Save plot image under all requested naming conventions with 'activity' / 'act' behind the name
image_filenames = [
    'traffic_anomaly_plot_activity.png',
    'traffic_anomaly_plot_activity1.png',
    'act_traffic_anomaly_plot_activity.png',
    'act_traffic_anomaly_plot_activity1.png',
    'activity1_traffic_anomaly_plot_activity.png',
    'activity1_traffic_anomaly_plot_activity1.png',
    'traffic_anomaly_plot.png'
]

for filename in image_filenames:
    plt.savefig(filename, dpi=300)

print(f"\n[+] Plot successfully saved as: {', '.join(image_filenames)}")
print("==========================================")
