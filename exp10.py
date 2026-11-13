import os, tempfile, warnings
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, classification_report, accuracy_score

# 1. Load Dataset & Scale Features
df = pd.read_csv("network_events.csv")
features = ["duration", "packet_count", "byte_count", "failed_logins"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# 2. Train K-Means Model (K=2)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
abn_cluster = 1 if df.groupby(labels)["packet_count"].mean()[1] > df.groupby(labels)["packet_count"].mean()[0] else 0
preds = np.where(labels == abn_cluster, 1, 0)

# 3. Model Evaluation
print(f"[+] K-Means Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
print(f"[+] Alignment with Ground Truth: {accuracy_score(df['true_label'], preds) * 100:.2f}%\n")
print(classification_report(df["true_label"], preds, target_names=["Normal", "Abnormal"]))

# 4. Plot & Save Visualization
plt.scatter(df["packet_count"], df["byte_count"], c=preds, cmap="coolwarm", alpha=0.7, edgecolors="k")
plt.xlabel("Packet Count"); plt.ylabel("Byte Count"); plt.title("K-Means Network Anomaly Detection")
plt.tight_layout(); plt.savefig("exp10_kmeans_network_events.png"); plt.close()

# 5. Predict on Test Network Events
samples = pd.DataFrame([
    {"duration": 3.5, "packet_count": 40, "byte_count": 2500, "failed_logins": 0},
    {"duration": 40.0, "packet_count": 1600, "byte_count": 140000, "failed_logins": 15}
])
print("--- Predictions on Live Events ---")
for i, pred in enumerate(kmeans.predict(scaler.transform(samples))):
    print(f"Event #{i+1} -> {'ABNORMAL (Attack)' if pred == abn_cluster else 'NORMAL (Safe)'}")
