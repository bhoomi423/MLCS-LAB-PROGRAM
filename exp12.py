import os, tempfile, warnings
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset & Encode Context Features
df = pd.read_csv("wsn_context_events.csv")
le_ip, le_type = LabelEncoder(), LabelEncoder()
df["source_ip"] = le_ip.fit_transform(df["source_ip"])
df["event_type"] = le_type.fit_transform(df["event_type"])

features = ["timestamp_hour", "source_ip", "event_type", "packet_rate", "rssi", "energy_drain"]
X, y = df[features], df["is_malicious"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 2. Train Random Forest Model & Evaluate
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
y_pred = model.predict(X_te)
print(f"[+] WSN Context Anomaly Accuracy: {accuracy_score(y_te, y_pred) * 100:.2f}%\n")
print(classification_report(y_te, y_pred, target_names=["Normal", "Malicious"]))

# 3. Save Confusion Matrix
sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=["Normal", "Malicious"], yticklabels=["Normal", "Malicious"])
plt.title("WSN Context Anomaly Confusion Matrix"); plt.tight_layout()
plt.savefig("exp12_confusion_matrix.png"); plt.close()

# 4. Predict on Test WSN Context Scenarios
scenarios = pd.DataFrame([
    {"timestamp_hour": 14, "source_ip": le_ip.transform(["192.168.1.10"])[0], "event_type": le_type.transform(["DATA_REPORT"])[0], "packet_rate": 3.5, "rssi": -60.0, "energy_drain": 0.5},
    {"timestamp_hour": 3, "source_ip": le_ip.transform(["192.168.1.99"])[0], "event_type": le_type.transform(["ROUTE_FLOOD"])[0], "packet_rate": 120.0, "rssi": -95.0, "energy_drain": 11.2}
])
print("--- Predictions on WSN Context Events ---")
for i, pred in enumerate(model.predict(scenarios)):
    print(f"Scenario #{i+1} -> {'MALICIOUS (Attack)' if pred == 1 else 'NORMAL (Safe)'}")
