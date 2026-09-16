import os, sys, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# 1. Load dataset (generate from DDoS dataset if missing)
path = sys.argv[1] if len(sys.argv) > 1 else "cyber_attacks.csv"
if not os.path.exists(path):
    df_raw = pd.read_csv("DDoS_dataset.csv") if os.path.exists("DDoS_dataset.csv") else pd.DataFrame({'Packet_Count': [50, 180]*50})
    df = pd.DataFrame({'Packet_Count': df_raw['Packet_Count'], 'Flow_Duration': 2.5, 'Label': (df_raw['Packet_Count'] > 100).astype(int)})
    df.to_csv(path, index=False)

df = pd.read_csv(path)
X, y = df.drop(columns=['Timestamp', 'Label'], errors='ignore'), df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Train Ensemble Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Print evaluation results
print("=" * 60)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

# 4. Plot & save confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Normal", "Attack"], cmap="Blues")
plt.title("Cyber Attack Classification (Random Forest)")
plt.tight_layout()
plt.savefig("cyber_attack_classification.png")
print("[+] Plot saved to 'cyber_attack_classification.png'")
