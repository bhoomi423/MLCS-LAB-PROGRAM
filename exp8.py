import os, tempfile, warnings
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset & Train-Test Split
df = pd.read_csv("email_spoofing.csv")
X, y = df.drop(columns=["is_spoofed"]), df["is_spoofed"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 2. Train Random Forest Model & Evaluate
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
y_pred = model.predict(X_te)
print(f"[+] Email Spoofing Accuracy: {accuracy_score(y_te, y_pred) * 100:.2f}%\n")
print(classification_report(y_te, y_pred, target_names=["Legitimate", "Spoofed"]))

# 3. Plot & Save Confusion Matrix
sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Spoofed"], yticklabels=["Legit", "Spoofed"])
plt.title("Email Spoofing Confusion Matrix"); plt.tight_layout()
plt.savefig("exp8_confusion_matrix.png"); plt.close()

# 4. Predict on Test Headers
samples = pd.DataFrame([
    {"sender_match": 1, "reply_to_match": 1, "spf_pass": 1, "dkim_pass": 1, "dmarc_pass": 1, "suspicious_display_name": 0, "urgent_keywords": 0},
    {"sender_match": 0, "reply_to_match": 0, "spf_pass": 0, "dkim_pass": 0, "dmarc_pass": 0, "suspicious_display_name": 1, "urgent_keywords": 1}
])
print("--- Predictions on Incoming Emails ---")
for i, pred in enumerate(model.predict(samples)):
    print(f"Sample #{i+1} -> {'SPOOFED (Malicious)' if pred == 1 else 'LEGITIMATE (Safe)'}")
