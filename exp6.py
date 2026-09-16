import math, re, joblib
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def extract_features(url):
    u = str(url)
    p = urlparse(u if u.startswith(('http://', 'https://')) else 'http://' + u)
    h = p.netloc or p.path.split('/')[0]
    l = len(u)
    counts = Counter(u)
    entropy = -sum((c / l) * math.log2(c / l) for c in counts.values()) if l else 0
    kws = ['login', 'verify', 'update', 'secure', 'account', 'bank', 'admin', 'pay', 'free']
    return {
        'url_len': l, 'hostname_len': len(h), 'num_dots': u.count('.'),
        'num_hyphens': u.count('-'), 'num_digits': sum(c.isdigit() for c in u),
        'has_ip': int(bool(re.search(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', h))),
        'is_https': int(u.startswith('https')),
        'keyword_count': sum(kw in u.lower() for kw in kws),
        'entropy': entropy
    }

# 1. Load Dataset & Standardize Labels
df = pd.read_csv('balanced_urls.csv')
df = df.sample(min(10000, len(df)), random_state=42).reset_index(drop=True)
y = df['result'] if 'result' in df.columns else df['label'].map({'benign': 0, 'malicious': 1})

# 2. Extract Features & Train Model
print("Extracting features from URLs...")
X = pd.DataFrame([extract_features(u) for u in df['url']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# 3. Model Evaluation
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))

# Plot & Save Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
plt.title('URL Abnormal Pattern Detection')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout()
plt.savefig('exp6_confusion_matrix.png'); plt.close()
joblib.dump(model, 'exp6_rf_model.pkl')

# 4. Predict on New URLs
new_urls = [
    "https://www.google.com/",
    "http://paypal-login-security-alert.com/verify",
    "http://192.168.1.1/admin/login.html"
]
predictions = model.predict(pd.DataFrame([extract_features(u) for u in new_urls]))
print("\n--- Predictions on New URLs ---")
for url, pred in zip(new_urls, predictions):
    print(f"{url} -> {'Abnormal' if pred == 1 else 'Normal'}")

