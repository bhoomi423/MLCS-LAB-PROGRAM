import sys
import math
import re
import joblib
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def extract_features(url):
    """Extract key security features from a given URL string."""
    url_str = str(url)
    url_lower = url_str.lower()
    length = len(url_str)
    
    # Suspicious keywords commonly found in phishing links
    keywords = ['login', 'verify', 'update', 'secure', 'account', 'bank', 'admin', 'pay', 'free']
    
    # Calculate character entropy (measure of randomness in URL string)
    counts = Counter(url_str)
    entropy = -sum((c / length) * math.log2(c / length) for c in counts.values()) if length else 0

    return {
        'url_len': length,
        'num_dots': url_str.count('.'),
        'num_hyphens': url_str.count('-'),
        'num_digits': sum(c.isdigit() for c in url_str),
        'has_ip': int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_str))),
        'is_https': int(url_lower.startswith('https')),
        'keyword_count': sum(kw in url_lower for kw in keywords),
        'entropy': entropy
    }

# 1. Load Dataset & Handle CLI Arguments
csv_file = sys.argv[1] if len(sys.argv) > 1 else 'balanced_urls.csv'
df = pd.read_csv(csv_file)
df = df.sample(min(10000, len(df)), random_state=42).reset_index(drop=True)
y = df['result'].astype(int) if 'result' in df.columns else df['label'].map({'benign': 0, 'malicious': 1}).fillna(0).astype(int)

# 2. Extract Features & Train Random Forest Model
print("Extracting features from URLs...")
X = pd.DataFrame([extract_features(url) for url in df['url']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Model Evaluation
y_pred = model.predict(X_test)
print(f"\n[+] Random Forest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Malicious"]))

# Plot & Save Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Malicious'], yticklabels=['Legitimate', 'Malicious'])
plt.title('URL Abnormal Pattern Detection - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('exp6_confusion_matrix.png')
plt.close()

# Save Feature Importances Plot
plt.figure(figsize=(8, 4))
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind='barh', color='skyblue')
plt.title('Feature Importances for Phishing Detection')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('exp6_feature_importance.png')
plt.close()

# Save trained model artifact
joblib.dump(model, 'exp6_rf_model.pkl')
print("[+] Model saved to 'exp6_rf_model.pkl'")
print("[+] Confusion matrix saved to 'exp6_confusion_matrix.png'")
print("[+] Feature importances plot saved to 'exp6_feature_importance.png'")

# 4. Predict on New URLs
new_urls = [
    "https://www.google.com/",
    "http://paypal-login-security-alert.com/verify",
    "http://192.168.1.1/admin/login.html",
    "https://www.amazon.in/gp/cart/view.html"
]

print("\n--- Predictions on Test URLs ---")
new_features = pd.DataFrame([extract_features(url) for url in new_urls])
predictions = model.predict(new_features)

for url, pred in zip(new_urls, predictions):
    label = 'Malicious' if pred == 1 else 'Legitimate'
    print(f"{url:<45} -> [{label}]")



