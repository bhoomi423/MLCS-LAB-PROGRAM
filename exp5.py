import sys, joblib, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset & Preprocess
csv_file = sys.argv[1] if len(sys.argv) > 1 else "balanced_urls.csv"
df = pd.read_csv(csv_file)
df = df.sample(min(10000, len(df)), random_state=42).reset_index(drop=True)
y = df["result"].astype(int) if "result" in df.columns else df["label"].map({"benign": 0, "malicious": 1}).fillna(0).astype(int)

# 2. TF-IDF Vectorization & Train-Test Split
print("Vectorizing URLs & training SVM model...")
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000, sublinear_tf=True)
X_vec = vectorizer.fit_transform(df["url"])
X_tr, X_te, y_tr, y_te = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

# 3. Model Training & Evaluation
model = SVC(kernel="linear", C=1.0, random_state=42).fit(X_tr, y_tr)
y_pred = model.predict(X_te)
print(f"\n[+] SVM Accuracy: {accuracy_score(y_te, y_pred) * 100:.2f}%\n")
print(classification_report(y_te, y_pred, target_names=["Legitimate", "Malicious"]))

# 4. Plot & Save Artifacts
sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Malicious"], yticklabels=["Legitimate", "Malicious"])
plt.title("Confusion Matrix - SVM"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
plt.savefig("exp5_svm_confusion_matrix.png"); plt.close()

joblib.dump(model, "exp5_svm_model.pkl")
joblib.dump(vectorizer, "exp5_svm_vectorizer.pkl")

# 5. Predict on Unseen URLs
test_urls = [
    "https://www.google.com",
    "http://paypal-login-security-alert.com",
    "https://www.amazon.in/gp/cart/view.html",
    "http://verify-account-amazon.biz/login"
]
preds = model.predict(vectorizer.transform(test_urls))
print("\nPredictions on Test URLs:")
for url, pred in zip(test_urls, preds):
    print(f"{url:<45} -> [{'Malicious' if pred == 1 else 'Legitimate'}]")

