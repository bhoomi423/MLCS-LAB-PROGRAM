import os, tempfile, warnings
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Feature Extraction for SQL Queries
def get_features(q):
    s = str(q).upper()
    return [len(s), s.count("'") + s.count('"'), sum(s.count(c) for c in [";", "-", "/", "*", "="]),
            int(any(kw in s for kw in ["OR", "UNION", "SELECT", "DROP", "--"]))]

# 2. Load Dataset & Prepare Features
df = pd.read_csv("sql_injection_dataset.csv")
X = pd.DataFrame([get_features(q) for q in df["query"]], columns=["len", "quotes", "specials", "has_kw"])
y = df["label"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 3. Train Decision Tree & Evaluate
model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
y_pred = model.predict(X_te)
print(f"[+] Decision Tree SQLi Accuracy: {accuracy_score(y_te, y_pred) * 100:.2f}%\n")
print(classification_report(y_te, y_pred, target_names=["Legitimate", "Malicious"]))

# 4. Save Confusion Matrix & Tree Diagram
sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d", cmap="Purples",
            xticklabels=["Legit", "SQLi"], yticklabels=["Legit", "SQLi"])
plt.title("SQL Injection Confusion Matrix"); plt.tight_layout()
plt.savefig("exp11_confusion_matrix.png"); plt.close()

plot_tree(model, feature_names=X.columns, class_names=["Legit", "SQLi"], filled=True, rounded=True)
plt.title("Decision Tree for SQL Injection Detection"); plt.tight_layout()
plt.savefig("exp11_decision_tree.png"); plt.close()

# 5. Predict on Test Queries
test_queries = ["SELECT name, email FROM users WHERE id = 10;", "' OR '1'='1' --", "1; DROP TABLE students; --"]
print("--- Predictions on Database Queries ---")
for q, pred in zip(test_queries, model.predict(pd.DataFrame([get_features(q) for q in test_queries], columns=X.columns))):
    print(f"[{'MALICIOUS (SQLi)' if pred == 1 else 'LEGITIMATE'}] -> \"{q}\"")
