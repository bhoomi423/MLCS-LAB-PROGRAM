import os, tempfile, warnings
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset & Vectorize Text
df = pd.read_csv("email_spam.csv")
cv = CountVectorizer(stop_words="english", lowercase=True)
X = cv.fit_transform(df["text"])
y = df["label"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 2. Train Naive Bayes Classifier & Evaluate
model = MultinomialNB().fit(X_tr, y_tr)
y_pred = model.predict(X_te)
print(f"[+] Naive Bayes Spam Accuracy: {accuracy_score(y_te, y_pred) * 100:.2f}%\n")
print(classification_report(y_te, y_pred, target_names=["Ham", "Spam"]))

# 3. Plot & Save Confusion Matrix
sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d", cmap="Reds",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Email Spam Confusion Matrix"); plt.tight_layout()
plt.savefig("exp9_confusion_matrix.png"); plt.close()

# 4. Predict on Test Emails
test_emails = [
    "Hey, please send me the MLCS lab report by tomorrow morning.",
    "Congratulations! You won a $1000 cash prize, click here to claim now!",
    "Let's meet in the conference hall for the project discussion."
]
print("--- Predictions on Test Emails ---")
for text, pred in zip(test_emails, model.predict(cv.transform(test_emails))):
    print(f"[{'SPAM' if pred == 1 else 'HAM'}] -> \"{text}\"")
