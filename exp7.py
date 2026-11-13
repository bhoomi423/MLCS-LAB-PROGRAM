import random, joblib, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Generate Synthetic CAPTCHA Dataset (Noisy Images)
def make_captcha(digit):
    img = Image.new("L", (28, 28), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((random.randint(6, 10), random.randint(4, 8)), str(digit), fill=0, font=ImageFont.load_default())
    for _ in range(10): draw.point((random.randint(0, 27), random.randint(0, 27)), fill=0)
    for _ in range(2): draw.line([(random.randint(0, 27), random.randint(0, 27)), (random.randint(0, 27), random.randint(0, 27))], fill=128)
    return np.array(img).flatten() / 255.0

print("Generating dataset & training CAPTCHA model...")
X = np.array([make_captcha(d) for d in range(10) for _ in range(100)])
y = np.array([d for d in range(10) for _ in range(100)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Train Random Forest Model & Evaluate
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
y_pred = model.predict(X_te)
print(f"\n[+] CAPTCHA Solver Accuracy: {accuracy_score(y_te, y_pred) * 100:.2f}%\n")
print(classification_report(y_te, y_pred))

# 3. Save Confusion Matrix Plot & Model Artifact
sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - CAPTCHA Solver"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
plt.savefig("exp7_confusion_matrix.png"); plt.close()
joblib.dump(model, "exp7_captcha_model.pkl")

# 4. Solve Test CAPTCHA Challenges
print("\n--- Solving CAPTCHA Challenges ---")
for text in ["7391", "4825", "1906", "5204"]:
    solved = "".join(str(model.predict(make_captcha(int(ch)).reshape(1, -1))[0]) for ch in text)
    print(f"CAPTCHA: '{text}' -> Solved: '{solved}'")
