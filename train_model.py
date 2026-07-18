import pandas as pd
import re
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
fake_path = BASE_DIR / "Fake.csv"
true_path = BASE_DIR / "True.csv"

if not fake_path.exists() or not true_path.exists():
    raise FileNotFoundError("Please put Fake.csv and True.csv in the project folder.")

df_fake = pd.read_csv(fake_path)
df_true = pd.read_csv(true_path)

df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

if "title" in df.columns and "text" in df.columns:
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
elif "text" in df.columns:
    df["content"] = df["text"].fillna("")
else:
    df["content"] = ""

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["content"] = df["content"].apply(clean_text)
df = df[df["content"].str.len() > 20]

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "naive_bayes": MultinomialNB()
}

scores = {}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, pred)
    scores[name] = acc
    print(f"\n{name} Accuracy: {acc}")
    print(classification_report(y_test, pred))

joblib.dump(vectorizer, BASE_DIR / "vectorizer.pkl")
for name, model in models.items():
    joblib.dump(model, BASE_DIR / f"{name}.pkl")

print("\nSaved vectorizer and all models.")
print("Scores:", scores)