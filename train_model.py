"""
train_model.py
----------------
Trains a fake-news classifier from Fake.csv + True.csv and saves:
  - vectorizer.pkl      (TF-IDF vectorizer)
  - logistic.pkl        (Logistic Regression model)
  - naive_bayes.pkl     (Multinomial Naive Bayes model)
  - random_forest.pkl   (Random Forest model, kept small on purpose)

Run this once locally before deploying:
    python train_model.py
"""

import re
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    """Lowercase, strip URLs, keep only letters/spaces, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data() -> pd.DataFrame:
    fake = pd.read_csv(BASE_DIR / "Fake.csv")
    true = pd.read_csv(BASE_DIR / "True.csv")

    fake["label"] = 0  # Fake
    true["label"] = 1  # Real

    df = pd.concat([fake, true], ignore_index=True)

    # Combine title + text for a richer signal
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna(""))
    df = df[["content", "label"]].dropna()
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows ({(df.label == 0).sum()} fake / {(df.label == 1).sum()} real)")

    print("Cleaning text...")
    df["clean"] = df["content"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )

    print("Fitting TF-IDF vectorizer...")
    # max_features caps vocabulary size -> keeps the vectorizer + models small
    # and fast, which matters a lot on small/free hosting instances.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}

    print("Training Logistic Regression...")
    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(X_train_vec, y_train)
    results["logistic"] = logistic

    print("Training Multinomial Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    results["naive_bayes"] = nb

    print("Training Random Forest (kept small for deployment)...")
    # n_estimators/max_depth are deliberately modest: a big, unbounded
    # random forest produces a huge pickle file (40MB+) and is slow to run
    # LIME explanations against on a low-CPU hosting plan like Render Free.
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train_vec, y_train)
    results["random_forest"] = rf

    print("\nEvaluation on held-out test set:")
    for name, model in results.items():
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {name} (accuracy: {acc:.4f}) ===")
        print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

    print("Saving artifacts...")
    joblib.dump(vectorizer, BASE_DIR / "vectorizer.pkl")
    for name, model in results.items():
        joblib.dump(model, BASE_DIR / f"{name}.pkl")

    for f in ["vectorizer.pkl", "logistic.pkl", "naive_bayes.pkl", "random_forest.pkl"]:
        size_kb = (BASE_DIR / f).stat().st_size / 1024
        print(f"  {f}: {size_kb:.1f} KB")

    print("\nDone. Run 'python app.py' to try the app locally.")


if __name__ == "__main__":
    main()
