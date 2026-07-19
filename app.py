from flask import Flask, render_template, request
import joblib
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from lime.lime_text import LimeTextExplainer

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model artifacts once at startup (not per-request)
# ---------------------------------------------------------------------------
vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")
models = {
    "logistic": joblib.load(BASE_DIR / "logistic.pkl"),
    "random_forest": joblib.load(BASE_DIR / "random_forest.pkl"),
    "naive_bayes": joblib.load(BASE_DIR / "naive_bayes.pkl"),
}

MODEL_LABELS = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "naive_bayes": "Naive Bayes",
}

explainer = LimeTextExplainer(class_names=["Fake", "Real"])

# LIME's default is 5000 perturbed samples per explanation, which is far too
# slow/heavy for a free-tier server with a fraction of a CPU core. 300 is
# enough to produce a stable, useful explanation while staying fast.
LIME_NUM_SAMPLES = 300
LIME_NUM_FEATURES = 8


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    return text.strip()


def predict_proba_for_lime(model, texts):
    X = vectorizer.transform(texts)
    return model.predict_proba(X)


@app.route("/")
def home():
    return render_template("index.html", model_labels=MODEL_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "").strip()
    news_url = request.form.get("news_url", "").strip()
    model_name = request.form.get("model_name", "logistic")
    explain = request.form.get("explain") == "on"

    if model_name not in models:
        model_name = "logistic"

    if news_url:
        try:
            news_text = extract_text_from_url(news_url)
        except Exception as e:
            return render_template(
                "index.html",
                model_labels=MODEL_LABELS,
                prediction=f"Could not fetch URL: {e}",
                color="warning",
                input_url=news_url,
                model_name=model_name,
            )

    if not news_text:
        return render_template(
            "index.html",
            model_labels=MODEL_LABELS,
            prediction="Please enter news text or a valid URL.",
            color="warning",
            model_name=model_name,
        )

    cleaned = clean_text(news_text)
    if not cleaned:
        return render_template(
            "index.html",
            model_labels=MODEL_LABELS,
            prediction="Couldn't find enough readable text to analyze.",
            color="warning",
            input_text=news_text,
            input_url=news_url,
            model_name=model_name,
        )

    model = models[model_name]
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    if pred == 1:
        result, color, confidence = "Real News", "success", round(prob[1] * 100, 2)
    else:
        result, color, confidence = "Fake News", "danger", round(prob[0] * 100, 2)

    explanation_list = None
    if explain:
        try:
            exp = explainer.explain_instance(
                cleaned,
                lambda texts: predict_proba_for_lime(model, texts),
                num_features=LIME_NUM_FEATURES,
                num_samples=LIME_NUM_SAMPLES,
            )
            explanation_list = exp.as_list()
        except Exception:
            # If LIME fails for any reason, still show the prediction.
            explanation_list = None

    return render_template(
        "index.html",
        model_labels=MODEL_LABELS,
        prediction=result,
        color=color,
        confidence=confidence,
        input_text=news_text,
        input_url=news_url,
        model_name=model_name,
        explanation=explanation_list,
        explain_checked=explain,
    )


@app.route("/health")
def health():
    """Simple endpoint hosting platforms can ping to confirm the app is up."""
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)
