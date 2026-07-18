from flask import Flask, render_template, request
import joblib
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from lime.lime_text import LimeTextExplainer

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")
models = {
    "logistic": joblib.load(BASE_DIR / "logistic.pkl"),
    "random_forest": joblib.load(BASE_DIR / "random_forest.pkl"),
    "naive_bayes": joblib.load(BASE_DIR / "naive_bayes.pkl")
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_from_url(url):
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
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "").strip()
    news_url = request.form.get("news_url", "").strip()
    model_name = request.form.get("model_name", "logistic")

    if model_name not in models:
        model_name = "logistic"

    if news_url:
        try:
            news_text = extract_text_from_url(news_url)
        except Exception as e:
            return render_template(
                "index.html",
                prediction=f"Could not fetch URL: {str(e)}",
                input_text=request.form.get("news_text", ""),
                input_url=news_url
            )

    if not news_text:
        return render_template("index.html", prediction="Please enter news text or a valid URL.")

    cleaned = clean_text(news_text)
    model = models[model_name]

    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    if pred == 1:
        result = "Real News"
        color = "success"
        confidence = round(prob[1] * 100, 2)
    else:
        result = "Fake News"
        color = "danger"
        confidence = round(prob[0] * 100, 2)

    explainer = LimeTextExplainer(class_names=["Fake", "Real"])
    exp = explainer.explain_instance(
        cleaned,
        lambda texts: predict_proba_for_lime(model, texts),
        num_features=8
    )
    explanation_list = exp.as_list()

    return render_template(
        "index.html",
        prediction=result,
        color=color,
        confidence=confidence,
        input_text=news_text,
        input_url=news_url,
        model_name=model_name,
        explanation=explanation_list
    )

if __name__ == "__main__":
    app.run(debug=True)