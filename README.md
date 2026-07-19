# Fake News Detector

A Flask web app that classifies news text (or a news article URL) as **Real**
or **Fake** using TF-IDF features and a choice of three scikit-learn models
(Logistic Regression, Naive Bayes, Random Forest), with optional word-level
LIME explanations.

## Project structure

```
.
├── app.py                 # Flask app (loads models, serves predictions)
├── train_model.py         # Trains the models from Fake.csv / True.csv
├── Fake.csv / True.csv    # Training data
├── vectorizer.pkl         # Saved TF-IDF vectorizer (generated)
├── logistic.pkl           # Saved Logistic Regression model (generated)
├── naive_bayes.pkl        # Saved Naive Bayes model (generated)
├── random_forest.pkl      # Saved Random Forest model (generated)
├── templates/index.html   # UI
├── static/style.css       # Styling
├── requirements.txt       # Pinned dependencies
├── runtime.txt            # Python version pin (for Render)
└── Procfile                # Start command for Render/Heroku
```

## Run it locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the models (only needed once, or whenever you change train_model.py)
python train_model.py

# Start the app
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Retraining the models

`train_model.py` reads `Fake.csv` and `True.csv`, cleans the text, fits a
TF-IDF vectorizer (capped at 5000 features), and trains all three models.
Re-run it any time you change the data or model settings:

```bash
python train_model.py
```

It prints accuracy/precision/recall for each model and the size of each
saved `.pkl` file, so you can see the trade-off if you change hyperparameters.

**Note on the Random Forest:** it's deliberately capped at
`n_estimators=100, max_depth=20`. An unbounded forest scores marginally
higher but produces a 40MB+ pickle file that's slow to run LIME explanations
against on a low-resource server — not worth it for a small accuracy gain.

## Deploying to Render (free tier)

1. Push this project to a GitHub repo (make sure the `.pkl` files are
   committed — see note below).
2. On [Render](https://render.com), click **New → Web Service** and connect
   the repo.
3. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --timeout 120 --workers 1 --threads 2`
     (already set in the `Procfile`, so Render should pick it up automatically)
   - **Instance Type:** Free
4. Deploy. First load after idling takes ~1 minute (free instances spin down
   after 15 minutes of inactivity) — this is normal for Render's free tier.

### Why this version won't hit the earlier "Internal Server Error"

The original version defaulted to LIME's `num_samples=5000` per explanation
and used a 44MB, unbounded Random Forest. On Render's free instance (512MB
RAM, 0.1 CPU), that combination could exceed gunicorn's request timeout or
the memory limit, and Render's proxy would show a generic error page.

This version:
- Caps the Random Forest at 100 trees / depth 20 (~11MB instead of ~44MB).
- Caps LIME at 300 samples (still gives a meaningful explanation, far less
  compute than the 5000-sample default).
- Makes the LIME explanation **opt-in** via a checkbox, so a plain prediction
  is always fast.
- Sets an explicit gunicorn `--timeout 120` as extra headroom.
- Pins all dependency versions so a future build can't silently pull an
  incompatible library version.

If you still hit errors after deploying, check **Render Dashboard → your
service → Logs** for the actual traceback (out-of-memory vs. timeout vs.
something else) — that will tell you exactly what to adjust next.

## Model files and Git

The `.pkl` files are a few MB to ~11MB. Regular `git add`/`commit`/`push`
handles this fine. If your repo host has strict size limits and rejects the
push, either regenerate the models on the server as a build step
(`python train_model.py` in the build command) or use [Git LFS](https://git-lfs.com/).
