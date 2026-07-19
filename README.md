# 📰 Fake News Detection using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</p>

---

## 📌 Overview

Fake news has become one of the biggest challenges in the digital era. This project is a **Machine Learning powered Fake News Detection System** that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP).

The application is built with **Flask** and allows users to either:

- 📝 Enter news text manually
- 🌐 Paste a news article URL
- 🤖 Select different Machine Learning models
- 💡 Generate an explainable prediction using **LIME**

---

## ✨ Features

- 🔍 Detect Fake and Real News
- 📰 Accepts News Text or URL
- 🤖 Multiple ML Models
  - Logistic Regression
  - Naive Bayes
  - Random Forest
- 📊 TF-IDF Vectorization
- 💡 LIME Explainability
- 🎨 Clean Responsive UI
- ⚡ Fast Predictions
- 🌍 Deployable on Render

---

# 📷 Application Preview

> Add screenshots here

```
Home Page Screenshot

Prediction Result Screenshot

LIME Explanation Screenshot
```

---

# 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Framework | Flask |
| Machine Learning | Scikit-Learn |
| NLP | TF-IDF Vectorizer |
| Explainability | LIME |
| Frontend | HTML, CSS |
| Deployment | Render |
| Version Control | Git & GitHub |

---

# 🧠 Machine Learning Models

The application supports multiple classifiers.

| Model | Purpose |
|--------|----------|
| Logistic Regression | Fast & Accurate |
| Multinomial Naive Bayes | Text Classification |
| Random Forest | Ensemble Learning |

---

# 📂 Project Structure

```text
Fake-News-Detection/
│
├── app.py
├── train_model.py
├── Fake.csv
├── True.csv
│
├── logistic.pkl
├── naive_bayes.pkl
├── random_forest.pkl
├── vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── requirements.txt
├── runtime.txt
├── Procfile
├── README.md
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/abhay9151/Fake-News-Detection

cd Fake-News-Detection
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Models

Run only once (or whenever the dataset changes)

```bash
python train_model.py
```

This generates:

- vectorizer.pkl
- logistic.pkl
- naive_bayes.pkl
- random_forest.pkl

---

## Run Application

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

# 🚀 Deployment (Render)

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app --timeout 120 --workers 1 --threads 2
```

---

# 📊 Dataset

The project uses two datasets:

- **True.csv**
- **Fake.csv**

Each dataset contains news articles labeled as **Real** or **Fake**.

---

# 💡 Explainable AI

This project integrates **LIME (Local Interpretable Model-Agnostic Explanations)** to explain why a news article is classified as **Fake** or **Real**.

LIME highlights the most influential words contributing to the prediction, making the model more transparent and trustworthy.

---

# 📈 Model Training Pipeline

```
Dataset
      │
      ▼
Text Cleaning
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Train ML Models
      │
      ▼
Save Models (.pkl)
      │
      ▼
Flask Application
      │
      ▼
Prediction + Explanation
```

---

# 📌 Future Improvements

- 🔹 Deep Learning Models (LSTM / BERT)
- 🔹 User Authentication
- 🔹 News History
- 🔹 Confidence Score Visualization
- 🔹 REST API
- 🔹 Docker Support
- 🔹 Database Integration
- 🔹 Dark Mode

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 👨‍💻 Author

**Abhay Pratap Singh**

🎓 B.Tech CS
🏫 KIET University, Ghaziabad

### Connect with Me

- 💼 LinkedIn: https://www.linkedin.com/in/abhay-pratap-singh-48121a318/
- 💻 GitHub: https://github.com/abhay9151

---

# ⭐ If you like this project

Give this repository a ⭐ on GitHub!

It motivates me to build more Machine Learning and AI projects.

---

## 📜 License

This project is licensed under the **MIT License**.

---

<p align="center">
Made with ❤️ using Python, Flask, Machine Learning and NLP
</p>
