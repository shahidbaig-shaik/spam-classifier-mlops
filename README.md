# 🛡️ Spam Classifier — End-to-End MLOps Pipeline

> Trained a text classification model, served it as a production REST API, containerized with Docker, and deployed with a fully automated CI/CD pipeline. **Zero manual steps from code push to live deployment.**

[![CI/CD Pipeline](https://github.com/shahidbaig-shaik/spam-classifier-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/shahidbaig-shaik/spam-classifier-mlops/actions/workflows/ci-cd.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-containerized-blue?logo=docker)](https://docker.com)
[![Deployed on Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render)](https://spam-classifier-api-5l30.onrender.com)

**🌐 Live API:** https://spam-classifier-api-5l30.onrender.com/docs

---

## 📸 What It Does

Send any text message to the API → get back an instant spam/ham classification with confidence score.

```bash
curl -X POST https://spam-classifier-api-5l30.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You won a FREE iPhone! Call NOW!"}'
```

```json
{
  "prediction": "spam",
  "confidence": 0.9986,
  "message": "Congratulations! You won a FREE iPhone! Call NOW!"
}
```

---

## 🏗️ Architecture

```
                      ┌─────────────────────────────────┐
                      │         CI/CD Pipeline           │
  git push ──────────▶│  GitHub Actions                  │
                      │  ├─ pytest (7 tests)              │
                      │  └─ Deploy hook → Render         │
                      └──────────────┬──────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │              Render (Cloud)                   │
              │  ┌──────────────────────────────────────┐   │
              │  │         Docker Container              │   │
              │  │  ┌───────────────────────────────┐   │   │
              │  │  │  FastAPI                       │   │   │
              │  │  │  ├── GET  /health              │   │   │
              │  │  │  ├── POST /predict             │   │   │
              │  │  │  └── GET  /docs                │   │   │
              │  │  │         │                      │   │   │
              │  │  │  joblib.load(model.joblib)     │   │   │
              │  │  │  TfidfVectorizer + NaiveBayes  │   │   │
              │  │  └───────────────────────────────┘   │   │
              │  └──────────────────────────────────────┘   │
              └──────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML** | Scikit-learn, joblib | Model training & serialization |
| **Tracking** | Weights & Biases | Experiment tracking & comparison |
| **API** | FastAPI, Pydantic, Uvicorn | REST API with auto-validation |
| **Container** | Docker (multi-stage) | Reproducible, portable deployment |
| **CI/CD** | GitHub Actions | Automated testing & deployment |
| **Cloud** | Render | Production hosting |

---

## 📊 Model Performance

Trained on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 real-world SMS messages.

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.5% |
| **Precision** | 100% |
| **Recall** | 88.6% |
| **F1 Score** | 0.94 |

> Compared MultinomialNB vs LogisticRegression via W&B experiment tracking — Naive Bayes outperformed on this dataset. Also tuned `alpha=0.1` and `max_features=10000`, boosting recall from 78.5% → 88.6%.

---

## ⚙️ CI/CD Pipeline

Every `git push` to `main` automatically:

```
Push → GitHub Actions → pytest (7 tests) → Render Deploy Hook → Live in ~60s
```

- ✅ Tests pass → Render redeploys automatically
- ❌ Tests fail → deployment is blocked, nothing breaks in production

**[View live pipeline runs →](https://github.com/shahidbaig-shaik/spam-classifier-mlops/actions)**

---

## 🚀 Run It Locally

```bash
# Clone
git clone https://github.com/shahidbaig-shaik/spam-classifier-mlops
cd spam-classifier-mlops

# Set up environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Train the model
python train.py

# Start the API
uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

### Or with Docker:

```bash
docker build -t spam-classifier-api .
docker run -p 8000:8000 spam-classifier-api
# → http://localhost:8000/docs
```

---

## 🧪 Tests

```bash
WANDB_MODE=disabled pytest tests/ -v
```

```
tests/test_api.py::test_root                              PASSED
tests/test_api.py::test_health_check_returns_200          PASSED
tests/test_api.py::test_predict_spam                      PASSED
tests/test_api.py::test_predict_ham                       PASSED
tests/test_api.py::test_predict_missing_message_returns_422  PASSED
tests/test_api.py::test_predict_empty_message_returns_422    PASSED
tests/test_api.py::test_predict_response_has_correct_fields  PASSED

7 passed in 1.53s
```

---

## 📁 Project Structure

```
spam-classifier-mlops/
├── train.py                    ← Training script with W&B tracking
├── app/
│   └── main.py                 ← FastAPI app (health + predict endpoints)
├── model/
│   └── spam_classifier.joblib  ← Serialized model pipeline
├── tests/
│   └── test_api.py             ← 7 integration tests
├── Dockerfile                  ← Multi-stage build (191MB final image)
├── docker-compose.yml          ← Local development
├── requirements.txt            ← Production dependencies
├── requirements-dev.txt        ← Dev/test dependencies
└── .github/
    └── workflows/
        └── ci-cd.yml           ← Automated test + deploy pipeline
```

---

## 🔗 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | API info and available endpoints |
| `/health` | `GET` | Health check — model load status |
| `/predict` | `POST` | Classify a message as `spam` or `ham` |
| `/docs` | `GET` | Interactive Swagger UI |

**Request body (`/predict`):**
```json
{ "message": "Your text here" }
```

**Response:**
```json
{
  "prediction": "spam",
  "confidence": 0.9986,
  "message": "Your text here"
}
```
