# 🛡️ Spam Classifier MLOps Pipeline

A complete MLOps pipeline built for learning — from training a model to deploying it with CI/CD.

## 🏗️ What's Inside

| Phase | What | Tech |
|-------|------|------|
| 1 | Train spam classifier | Scikit-learn, joblib |
| 2 | Experiment tracking | Weights & Biases |
| 3 | REST API | FastAPI, Uvicorn |
| 4 | Containerization | Docker |
| 5 | CI/CD + Deploy | GitHub Actions, Render |

## 🚀 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Classify a message as spam or ham |
| `/docs` | GET | Interactive Swagger UI |

## 📦 Quick Start

```bash
# Clone and set up
git clone https://github.com/shahidbaig-shaik/spam-classifier-mlops
cd spam-classifier-mlops
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Train the model
python train.py

# Run the API
uvicorn app.main:app --reload

# Test it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "You won a FREE prize! Call now!"}'
```

## 🐳 Run with Docker

```bash
docker build -t spam-classifier-api .
docker run -p 8000:8000 spam-classifier-api
```

## ✅ Run Tests

```bash
WANDB_MODE=disabled pytest tests/ -v
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98.5% |
| Precision | 100% |
| Recall | 88.6% |
| F1 Score | 0.94 |

Dataset: [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 messages
