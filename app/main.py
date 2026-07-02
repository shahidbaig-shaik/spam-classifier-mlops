"""
FastAPI service for spam classification.
Loads the trained scikit-learn pipeline at startup and exposes a /predict endpoint.
"""

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from pathlib import Path

# Resolved relative to this file so the server works from any working directory
MODEL_PATH = Path(__file__).parent.parent / "model" / "spam_classifier.joblib"


class PredictionRequest(BaseModel):
    """Incoming classification request."""
    message: str = Field(
        ...,
        min_length=1,
        description="The SMS message to classify as spam or ham",
        json_schema_extra={"examples": ["You won a free prize! Call now!"]},
    )


class PredictionResponse(BaseModel):
    """Classification result returned to the caller."""
    prediction: str = Field(description="Either 'spam' or 'ham'")
    confidence: float = Field(description="Model confidence score (0.0 to 1.0)")
    message: str = Field(description="The original message that was classified")


# Loaded once at startup, not per-request — avoids ~100ms disk-read overhead on every call
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; release on shutdown."""
    print("🚀 Loading spam classifier model...")
    print(f"   Model path: {MODEL_PATH}")
    ml_models["spam_classifier"] = joblib.load(MODEL_PATH)
    print("✅ Model loaded and ready to serve predictions!")
    yield
    print("👋 Shutting down, cleaning up...")
    ml_models.clear()


app = FastAPI(
    title="Spam Classifier API",
    description=(
        "A REST API that classifies SMS messages as spam or ham (not spam). "
        "Built with scikit-learn and served with FastAPI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """API info and available endpoints."""
    return {
        "name": "Spam Classifier API",
        "version": "1.0.0",
        "description": "Classify SMS messages as spam or ham",
        "endpoints": {
            "GET  /":         "This info page",
            "GET  /health":   "Health check",
            "POST /predict":  "Classify a message",
            "GET  /docs":     "Interactive API documentation (auto-generated!)",
        },
    }


@app.get("/health")
def health_check():
    """Liveness/readiness probe — confirms model is loaded and ready."""
    model_loaded = "spam_classifier" in ml_models
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Classify a single SMS message as spam or ham.

    Synchronous (def, not async def) so FastAPI dispatches it to a thread pool,
    keeping the event loop free for concurrent requests.
    """
    model = ml_models["spam_classifier"]

    prediction = model.predict([request.message])[0]
    probabilities = model.predict_proba([request.message])[0]
    confidence = float(probabilities.max())
    label = "spam" if prediction == 1 else "ham"

    return PredictionResponse(
        prediction=label,
        confidence=round(confidence, 4),
        message=request.message,
    )
