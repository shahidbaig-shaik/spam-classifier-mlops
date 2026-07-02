"""
PHASE 3: Serve the Spam Classifier via FastAPI
================================================
This script turns your trained model into a REST API.

WHAT IS A REST API?
-------------------
Think of it like a waiter at a restaurant:
  1. You (the client) send a REQUEST:  "Is this message spam?"
  2. The API (the waiter) takes it to the kitchen (the model)
  3. The API returns a RESPONSE:      "Yes, it's spam (95% confident)"

Anyone can talk to your API — a website, a mobile app, another server,
or just a curl command from the terminal.

KEY CONCEPTS:
-------------
- FastAPI:     A modern Python web framework. It's fast, auto-generates docs,
               and validates input/output automatically.

- Endpoints:   URLs that do specific things:
                 GET  /health   → "Is the API alive?"
                 POST /predict  → "Classify this message"

- Pydantic:    Defines the SHAPE of requests and responses.
               If someone sends bad data, FastAPI auto-rejects it with
               a helpful error message (no manual validation needed).

- Lifespan:    Runs code at startup (load model) and shutdown (cleanup).
               This ensures the model is loaded ONCE, not on every request.

WHY NOT LOAD THE MODEL INSIDE THE ENDPOINT?
--------------------------------------------
Loading a model from disk takes ~100ms. If you load it on every request,
that's 100ms of unnecessary delay. By loading at startup, the model
sits in memory and predictions take <1ms.
"""

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from pathlib import Path

# Absolute path to the model file — works from ANY working directory.
# Path(__file__) = the location of this file (app/main.py)
# .parent = app/
# .parent = project root
# / "model/..." = project root / model / spam_classifier.joblib
MODEL_PATH = Path(__file__).parent.parent / "model" / "spam_classifier.joblib"


# ============================================================
#  STEP 1: Define the request/response shapes (Pydantic models)
# ============================================================

class PredictionRequest(BaseModel):
    """
    What the client sends TO us.

    Example JSON:
        {"message": "You won a free prize! Call now!"}

    Pydantic automatically:
      - Validates that 'message' exists
      - Validates that it's a string
      - Returns a 422 error if the input is wrong
    """
    message: str = Field(
        ...,  # ... means "required"
        min_length=1,
        description="The SMS message to classify as spam or ham",
        json_schema_extra={"examples": ["You won a free prize! Call now!"]},
    )


class PredictionResponse(BaseModel):
    """
    What we send BACK to the client.

    Example JSON:
        {"prediction": "spam", "confidence": 0.956, "message": "You won a free prize!"}
    """
    prediction: str = Field(description="Either 'spam' or 'ham'")
    confidence: float = Field(description="How confident the model is (0.0 to 1.0)")
    message: str = Field(description="The original message that was classified")


# ============================================================
#  STEP 2: Load the model at startup (lifespan)
# ============================================================

# This dictionary holds our loaded model(s) in memory.
# Think of it as a shelf where we place the model at startup
# and grab it whenever a prediction request comes in.
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler — runs at startup and shutdown.

    STARTUP (before 'yield'):
      - Load the trained model from disk into memory
      - This runs ONCE when the server starts

    SHUTDOWN (after 'yield'):
      - Clean up (free memory)
      - This runs when the server stops (Ctrl+C)

    WHY 'async context manager'?
      FastAPI needs to know when startup is done and when
      shutdown should happen. The 'yield' separates the two.
      Everything before yield = startup.
      Everything after yield = shutdown.
    """
    # ---- STARTUP ----
    print("🚀 Loading spam classifier model...")
    print(f"   Model path: {MODEL_PATH}")
    ml_models["spam_classifier"] = joblib.load(MODEL_PATH)
    print("✅ Model loaded and ready to serve predictions!")
    yield
    # ---- SHUTDOWN ----
    print("👋 Shutting down, cleaning up...")
    ml_models.clear()


# ============================================================
#  STEP 3: Create the FastAPI app
# ============================================================

app = FastAPI(
    title="Spam Classifier API",
    description=(
        "A REST API that classifies SMS messages as spam or ham (not spam). "
        "Built with scikit-learn and served with FastAPI."
    ),
    version="1.0.0",
    lifespan=lifespan,  # <-- This hooks up our startup/shutdown logic
)


# ============================================================
#  STEP 4: Define the endpoints
# ============================================================

@app.get("/")
def root():
    """
    Root endpoint — basic info about the API.

    This is what you see when you visit http://localhost:8000/
    It's like a welcome sign on the front door.
    """
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
    """
    Health check endpoint — "Is the API alive and is the model loaded?"

    WHY DO WE NEED THIS?
    In production, monitoring tools (like Render, Kubernetes, etc.)
    periodically hit this endpoint. If it returns anything other than
    200 OK, they know something is wrong and can alert you or restart
    the service.

    It's also what our GitHub Actions test will check!
    """
    model_loaded = "spam_classifier" in ml_models
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Prediction endpoint — the main event!

    Takes a message, runs it through the model, returns the prediction.

    WHY 'def' AND NOT 'async def'?
    Our model prediction is CPU-bound (math, not network I/O).
    With 'def', FastAPI automatically runs it in a thread pool,
    which prevents it from blocking other requests.
    With 'async def', it would block the event loop — bad for
    concurrent requests.

    THE FLOW:
    1. Client sends: POST /predict {"message": "Free prize!"}
    2. Pydantic validates the input (is it a string? is it non-empty?)
    3. We grab the model from ml_models
    4. model.predict() → 0 or 1 (ham or spam)
    5. model.predict_proba() → [0.04, 0.96] (probability for each class)
    6. We return: {"prediction": "spam", "confidence": 0.96}
    """
    model = ml_models["spam_classifier"]

    # Get prediction (0 = ham, 1 = spam)
    prediction = model.predict([request.message])[0]

    # Get probability for each class: [P(ham), P(spam)]
    probabilities = model.predict_proba([request.message])[0]
    confidence = float(probabilities.max())

    # Convert numeric prediction to human-readable label
    label = "spam" if prediction == 1 else "ham"

    return PredictionResponse(
        prediction=label,
        confidence=round(confidence, 4),
        message=request.message,
    )
