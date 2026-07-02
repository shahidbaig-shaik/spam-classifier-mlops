"""
PHASE 5: API Tests
==================
These tests run automatically in GitHub Actions every time you push code.

WHY DO WE TEST?
---------------
Without tests, the only way to know your API works is to deploy it and
manually check. That's slow and risky.

With tests:
  - Every push automatically runs these checks
  - If something breaks, GitHub Actions catches it BEFORE deploying
  - You get a green ✅ or red ❌ on every commit

HOW DOES TestClient WORK?
--------------------------
FastAPI's TestClient lets us call the API endpoints directly in Python
WITHOUT starting a real HTTP server. It's fast, reliable, and works
perfectly in CI environments like GitHub Actions.

Think of it like a "fake browser" that makes requests to your app
in the same process — no ports, no networking needed.

WHY A FIXTURE?
--------------
Our app uses a lifespan handler to load the model at startup.
Using TestClient as a context manager (via a fixture) tells FastAPI:
"Run the full lifespan — startup AND shutdown."
Without this, ml_models is empty and all prediction tests fail.

'scope=module' = start the app once, reuse for ALL tests, then shut down.
This is fast — we don't reload the model for every single test.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


# ============================================================
#  Pytest fixture: starts the app (runs lifespan = loads model)
# ============================================================
@pytest.fixture(scope="module")
def client():
    """
    A shared TestClient that runs the full app lifespan.
    Each test function that lists 'client' as a parameter gets this.
    """
    with TestClient(app) as c:
        yield c


# ============================================================
#  TEST 1: Root endpoint
# ============================================================
def test_root(client):
    """GET / should return 200 with API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data


# ============================================================
#  TEST 2: Health check — this is what GitHub Actions checks!
# ============================================================
def test_health_check_returns_200(client):
    """
    GET /health must return 200 OK.

    This is THE critical test for CI/CD. GitHub Actions will:
      1. Start the app via TestClient
      2. Hit /health
      3. Check for 200 OK
      4. Only deploy if this passes

    It also confirms the model is actually loaded in memory.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# ============================================================
#  TEST 3: Predict spam
# ============================================================
def test_predict_spam(client):
    """POST /predict with obvious spam should return 'spam'."""
    response = client.post(
        "/predict",
        json={"message": "WINNER!! Claim your FREE prize now! Call 09061701461!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "spam"
    assert 0.0 <= data["confidence"] <= 1.0
    assert "message" in data


# ============================================================
#  TEST 4: Predict ham
# ============================================================
def test_predict_ham(client):
    """POST /predict with a normal message should return 'ham'."""
    response = client.post(
        "/predict",
        json={"message": "Hey, are we still meeting for lunch tomorrow?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "ham"
    assert 0.0 <= data["confidence"] <= 1.0


# ============================================================
#  TEST 5: Validation — missing field returns 422
# ============================================================
def test_predict_missing_message_returns_422(client):
    """
    POST /predict with no 'message' field should return 422.
    422 = Unprocessable Entity — Pydantic rejected the bad input.
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422


# ============================================================
#  TEST 6: Validation — empty string returns 422
# ============================================================
def test_predict_empty_message_returns_422(client):
    """
    POST /predict with an empty string should return 422.
    We set min_length=1 on the message field.
    """
    response = client.post("/predict", json={"message": ""})
    assert response.status_code == 422


# ============================================================
#  TEST 7: Response shape
# ============================================================
def test_predict_response_has_correct_fields(client):
    """
    Every prediction response must have exactly the right fields.
    This catches bugs where we accidentally change the API shape.
    """
    response = client.post(
        "/predict",
        json={"message": "Test message for field checking"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "message" in data
    assert data["prediction"] in ["spam", "ham"]
