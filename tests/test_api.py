"""
API integration tests using FastAPI's TestClient.
The module-scoped fixture runs the full lifespan (model load) once per test session.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Shared TestClient with full app lifespan (loads model at startup)."""
    with TestClient(app) as c:
        yield c


def test_root(client):
    """GET / should return 200 with API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data


def test_health_check_returns_200(client):
    """GET /health must return 200 with model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


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


def test_predict_missing_message_returns_422(client):
    """POST /predict with no 'message' field should return 422 (Pydantic validation)."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_empty_message_returns_422(client):
    """POST /predict with an empty string should return 422 (min_length=1 constraint)."""
    response = client.post("/predict", json={"message": ""})
    assert response.status_code == 422


def test_predict_response_has_correct_fields(client):
    """Every prediction response must contain the expected fields with valid types."""
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
