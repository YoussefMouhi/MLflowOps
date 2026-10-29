from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.main as api_module


@pytest.fixture
def client(monkeypatch):
    # Avoid connecting to MLflow during API startup.
    monkeypatch.setattr(api_module, "load_champion_model", lambda: None)

    # Supply a predictable model for testing.
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.9, 0.1]]

    monkeypatch.setattr(api_module, "model", mock_model)
    monkeypatch.setattr(api_module, "model_version", "test-1")
    monkeypatch.setattr(api_module, "decision_threshold", 0.31)

    with TestClient(api_module.app) as test_client:
        yield test_client


VALID_PAYLOAD = {
    "machine_type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0,
}


def test_home_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint(client):
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 0
    assert result["failure_probability"] == pytest.approx(0.1)
    assert result["model_version"] == "test-1"


def test_invalid_prediction_input(client):
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["rotational_speed"] = -100

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
