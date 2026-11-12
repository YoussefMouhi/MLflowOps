from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.main as api_module


@pytest.fixture
def client(monkeypatch):
    # Prevent real MLflow connections during tests.
    monkeypatch.setattr(
        api_module,
        "load_champion_model",
        lambda: None,
    )

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.9, 0.1]]

    monkeypatch.setattr(api_module, "model", mock_model)
    monkeypatch.setattr(api_module, "model_version", "test-1")
    monkeypatch.setattr(api_module, "decision_threshold", 0.31)

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "mlflowops_predictions_total" in response.text
    assert "mlflowops_prediction_latency_seconds" in response.text


def test_prediction_counter(client):
    payload = {
        "machine_type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0,
    }

    before = client.get("/metrics").text

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    after = client.get("/metrics").text

    assert before != after
    assert "mlflowops_prediction_risk_total" in after


def test_inference_error_increments_error_counter(client, monkeypatch):
    from prometheus_client import REGISTRY

    payload = {
        "machine_type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0,
    }

    before_errors = REGISTRY.get_sample_value(
        "mlflowops_prediction_errors_total"
    ) or 0
    before_success = REGISTRY.get_sample_value(
        "mlflowops_predictions_total"
    ) or 0

    api_module.model.predict_proba.side_effect = RuntimeError(
        "Internal model details should stay private"
    )

    response = client.post("/predict", json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model inference temporarily unavailable"
    assert "Internal model details" not in response.text
    assert REGISTRY.get_sample_value(
        "mlflowops_prediction_errors_total"
    ) == before_errors + 1
    assert (REGISTRY.get_sample_value(
        "mlflowops_predictions_total"
    ) or 0) == before_success
