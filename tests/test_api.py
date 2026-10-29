import pytest
from pydantic import ValidationError

from api.main import MachineInput, PredictionResponse


VALID_MACHINE = {
    "machine_type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0,
}


def test_valid_machine_input():
    machine = MachineInput(**VALID_MACHINE)

    assert machine.machine_type == "M"
    assert machine.rotational_speed == 1551
    assert machine.torque == 42.8


@pytest.mark.parametrize(
    "field",
    ["rotational_speed", "torque", "tool_wear"],
)
def test_negative_values_rejected(field):
    invalid_data = VALID_MACHINE.copy()
    invalid_data[field] = -1

    with pytest.raises(ValidationError):
        MachineInput(**invalid_data)


def test_missing_required_field():
    invalid_data = VALID_MACHINE.copy()
    del invalid_data["torque"]

    with pytest.raises(ValidationError):
        MachineInput(**invalid_data)


def test_prediction_response_schema():
    response = PredictionResponse(
        prediction=0,
        failure_probability=0.12,
        decision_threshold=0.31,
        risk_level="LOW",
        model_name="MLflowOps-Predictive-Maintenance",
        model_version="1",
        timestamp="2026-10-29T23:00:00Z",
    )

    assert response.prediction == 0
    assert response.failure_probability == 0.12
    assert response.risk_level == "LOW"
