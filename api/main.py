import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from api.monitoring import (
    PREDICTIONS_TOTAL,
    PREDICTION_LATENCY,
    PREDICTION_RISK,
    PREDICTION_ERRORS,
)
from mlflow import MlflowClient
from pydantic import BaseModel, Field


# ==================================================
# CONFIGURATION
# ==================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

MODEL_NAME = "MLflowOps-Predictive-Maintenance"

MODEL_ALIAS = "champion"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

model = None
model_version = None
decision_threshold = None


# ==================================================
# INPUT SCHEMA
# ==================================================

class MachineInput(BaseModel):

    machine_type: str = Field(
        ...,
        description="Machine quality category: L, M, or H"
    )

    air_temperature: float = Field(
        ...,
        description="Air temperature in Kelvin"
    )

    process_temperature: float = Field(
        ...,
        description="Process temperature in Kelvin"
    )

    rotational_speed: int = Field(
        ...,
        ge=0,
        description="Rotational speed in RPM"
    )

    torque: float = Field(
        ...,
        ge=0,
        description="Torque in Nm"
    )

    tool_wear: int = Field(
        ...,
        ge=0,
        description="Tool wear in minutes"
    )


# ==================================================
# OUTPUT SCHEMA
# ==================================================

class PredictionResponse(BaseModel):

    prediction: int

    failure_probability: float

    decision_threshold: float

    risk_level: str

    model_name: str

    model_version: str

    timestamp: str


# ==================================================
# LOAD CHAMPION MODEL
# ==================================================

def load_champion_model():

    global model
    global model_version
    global decision_threshold

    print("\nLoading champion model from MLflow...")

    version_info = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS
    )

    model_version = version_info.version

    decision_threshold = float(
        version_info.tags.get(
            "decision_threshold",
            "0.5"
        )
    )

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    model = mlflow.sklearn.load_model(model_uri)

    print("Champion model loaded successfully!")
    print(f"Model version: {model_version}")
    print(f"Decision threshold: {decision_threshold}")


# ==================================================
# FASTAPI LIFESPAN
# ==================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    load_champion_model()

    yield


# ==================================================
# INITIALIZE FASTAPI
# ==================================================

app = FastAPI(

    title="MLflowOps Predictive Maintenance API",

    description=(
        "Machine failure prediction using "
        "XGBoost and MLflow Model Registry."
    ),

    version="1.0.0",

    lifespan=lifespan
)


# ==================================================
# HOME ENDPOINT
# ==================================================



@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

@app.get("/")
def home():

    return {
        "project": "MLflowOps",
        "service": "Predictive Maintenance API",
        "status": "running",
        "model": MODEL_NAME,
        "version": model_version
    }


# ==================================================
# HEALTH ENDPOINT
# ==================================================

@app.get("/health")
def health():

    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None,
        "model_version": model_version
    }


# ==================================================
# PREDICTION ENDPOINT
# ==================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(machine: MachineInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    machine_type = machine.machine_type.upper()

    if machine_type not in ["L", "M", "H"]:
        raise HTTPException(
            status_code=422,
            detail="Machine type must be L, M, or H"
        )

    # Convert API input into training feature format
    input_data = pd.DataFrame([{

        "Type": machine_type,

        "Air temperature": machine.air_temperature,

        "Process temperature": machine.process_temperature,

        "Rotational speed": machine.rotational_speed,

        "Torque": machine.torque,

        "Tool wear": machine.tool_wear

    }])

    start_time = time.perf_counter()

    # Generate probability
    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    # Apply optimized threshold
    prediction = int(
        probability >= decision_threshold
    )

    # Assign descriptive risk category
    if prediction == 1:
        risk_level = "HIGH"
    elif probability >= decision_threshold * 0.7:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    PREDICTIONS_TOTAL.inc()
    PREDICTION_RISK.labels(risk_level=risk_level).inc()
    PREDICTION_LATENCY.observe(
        time.perf_counter() - start_time
    )

    return PredictionResponse(

        prediction=prediction,

        failure_probability=round(
            probability, 4
        ),

        decision_threshold=decision_threshold,

        risk_level=risk_level,

        model_name=MODEL_NAME,

        model_version=str(model_version),

        timestamp=datetime.now(
            timezone.utc
        ).isoformat()
    )