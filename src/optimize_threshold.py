import os
from pathlib import Path

import json

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)


# ==================================================
# CONFIGURATION
# ==================================================

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
)

EXPERIMENT_NAME = "MLflowOps-AI4I-Predictive-Maintenance"

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError(
        f"MLflow experiment not found: {EXPERIMENT_NAME}"
    )

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'XGBoost'",
    order_by=["start_time DESC"],
    max_results=1,
)

if not runs:
    raise RuntimeError("No XGBoost training run found.")

RUN_ID = runs[0].info.run_id

MODEL_URI = f"runs:/{RUN_ID}/model"

print(f"Selected XGBoost run: {RUN_ID}")

DATA_PATH = Path("data/processed/validation.csv")

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Machine failure"


# ==================================================
# LOAD MODEL AND VALIDATION DATA
# ==================================================

print("\nLoading trained XGBoost model...")

model = mlflow.sklearn.load_model(
    MODEL_URI
)

df = pd.read_csv(DATA_PATH)

X_val = df.drop(columns=[TARGET])
y_val = df[TARGET]

probabilities = model.predict_proba(X_val)[:, 1]


# ==================================================
# FIND OPTIMAL THRESHOLD
# ==================================================

print("\nOptimizing classification threshold...")

precision, recall, thresholds = precision_recall_curve(
    y_val,
    probabilities,
)

# The last precision/recall point has no threshold.
f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_index])

optimized_predictions = (
    probabilities >= best_threshold
).astype(int)

default_predictions = (
    probabilities >= 0.5
).astype(int)


# ==================================================
# COMPARE THRESHOLDS
# ==================================================

def calculate_metrics(predictions):
    return {
        "precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_val, predictions, zero_division=0)
        ),
    }


default_metrics = calculate_metrics(default_predictions)
optimized_metrics = calculate_metrics(optimized_predictions)

comparison = pd.DataFrame([
    {
        "Threshold": 0.5,
        **default_metrics,
    },
    {
        "Threshold": best_threshold,
        **optimized_metrics,
    },
], index=["Default", "Optimized"])

print("\n" + "=" * 60)
print("THRESHOLD COMPARISON")
print("=" * 60)

print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))

print(f"\nOptimal threshold: {best_threshold:.4f}")


# ==================================================
# SAVE THRESHOLD CONFIGURATION
# ==================================================

configuration = {
    "model": "XGBoost",
    "source_run_id": RUN_ID,
    "threshold": best_threshold,
    "selection_metric": "validation_f1",
    "validation_f1": optimized_metrics["f1"],
}

config_path = OUTPUT_DIR / "best_threshold.json"

with open(config_path, "w") as file:
    json.dump(configuration, file, indent=4)

comparison.to_csv(
    OUTPUT_DIR / "threshold_comparison.csv"
)

print(f"\nThreshold saved to: {config_path}")


# ==================================================
# TRACK OPTIMIZATION IN MLFLOW
# ==================================================

mlflow.set_experiment("MLflowOps-AI4I-Predictive-Maintenance")

with mlflow.start_run(
    run_name="XGBoost-Threshold-Optimization"
):

    mlflow.log_param("source_run_id", RUN_ID)
    mlflow.log_param("selection_metric", "validation_f1")

    mlflow.log_metric(
        "optimal_threshold",
        best_threshold
    )

    for name, value in optimized_metrics.items():
        mlflow.log_metric(f"optimized_val_{name}", value)

    mlflow.log_artifact(str(config_path))

print("\nThreshold optimization completed successfully!")