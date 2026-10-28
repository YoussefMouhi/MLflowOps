import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ==================================================
# CONFIGURATION
# ==================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "MLflowOps-AI4I-Predictive-Maintenance"

MODEL_NAME = "MLflowOps-Predictive-Maintenance"

SOURCE_RUN_ID = "ebbe98a1d9a44e6293a4589dfa900a01"

MODEL_URI = f"runs:/{SOURCE_RUN_ID}/model"

THRESHOLD_PATH = Path("data/processed/best_threshold.json")

TEST_DATA_PATH = Path("data/processed/test.csv")

TARGET = "Machine failure"


# ==================================================
# 1. LOAD THRESHOLD CONFIGURATION
# ==================================================

print("\nLoading optimized threshold...")

with open(THRESHOLD_PATH, "r") as file:
    threshold_config = json.load(file)

threshold = float(threshold_config["threshold"])

if threshold_config["source_run_id"] != SOURCE_RUN_ID:
    raise ValueError("Threshold configuration does not match model run.")

print(f"Optimized threshold: {threshold:.4f}")


# ==================================================
# 2. LOAD TRAINED MODEL
# ==================================================

print("\nLoading trained XGBoost model...")

model = mlflow.sklearn.load_model(MODEL_URI)


# ==================================================
# 3. LOAD UNSEEN TEST DATA
# ==================================================

print("\nLoading untouched test dataset...")

test_df = pd.read_csv(TEST_DATA_PATH)

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print(f"Test observations: {len(X_test)}")
print(f"Actual failures: {int(y_test.sum())}")


# ==================================================
# 4. GENERATE PREDICTIONS
# ==================================================

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= threshold).astype(int)


# ==================================================
# 5. EVALUATE MODEL
# ==================================================

metrics = {
    "test_accuracy": accuracy_score(y_test, predictions),
    "test_precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "test_recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "test_f1": f1_score(
        y_test, predictions, zero_division=0
    ),
    "test_roc_auc": roc_auc_score(y_test, probabilities),
    "test_pr_auc": average_precision_score(
        y_test, probabilities
    ),
}

cm = confusion_matrix(y_test, predictions, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()


# ==================================================
# 6. PRINT FINAL RESULTS
# ==================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")

print("\nConfusion Matrix:")
print(f"True Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")


# ==================================================
# 7. LOG FINAL TEST RESULTS
# ==================================================

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="XGBoost-Final-Test-Evaluation"):

    mlflow.log_param("source_run_id", SOURCE_RUN_ID)
    mlflow.log_param("decision_threshold", threshold)

    mlflow.log_metrics(metrics)

    mlflow.log_metrics({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    })

    mlflow.log_artifact(str(THRESHOLD_PATH))


# ==================================================
# 8. REGISTER MODEL
# ==================================================

print("\nRegistering XGBoost model...")

registered_model = mlflow.register_model(
    model_uri=MODEL_URI,
    name=MODEL_NAME
)

client = MlflowClient()

version = registered_model.version

client.set_model_version_tag(
    name=MODEL_NAME,
    version=version,
    key="decision_threshold",
    value=str(threshold)
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=version,
    key="dataset",
    value="AI4I_2020"
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=version,
    key="selection_metric",
    value="validation_f1"
)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=version
)


# ==================================================
# 9. SAVE DEPLOYMENT CONFIGURATION
# ==================================================

deployment_config = {
    "registered_model_name": MODEL_NAME,
    "model_version": version,
    "model_alias": "champion",
    "source_run_id": SOURCE_RUN_ID,
    "threshold": threshold,
    "test_metrics": {
        key: float(value)
        for key, value in metrics.items()
    },
}

config_path = Path("data/processed/deployment_config.json")

with open(config_path, "w") as file:
    json.dump(deployment_config, file, indent=4)


print("\n" + "=" * 60)
print("MODEL REGISTRATION COMPLETED")
print("=" * 60)

print(f"Registered model: {MODEL_NAME}")
print(f"Model version: {version}")
print("Model alias: champion")
print(f"Decision threshold: {threshold:.4f}")

print(f"\nDeployment configuration saved to: {config_path}")

print("\nMLflowOps model is ready for API development!")