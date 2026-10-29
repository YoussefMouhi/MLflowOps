import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)

from xgboost import XGBClassifier


# ==================================================
# 1. CONFIGURATION
# ==================================================

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
mlflow.set_experiment("MLflowOps-AI4I-Predictive-Maintenance")

DATA_DIR = Path("data/processed")
TARGET = "Machine failure"

CATEGORICAL_FEATURES = ["Type"]

NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

RANDOM_STATE = 42


# ==================================================
# 2. LOAD DATA
# ==================================================

print("\nLoading AI4I datasets...")

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

print(f"Training failures: {int(y_train.sum())}")
print(f"Validation failures: {int(y_val.sum())}")


# ==================================================
# 3. PREPROCESSING PIPELINE
# ==================================================

def create_preprocessor(scale_numeric=False):

    numeric_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_transformer,
                NUMERIC_FEATURES,
            ),
        ]
    )


# ==================================================
# 4. DEFINE MODELS
# ==================================================

models = {

    "LogisticRegression": Pipeline([
        (
            "preprocessor",
            create_preprocessor(scale_numeric=True)
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "RandomForest": Pipeline([
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]),

    "XGBoost": Pipeline([
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])
}


# ==================================================
# 5. TRAIN AND TRACK EXPERIMENTS
# ==================================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print(f"Training {model_name}...")
    print("=" * 55)

    with mlflow.start_run(run_name=model_name) as run:

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]

        metrics = {
            "val_accuracy": accuracy_score(
                y_val, predictions
            ),
            "val_precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "val_recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "val_f1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "val_roc_auc": roc_auc_score(
                y_val, probabilities
            ),
            "val_pr_auc": average_precision_score(
                y_val, probabilities
            ),
        }

        mlflow.log_params({
            "model_type": model_name,
            "dataset": "AI4I_2020",
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "random_state": RANDOM_STATE,
            "decision_threshold": 0.5,
        })

        mlflow.log_metrics(metrics)

        # ------------------------------------------
        # CONFUSION MATRIX
        # ------------------------------------------

        cm = confusion_matrix(y_val, predictions)

        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Failure", "Failure"]
        ).plot(ax=ax, colorbar=False)

        ax.set_title(f"{model_name} - Validation Confusion Matrix")

        mlflow.log_figure(
            fig,
            f"evaluation/{model_name}_confusion_matrix.png"
        )

        plt.close(fig)

        # ------------------------------------------
        # PRECISION-RECALL CURVE
        # ------------------------------------------

        precision, recall, _ = precision_recall_curve(
            y_val, probabilities
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{model_name} - Precision-Recall Curve")
        ax.grid(alpha=0.3)

        mlflow.log_figure(
            fig,
            f"evaluation/{model_name}_precision_recall.png"
        )

        plt.close(fig)

        # ------------------------------------------
        # SAVE MODEL
        # ------------------------------------------

        print(f"Saving {model_name} to MLflow...")

        # All three are sklearn-compatible pipelines.
        # The XGBoost estimator is wrapped inside a
        # preprocessing pipeline, so sklearn logging
        # preserves preprocessing and prediction together.

        if model_name == "LogisticRegression":

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model"
            )

        elif model_name == "RandomForest":

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

        else:

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                skops_trusted_types=[
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier"
                ]
            )

        # ------------------------------------------
        # STORE RESULTS
        # ------------------------------------------

        results.append({
            "Model": model_name,
            "Run ID": run.info.run_id,
            **metrics,
        })

        print(f"\n{model_name} validation results:")

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        print(f"\nRun ID: {run.info.run_id}")


# ==================================================
# 6. MODEL COMPARISON
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="val_f1",
    ascending=False
)

print("\n" + "=" * 85)
print("AI4I MODEL COMPARISON")
print("=" * 85)

print(
    results_df.drop(columns=["Run ID"]).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

best_model = results_df.iloc[0]

print("\nBEST VALIDATION MODEL")
print("-" * 45)

print(f"Model: {best_model['Model']}")
print(f"F1-score: {best_model['val_f1']:.4f}")
print(f"PR-AUC: {best_model['val_pr_auc']:.4f}")
print(f"MLflow Run ID: {best_model['Run ID']}")

# Save comparison locally for the next phase.
output_path = Path("data/processed/model_comparison.csv")

results_df.to_csv(output_path, index=False)

print(f"\nComparison saved to: {output_path}")
print("\nAll AI4I experiments successfully logged!")