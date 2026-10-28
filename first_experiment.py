
import mlflow
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Connect to the local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MLflowOps-Predictive-Maintenance")

# Generate synthetic equipment failure data
X, y = make_classification(
    n_samples=2000,
    n_features=12,
    n_informative=8,
    weights=[0.9, 0.1],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Start the tracked experiment
with mlflow.start_run(run_name="RandomForest-Baseline"):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "dataset": "synthetic_equipment_failure"
    })

    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_score": f1
    })

    mlflow.sklearn.log_model(
        model,
        name="random_forest"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Experiment successfully logged to MLflow!")

