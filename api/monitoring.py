from prometheus_client import Counter, Histogram

PREDICTIONS_TOTAL = Counter(
    "mlflowops_predictions_total",
    "Total successful machine failure predictions",
)

PREDICTION_LATENCY = Histogram(
    "mlflowops_prediction_latency_seconds",
    "Time spent processing prediction requests",
)

PREDICTION_RISK = Counter(
    "mlflowops_prediction_risk_total",
    "Predictions grouped by risk category",
    ["risk_level"],
)

PREDICTION_ERRORS = Counter(
    "mlflowops_prediction_errors_total",
    "Prediction processing failures",
)
