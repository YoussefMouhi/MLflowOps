import os
from datetime import datetime

import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MLflowOps | Predictive Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


# ============================================================
# CUSTOM DARK THEME
# ============================================================

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    color-scheme: dark;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            ellipse at 85% 0%,
            rgba(20, 184, 166, 0.08),
            transparent 35%
        ),
        #080D18;
    color: #F1F5F9;
}

.block-container {
    max-width: 1380px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* GENERAL TEXT */

.stApp p,
.stApp li {
    color: #E2E8F0;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #FFFFFF !important;
    font-weight: 700;
    letter-spacing: -0.035em;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #101B2B !important;
    border-right: 1px solid #34465F;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #E2E8F0 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* BRAND */

.brand {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.06em;
    color: #FFFFFF;
    margin-bottom: 0.25rem;
}

.brand span {
    color: #5EEAD4;
}

.brand-caption {
    color: #C2D0E0;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    margin-bottom: 1.8rem;
}

/* STATUS */

.status-online {
    color: #6EE7B7;
    font-size: 0.85rem;
    font-weight: 700;
}

.status-offline {
    color: #FDA4AF;
    font-size: 0.85rem;
    font-weight: 700;
}

/* HERO */

.eyebrow {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5EEAD4;
    margin-bottom: 0.7rem;
}

.hero-title {
    font-size: clamp(2.2rem, 4vw, 3.5rem);
    font-weight: 800;
    line-height: 1.13;
    letter-spacing: -0.055em;
    color: #FFFFFF;
    margin-bottom: 1rem;
}

.hero-title span {
    color: #5EEAD4;
}

.hero-subtitle {
    color: #C2D0E0;
    font-size: 1.05rem;
    line-height: 1.8;
    max-width: 780px;
}

/* SECTION HEADINGS */

.section-heading {
    font-size: 1.25rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 2rem 0 0.9rem;
}

/* FEATURE CARDS */

.feature-card {
    background: #142238;
    border: 1px solid #34465F;
    border-radius: 16px;
    padding: 24px;
    min-height: 145px;
    margin-top: 0.5rem;
}

.feature-card-title {
    color: #FFFFFF;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.feature-card-text {
    color: #C2D0E0;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* FORMS */

[data-testid="stForm"] {
    background: #111D2E !important;
    border: 1px solid #34465F !important;
    border-radius: 18px;
    padding: 24px;
}

/* INPUT LABELS */

[data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* INPUT VALUES */

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background-color: #162235 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* SELECT BOXES */

[data-baseweb="select"] > div {
    background-color: #162235 !important;
    color: #FFFFFF !important;
    border-color: #40536B !important;
}

[data-baseweb="select"] span {
    color: #FFFFFF !important;
}

/* BUTTONS */

div.stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(
        110deg,
        #14B8A6,
        #0891B2
    ) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    border-radius: 10px;
    min-height: 50px;
}

div.stFormSubmitButton > button[kind="primary"] p {
    color: #FFFFFF !important;
}

/* METRIC CARDS */

div[data-testid="stMetric"] {
    background: #142238 !important;
    border: 1px solid #34465F !important;
    border-radius: 14px;
    padding: 22px;
}

[data-testid="stMetricLabel"] p {
    color: #CBD5E1 !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

/* CAPTIONS */

[data-testid="stCaptionContainer"] p {
    color: #B8C8DA !important;
}

/* RESULT BANNERS */

.result-banner {
    border-radius: 14px;
    padding: 22px 24px;
    margin-top: 1rem;
    font-size: 0.95rem;
    line-height: 1.7;
}

.result-safe {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.45);
    color: #A7F3D0;
}

.result-alert {
    background: rgba(251, 113, 133, 0.12);
    border: 1px solid rgba(251, 113, 133, 0.45);
    color: #FECDD3;
}

/* FOOTER */

.footer {
    border-top: 1px solid #34465F;
    margin-top: 3rem;
    padding-top: 1.6rem;
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    color: #B8C8DA;
    font-size: 0.82rem;
}

.footer strong {
    color: #FFFFFF;
}

/* MOBILE */

@media (max-width: 640px) {
    .hero-title {
        font-size: 2rem;
    }

    .feature-card {
        min-height: auto;
    }
}
</style>
""")


# ============================================================
# API FUNCTIONS
# ============================================================

def get_api_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )
        response.raise_for_status()
        return response.json()

    except (requests.RequestException, ValueError):
        return None


def predict_machine(payload):
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html("""
    <div class="brand">
        MLflow<span>Ops</span>
    </div>
    <div class="brand-caption">
        MACHINE INTELLIGENCE PLATFORM
    </div>
    """)

    st.markdown("### Overview")

    st.caption(
        "Predictive maintenance powered by "
        "machine learning and MLOps."
    )

    st.divider()

    st.markdown("### System Health")

    health = get_api_health()

    if health and health.get("model_loaded"):

        st.html("""
        <div class="status-online">
            ● API CONNECTED
        </div>
        """)

        st.caption(
            f"Active model version: "
            f"{health.get('model_version', 'Unknown')}"
        )

    else:

        st.html("""
        <div class="status-offline">
            ● API OFFLINE
        </div>
        """)

    st.divider()

    st.markdown("### Technology Stack")

    st.markdown("""
    - **XGBoost** · Machine Learning
    - **MLflow** · Experiment Tracking
    - **FastAPI** · Model Serving
    - **Streamlit** · User Interface
    - **Plotly** · Visualization
    """)

    st.divider()

    st.caption("MLflowOps v1.0.0")
    st.caption("Developed by Youssef MOUHI")


# ============================================================
# HERO SECTION
# ============================================================

st.html("""
<div class="eyebrow">
    INTELLIGENT INDUSTRIAL ANALYTICS
</div>

<div class="hero-title">
    Predict failures.<br>
    <span>Make smarter decisions.</span>
</div>

<div class="hero-subtitle">
    An end-to-end machine learning platform for
    predictive maintenance. Analyze operating conditions,
    estimate machine failure probabilities, and explore
    model-driven insights through an integrated MLOps workflow.
</div>
""")

st.write("")

st.info(
    "Portfolio demonstration using the synthetic AI4I 2020 "
    "Predictive Maintenance dataset. Not intended for "
    "operational safety decisions."
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.html("""
    <div class="feature-card">
        <div class="feature-card-title">
            ◈ Predictive Analytics
        </div>
        <div class="feature-card-text">
            Estimate failure probability from
            machine operating parameters.
        </div>
    </div>
    """)

with col2:

    st.html("""
    <div class="feature-card">
        <div class="feature-card-title">
            ◈ Model Lifecycle
        </div>
        <div class="feature-card-text">
            MLflow-backed experiment tracking,
            model registration, and versioning.
        </div>
    </div>
    """)

with col3:

    st.html("""
    <div class="feature-card">
        <div class="feature-card-title">
            ◈ API Integration
        </div>
        <div class="feature-card-text">
            Real-time inference requests through
            a dedicated FastAPI service.
        </div>
    </div>
    """)


# ============================================================
# MACHINE INPUT FORM
# ============================================================

st.html("""
<div class="section-heading">
    01 / Machine Configuration
</div>
""")

st.caption(
    "Configure the operating conditions to generate "
    "a machine failure prediction."
)

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        machine_type = st.selectbox(
            "Machine Quality Type",
            ["L", "M", "H"],
            index=1,
        )

        air_temperature = st.number_input(
            "Air Temperature (K)",
            min_value=250.0,
            max_value=350.0,
            value=298.1,
            step=0.1,
        )

    with col2:

        process_temperature = st.number_input(
            "Process Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=308.6,
            step=0.1,
        )

        rotational_speed = st.number_input(
            "Rotational Speed (RPM)",
            min_value=0,
            max_value=10000,
            value=1551,
            step=1,
        )

    with col3:

        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=200.0,
            value=42.8,
            step=0.1,
        )

        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            max_value=1000,
            value=0,
            step=1,
        )

    st.write("")

    submitted = st.form_submit_button(
        "Run Predictive Analysis →",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

if submitted:

    payload = {
        "machine_type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }

    try:

        with st.spinner("Running predictive analysis..."):
            result = predict_machine(payload)

        st.session_state["prediction_result"] = result
        st.session_state["prediction_input"] = payload

    except (requests.RequestException, ValueError) as error:

        st.error(
            "Prediction request failed. "
            f"Check that FastAPI is running. Details: {error}"
        )


# ============================================================
# PREDICTION RESULTS
# ============================================================

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    probability = float(result["failure_probability"])
    threshold = float(result["decision_threshold"])
    prediction = int(result["prediction"])

    st.html("""
    <div class="section-heading">
        02 / Prediction Intelligence
    </div>
    """)

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Failure Probability",
        f"{probability * 100:.2f}%",
    )

    metric2.metric(
        "Decision Threshold",
        f"{threshold * 100:.2f}%",
    )

    metric3.metric(
        "Risk Category",
        result["risk_level"],
    )

    metric4.metric(
        "Model Version",
        result["model_version"],
    )

    st.write("")

    chart_col, detail_col = st.columns([1.4, 1])

    with chart_col:

        st.markdown("#### Failure Probability")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={
                    "suffix": "%",
                    "font": {
                        "size": 40,
                        "color": "#FFFFFF",
                    },
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#CBD5E1",
                        "tickfont": {
                            "color": "#E2E8F0"
                        },
                    },
                    "bar": {
                        "color": "#2DD4BF",
                    },
                    "bgcolor": "#142238",
                    "steps": [
                        {
                            "range": [0, threshold * 100],
                            "color": "#12382F",
                        },
                        {
                            "range": [threshold * 100, 100],
                            "color": "#42212E",
                        },
                    ],
                    "threshold": {
                        "line": {
                            "color": "#FFFFFF",
                            "width": 4,
                        },
                        "value": threshold * 100,
                    },
                },
            )
        )

        fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#FFFFFF"},
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with detail_col:

        st.markdown("#### Model Decision")

        if prediction == 1:

            st.html("""
            <div class="result-banner result-alert">
                <strong>Potential Failure Detected</strong>
                <br><br>
                The estimated failure probability
                exceeds the configured decision threshold.
            </div>
            """)

        else:

            st.html("""
            <div class="result-banner result-safe">
                <strong>No Failure Predicted</strong>
                <br><br>
                The estimated failure probability
                is below the configured decision threshold.
            </div>
            """)

        st.write("")

        st.caption(
            f"Model: {result['model_name']}"
        )

        st.caption(
            f"Generated: {result.get('timestamp', 'N/A')}"
        )

    with st.expander("View Prediction JSON"):
        st.json(result)

    with st.expander("View Input Parameters"):
        st.json(st.session_state["prediction_input"])



# ============================================================
# LIVE SYSTEM MONITORING
# ============================================================

st.html("""
<div class="section-heading">
    03 / Live System Monitoring
</div>
""")

st.caption(
    "Operational metrics collected from the FastAPI "
    "Prometheus instrumentation."
)

if st.button("Refresh Monitoring Metrics", key="refresh_monitoring"):
    st.rerun()


def get_monitoring_metrics():
    response = requests.get(
        f"{API_URL}/metrics",
        timeout=5,
    )
    response.raise_for_status()
    return response.text


def parse_prometheus_metrics(raw_text):
    metrics = {}

    for line in raw_text.splitlines():
        if not line or line.startswith("#"):
            continue

        try:
            name_and_labels, value = line.rsplit(" ", 1)
            metrics[name_and_labels] = float(value)
        except (ValueError, IndexError):
            continue

    return metrics


try:
    raw_metrics = get_monitoring_metrics()
    metrics = parse_prometheus_metrics(raw_metrics)

    total_predictions = metrics.get(
        "mlflowops_predictions_total", 0
    )

    latency_sum = metrics.get(
        "mlflowops_prediction_latency_seconds_sum", 0
    )

    latency_count = metrics.get(
        "mlflowops_prediction_latency_seconds_count", 0
    )

    total_errors = metrics.get(
        "mlflowops_prediction_errors_total", 0
    )

    average_latency_ms = (
        (latency_sum / latency_count) * 1000
        if latency_count > 0
        else 0
    )

    total_requests = total_predictions + total_errors

    error_rate = (
        (total_errors / total_requests) * 100
        if total_requests > 0
        else 0
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Total Predictions",
        f"{int(total_predictions):,}",
    )

    metric2.metric(
        "Average Latency",
        f"{average_latency_ms:.2f} ms",
    )

    metric3.metric(
        "Prediction Errors",
        f"{int(total_errors):,}",
    )

    metric4.metric(
        "Error Rate",
        f"{error_rate:.2f}%",
    )

    st.write("")
    st.markdown("#### Prediction Risk Distribution")

    risk_counts = {}

    for name, value in metrics.items():
        if name.startswith("mlflowops_prediction_risk_total{"):
            for risk in ("LOW", "MODERATE", "HIGH"):
                if f'risk_level="{risk}"' in name:
                    risk_counts[risk] = value

    if risk_counts:
        fig = go.Figure(
            go.Bar(
                x=list(risk_counts.keys()),
                y=list(risk_counts.values()),
                marker_color=[
                    {
                        "LOW": "#14B8A6",
                        "MODERATE": "#F59E0B",
                        "HIGH": "#FB7185",
                    }.get(risk, "#64748B")
                    for risk in risk_counts
                ],
            )
        )

        fig.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#FFFFFF"},
            xaxis_title="Risk Category",
            yaxis_title="Predictions",
            margin=dict(l=20, r=20, t=20, b=40),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )
    else:
        st.info(
            "No risk distribution data yet. "
            "Run a prediction to populate the chart."
        )

    st.caption(
        "Metrics are collected from the running API instance. "
        "Counters reset when the API process restarts."
    )

except (requests.RequestException, ValueError) as error:
    st.warning(
        "Monitoring metrics are temporarily unavailable. "
        f"Details: {error}"
    )


# ============================================================
# COPYRIGHT FOOTER
# ============================================================

st.html(f"""
<div class="footer">
    <div>
        © {datetime.now().year}
        <strong>Youssef MOUHI</strong>.
        All rights reserved.
    </div>

    <div>
        MLflowOps · Predictive Maintenance Intelligence
    </div>

    <div>
        Built with Python · MLflow · XGBoost · FastAPI
    </div>
</div>
""")