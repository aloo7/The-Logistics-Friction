from __future__ import annotations

from typing import Any

import streamlit as st

from model_contract import FEATURE_CONTRACT, MODEL_ARTIFACT_PATH, predict_delay


APP_TITLE = "Supply Chain Delay Oracle"
APP_SUBTITLE = "Checkpoint demo using the saved logistic regression baseline artifact."

RISK_BANDS = (
    (0.40, "Low", "Standard fulfilment and routine customer messaging."),
    (0.65, "Medium", "Monitor closely and prepare a proactive customer update."),
    (1.01, "High", "Escalate to operations, prioritize intervention, and contact the customer early."),
)

DAY_LABELS = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

DEMO_SCENARIOS = {
    "Routine order": {
        "approval_delay": 0.0,
        "estimated_delivery_time": 30.0,
        "purchase_day_of_week": 1,
        "purchase_hour": 8,
        "total_items": 4.0,
        "total_price": 220.0,
        "total_freight_value": 5.0,
    },
    "Borderline watchlist": {
        "approval_delay": 1.0,
        "estimated_delivery_time": 18.0,
        "purchase_day_of_week": 2,
        "purchase_hour": 12,
        "total_items": 2.0,
        "total_price": 90.0,
        "total_freight_value": 18.0,
    },
    "Approval bottleneck": {
        "approval_delay": 1.0,
        "estimated_delivery_time": 14.0,
        "purchase_day_of_week": 4,
        "purchase_hour": 15,
        "total_items": 1.0,
        "total_price": 65.0,
        "total_freight_value": 25.0,
    },
    "Weekend escalation": {
        "approval_delay": 3.0,
        "estimated_delivery_time": 7.0,
        "purchase_day_of_week": 5,
        "purchase_hour": 20,
        "total_items": 1.0,
        "total_price": 30.0,
        "total_freight_value": 45.0,
    },
    "Severe friction": {
        "approval_delay": 5.0,
        "estimated_delivery_time": 5.0,
        "purchase_day_of_week": 6,
        "purchase_hour": 22,
        "total_items": 1.0,
        "total_price": 20.0,
        "total_freight_value": 60.0,
    },
}


def apply_scenario(name: str) -> None:
    for feature_name, value in DEMO_SCENARIOS[name].items():
        st.session_state[feature_name] = value


def initialize_state() -> None:
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "prediction_error" not in st.session_state:
        st.session_state.prediction_error = None
    if "approval_delay" not in st.session_state:
        apply_scenario("Routine order")


def get_risk_band(delay_probability: float) -> tuple[str, str]:
    for threshold, label, action in RISK_BANDS:
        if delay_probability < threshold:
            return label, action
    return "High", RISK_BANDS[-1][2]


def render_prediction(result: dict[str, Any]) -> None:
    delay_probability = float(result["delay_probability"])
    predicted_class = "Delayed" if int(result["is_delayed"]) == 1 else "Not delayed"
    risk_band, action = get_risk_band(delay_probability)

    st.subheader("Prediction")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted class", predicted_class)
    col2.metric("Delay probability", f"{delay_probability:.1%}")
    col3.metric("Risk band", risk_band)
    st.info(f"Recommended action: {action}")


st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")
initialize_state()

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f5ef 0%, #f2efe6 100%);
        color: #14213d;
    }
    .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(20, 33, 61, 0.08);
        border-radius: 18px;
        padding: 0.8rem 1rem;
        box-shadow: 0 10px 30px rgba(20, 33, 61, 0.05);
    }
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(20, 33, 61, 0.08);
        border-radius: 24px;
        padding: 1.1rem 1.15rem 0.6rem 1.15rem;
        box-shadow: 0 16px 40px rgba(20, 33, 61, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

model_exists = MODEL_ARTIFACT_PATH.exists()
if not model_exists:
    st.error(
        f"Model artifact missing: {MODEL_ARTIFACT_PATH}. "
        "Train it first with `python3 train_baseline_model.py`."
    )
else:
    st.success(f"Using saved model artifact: {MODEL_ARTIFACT_PATH.name}")

st.subheader("Demo scenarios")
scenario_columns = st.columns(len(DEMO_SCENARIOS))
for column, scenario_name in zip(scenario_columns, DEMO_SCENARIOS):
    if column.button(scenario_name, use_container_width=True):
        apply_scenario(scenario_name)
        st.session_state.prediction_result = None
        st.session_state.prediction_error = None

with st.form("delay_demo_form"):
    left, right = st.columns(2)
    with left:
        st.number_input(
            "Approval delay (days)",
            min_value=0.0,
            max_value=30.0,
            step=1.0,
            key="approval_delay",
            help=FEATURE_CONTRACT["approval_delay"],
        )
        st.number_input(
            "Estimated delivery time (days)",
            min_value=1.0,
            max_value=60.0,
            step=1.0,
            key="estimated_delivery_time",
            help=FEATURE_CONTRACT["estimated_delivery_time"],
        )
        st.selectbox(
            "Purchase day of week",
            options=list(DAY_LABELS),
            format_func=lambda value: DAY_LABELS[value],
            key="purchase_day_of_week",
            help=FEATURE_CONTRACT["purchase_day_of_week"],
        )
        st.slider(
            "Purchase hour",
            min_value=0,
            max_value=23,
            step=1,
            key="purchase_hour",
            help=FEATURE_CONTRACT["purchase_hour"],
        )
    with right:
        st.number_input(
            "Total items",
            min_value=1.0,
            max_value=25.0,
            step=1.0,
            key="total_items",
            help=FEATURE_CONTRACT["total_items"],
        )
        st.number_input(
            "Total price",
            min_value=0.0,
            max_value=5000.0,
            step=10.0,
            key="total_price",
            help=FEATURE_CONTRACT["total_price"],
        )
        st.number_input(
            "Total freight value",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            key="total_freight_value",
            help=FEATURE_CONTRACT["total_freight_value"],
        )
        st.markdown("")
        st.markdown("")
        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True,
            type="primary",
            disabled=not model_exists,
        )

if submitted and model_exists:
    try:
        payload = {
            "approval_delay": float(st.session_state.approval_delay),
            "estimated_delivery_time": float(st.session_state.estimated_delivery_time),
            "purchase_day_of_week": int(st.session_state.purchase_day_of_week),
            "purchase_hour": int(st.session_state.purchase_hour),
            "total_items": float(st.session_state.total_items),
            "total_price": float(st.session_state.total_price),
            "total_freight_value": float(st.session_state.total_freight_value),
        }
        st.session_state.prediction_result = predict_delay(payload, model_path=MODEL_ARTIFACT_PATH)
        st.session_state.prediction_error = None
    except (FileNotFoundError, ValueError) as exc:
        st.session_state.prediction_result = None
        st.session_state.prediction_error = str(exc)
    except Exception as exc:  # pragma: no cover - presentation fallback
        st.session_state.prediction_result = None
        st.session_state.prediction_error = f"Unexpected prediction error: {exc}"

if st.session_state.prediction_error:
    st.error(st.session_state.prediction_error)
elif st.session_state.prediction_result:
    render_prediction(st.session_state.prediction_result)
