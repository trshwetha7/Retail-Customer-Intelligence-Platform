from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.shared import artifacts_ready, load_metrics, load_models, load_tables, render_missing_artifacts_message


st.title("Future Customer Value")
if not artifacts_ready():
    render_missing_artifacts_message()
    st.stop()

tables = load_tables()
models = load_models()
metrics = load_metrics()["future_value"]

future_value_artifacts = models["future_value"]["artifacts"]
predictions = future_value_artifacts.predictions

st.write(future_value_artifacts.recommendation)
st.dataframe(metrics, use_container_width=True)

selected_model = st.selectbox("Prediction set", predictions["model_name"].unique())
pred_slice = predictions.loc[predictions["model_name"] == selected_model]
plot_y = "score" if "score" in pred_slice.columns else "prediction"
scatter = px.scatter(
    pred_slice,
    x="actual",
    y=plot_y,
    title=f"Actual vs modeled outcome: {selected_model}",
)
st.plotly_chart(scatter, use_container_width=True)

top_customers = (
    pred_slice.sort_values(plot_y, ascending=False)
    .head(25)
    .merge(
        tables["customer_features"][["customer_id", "primary_country", "spend_total", "order_count", "recency_days"]],
        on="customer_id",
        how="left",
    )
)
st.subheader("Highest-ranked customers in the holdout snapshot")
st.dataframe(top_customers, use_container_width=True)
