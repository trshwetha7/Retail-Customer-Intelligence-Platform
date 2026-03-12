from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.shared import artifacts_ready, load_tables, render_missing_artifacts_message


st.title("Customer Segmentation")
if not artifacts_ready():
    render_missing_artifacts_message()
    st.stop()

tables = load_tables()
segments = tables["customer_segments"]
summary = tables["segment_summary"]

selected_segments = st.multiselect(
    "Filter segments",
    options=sorted(segments["segment_name"].dropna().unique()),
    default=sorted(segments["segment_name"].dropna().unique()),
)
filtered = segments.loc[segments["segment_name"].isin(selected_segments)].copy()

fig = px.scatter(
    filtered,
    x="segment_pca_x",
    y="segment_pca_y",
    color="segment_name",
    size="spend_total",
    hover_data=["customer_id", "order_count", "avg_order_value", "recency_days", "primary_country"],
    title="Segment map",
)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Segment summary")
    st.dataframe(summary, use_container_width=True)
with col2:
    metric = st.selectbox("Profile metric", ["spend_total", "order_count", "avg_order_value", "recency_days"])
    bars = px.bar(summary.sort_values(metric, ascending=False), x="segment_name", y=metric, color="segment_name", title=f"Median {metric} by segment")
    st.plotly_chart(bars, use_container_width=True)

st.subheader("Sample customers")
sample_size = min(20, len(filtered))
st.dataframe(
    filtered[["customer_id", "segment_name", "primary_country", "spend_total", "order_count", "avg_order_value", "recency_days"]]
    .sort_values(["segment_name", "spend_total"], ascending=[True, False])
    .head(sample_size),
    use_container_width=True,
)
