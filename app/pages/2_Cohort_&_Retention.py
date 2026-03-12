from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.shared import artifacts_ready, load_retention_matrix, load_tables, render_missing_artifacts_message


st.title("Cohort & Retention")
if not artifacts_ready():
    render_missing_artifacts_message()
    st.stop()

tables = load_tables()
retention_matrix = load_retention_matrix()
cohort_revenue = tables["cohort_revenue"].copy()
cohort_revenue["cohort_month"] = cohort_revenue["cohort_month"].astype(str)

heatmap = px.imshow(
    retention_matrix,
    aspect="auto",
    color_continuous_scale="YlGnBu",
    title="Monthly retention by acquisition cohort",
    labels={"x": "Months since first purchase", "y": "Cohort month", "color": "Retention"},
)
st.plotly_chart(heatmap, use_container_width=True)

cohort_selector = st.selectbox("Select cohort", sorted(cohort_revenue["cohort_month"].unique()))
cohort_slice = cohort_revenue.loc[cohort_revenue["cohort_month"] == cohort_selector]
line = px.line(cohort_slice, x="cohort_index", y="revenue", markers=True, title=f"Revenue curve for cohort {cohort_selector}")
st.plotly_chart(line, use_container_width=True)

st.subheader("Revenue by cohort")
pivot = cohort_revenue.pivot(index="cohort_month", columns="cohort_index", values="revenue").fillna(0)
st.dataframe(pivot, use_container_width=True)
