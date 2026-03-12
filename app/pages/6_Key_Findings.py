from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.shared import artifacts_ready, load_metrics, load_tables, render_missing_artifacts_message


st.title("Key Findings")
if not artifacts_ready():
    render_missing_artifacts_message()
    st.stop()

tables = load_tables()
metrics = load_metrics()
segments = tables["segment_summary"].sort_values("spend_total", ascending=False)
retention = tables["cohort_retention"]
monthly_demand = tables["monthly_demand"]

top_segment = segments.iloc[0]
recent_retention = retention.loc[retention["cohort_index"].between(1, 3)].groupby("cohort_month")["retention_rate"].mean().sort_values(ascending=False)
top_category = monthly_demand.groupby("category")["revenue"].sum().sort_values(ascending=False).index[0]
best_recommender = metrics["recommender"].sort_values("precision_at_k", ascending=False).iloc[0]

st.markdown(
    f"""
    - The highest-value segment is **{top_segment['segment_name']}**, combining strong median spend with comparatively fresh activity.
    - Early lifecycle retention differs meaningfully across cohorts; the strongest recent cohort by 1-3 month retention is **{recent_retention.index[0]}**.
    - The primary future value task is framed around **90-day customer value**, giving the business a near-term planning horizon for prioritization.
    - The top revenue-driving product family in the prepared data is **{top_category}**, making it a useful anchor for demand monitoring.
    - In offline recommendation checks, **{best_recommender['model_name']}** delivered the best precision at 10 among the implemented approaches.
    """
)

st.subheader("What to do with these outputs")
st.markdown(
    """
    - Use segment definitions to tailor customer treatment instead of treating the entire base as homogeneous.
    - Focus retention interventions on weaker cohorts early, when drop-off is most visible.
    - Route predicted high-value returners into more personalized product journeys.
    - Monitor leading product families for month-over-month shifts and seasonal peaks.
    """
)
