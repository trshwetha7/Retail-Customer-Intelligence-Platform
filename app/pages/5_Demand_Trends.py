from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.shared import artifacts_ready, load_tables, render_missing_artifacts_message
from src.models.demand import forecast_category_demand, purchase_timing_profile, top_category_shifts


st.title("Demand Trends")
if not artifacts_ready():
    render_missing_artifacts_message()
    st.stop()

tables = load_tables()
monthly_demand = tables["monthly_demand"]
transactions = tables["transactions"]

top_categories = (
    monthly_demand.groupby("category")["revenue"].sum().sort_values(ascending=False).head(8).index.tolist()
)
selected_categories = st.multiselect("Product families", options=top_categories, default=top_categories[:4])
filtered = monthly_demand.loc[monthly_demand["category"].isin(selected_categories)]
line = px.line(filtered, x="month", y="revenue", color="category", markers=True, title="Product-family demand over time")
st.plotly_chart(line, use_container_width=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Recent product-family shifts")
    st.dataframe(top_category_shifts(monthly_demand), use_container_width=True)
with col2:
    st.subheader("Purchase timing")
    timing = purchase_timing_profile(transactions)
    heatmap = px.density_heatmap(timing, x="month_name", y="weekday", z="revenue", histfunc="sum", title="Revenue heatmap by weekday and month")
    st.plotly_chart(heatmap, use_container_width=True)

forecast_category = st.selectbox("Forecast product family", top_categories)
forecast = forecast_category_demand(monthly_demand, forecast_category)
if not forecast.empty:
    history = monthly_demand.loc[monthly_demand["category"] == forecast_category, ["month", "revenue"]].copy()
    history["series"] = "History"
    projected = forecast.rename(columns={"forecast_revenue": "revenue"})
    projected["series"] = "Forecast"
    combined = px.line(
        pd.concat([history, projected], ignore_index=True),
        x="month",
        y="revenue",
        color="series",
        markers=True,
        title=f"Short-horizon demand view for {forecast_category}",
    )
    st.plotly_chart(combined, use_container_width=True)
