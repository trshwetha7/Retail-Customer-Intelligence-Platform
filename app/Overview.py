from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.shared import artifacts_ready, load_metrics, load_tables, render_missing_artifacts_message


st.set_page_config(page_title="Retail Customer Intelligence Platform", layout="wide")
st.title("Retail Customer Intelligence Platform")
st.caption("General retail analytics and machine learning on the UCI Online Retail dataset.")

if not artifacts_ready():
    render_missing_artifacts_message()
    st.stop()

tables = load_tables()
metrics = load_metrics()
transactions = tables["transactions"]
customer_features = tables["customer_features"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{customer_features['customer_id'].nunique():,}")
col2.metric("Invoice Lines", f"{len(transactions):,}")
col3.metric("Revenue", f"{transactions['revenue'].sum():,.2f}")
col4.metric("Product Families", f"{transactions['product_family'].nunique():,}")

timeline = (
    transactions.groupby("month")
    .agg(revenue=("revenue", "sum"), customers=("customer_id", "nunique"))
    .reset_index()
)
fig = px.line(timeline, x="month", y=["revenue", "customers"], markers=True, title="Revenue and active customers over time")
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.2, 1])
with left:
    st.subheader("What the platform covers")
    st.markdown(
        """
        - Customer segmentation using recency, spend, basket depth, diversity, and seasonality
        - Cohort retention and revenue analysis from first-purchase month onward
        - 90-day future customer value modeling with baseline and non-linear models
        - Product recommendations using popularity and personalized item similarity
        - Demand monitoring for product-family shifts and seasonal patterns
        """
    )

with right:
    st.subheader("Model scorecards")
    st.write("Future customer value")
    st.dataframe(metrics["future_value"], use_container_width=True)
    st.write("Recommendations")
    st.dataframe(metrics["recommender"], use_container_width=True)
