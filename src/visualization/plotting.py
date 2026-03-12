from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_segment_scatter(customer_segments: pd.DataFrame):
    return px.scatter(
        customer_segments,
        x="segment_pca_x",
        y="segment_pca_y",
        color="segment_name",
        hover_data=["customer_id", "spend_total", "order_count", "recency_days"],
        title="Customer segments projected into two dimensions",
    )


def plot_retention_heatmap(retention_df: pd.DataFrame):
    return px.imshow(
        retention_df,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        labels={"x": "Months Since First Purchase", "y": "Cohort Month", "color": "Retention"},
        title="Monthly customer retention heatmap",
    )


def plot_monthly_demand(monthly_demand: pd.DataFrame):
    return px.line(
        monthly_demand,
        x="month",
        y="revenue",
        color="category",
        title="Monthly product-family demand",
    )
