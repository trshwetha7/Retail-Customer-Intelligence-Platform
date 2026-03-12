from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import NOTEBOOKS_DIR, ensure_directories


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            """# Retail Customer Intelligence Platform

This notebook develops a business-facing analytics and machine learning workflow on the UCI Online Retail dataset. The objective is to move from raw transaction records to practical decisions around customer segmentation, retention, future customer value, personalization, and demand monitoring.

## Analytical questions
- Which customer groups behave differently enough to warrant distinct lifecycle strategies?
- How does retention evolve by acquisition cohort, and which cohorts generate stronger revenue over time?
- Which customers are most likely to create value over the next 90 days?
- How much lift does a simple personalized recommender provide over a popularity baseline?
- What product-family and seasonality patterns matter most for planning?
"""
        ),
        markdown_cell(
            """## Notebook design

The notebook intentionally uses reusable modules from `src/` rather than embedding all logic inline. That keeps the analytical narrative readable and makes the Streamlit application consume the same prepared data and trained artifacts.

### Data placement
Place the UCI Online Retail file under `data/raw/`. The loader searches recursively for common filenames such as:

- `Online Retail.xlsx`
- `online_retail.xlsx`
- `Online Retail.csv`
- `online_retail.csv`

The platform standardizes that file into a reusable retail schema with orders, products, customers, quantities, revenue, countries, and derived product families.
"""
        ),
        code_cell(
            """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

ROOT = Path.cwd().resolve().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CUSTOMER_SAMPLE_FRAC
from src.models.cohort import build_retention_matrix
from src.models.demand import forecast_category_demand, top_category_shifts
from src.models.future_value import build_snapshot_dataset, choose_target_definition, train_future_value_models
from src.models.recommender import recommend_personalized, recommend_popular, train_recommender
from src.models.segmentation import compare_segmentation_models, fit_segmentation_model
from src.pipeline import prepare_data_assets

plt.style.use("ggplot")
pd.set_option("display.max_columns", 120)
"""
        ),
        code_cell(
            """prepared = prepare_data_assets(sample_frac=DEFAULT_CUSTOMER_SAMPLE_FRAC)
transactions = prepared["transactions"]
customer_features = prepared["customer_features"]
cohort_retention = prepared["cohort_retention"]
cohort_revenue = prepared["cohort_revenue"]
monthly_demand = prepared["monthly_demand"]
product_lookup = prepared["product_lookup"]

print(f"Transactions loaded: {len(transactions):,}")
print(f"Customers in feature table: {customer_features['customer_id'].nunique():,}")
print(f"Date range: {transactions['t_dat'].min().date()} to {transactions['t_dat'].max().date()}")
"""
        ),
        markdown_cell(
            """## 1. Data quality and retail context

The UCI Online Retail dataset offers invoice-level purchase history with customer identifiers, product descriptions, quantities, unit prices, and country. It does not include marketing exposure, customer demographics, product hierarchy, or inventory state, so this project focuses on behavioral and transaction-derived intelligence rather than causal attribution.
"""
        ),
        code_cell(
            """quality_summary = pd.DataFrame(
    {
        "rows": [len(transactions), len(customer_features), len(product_lookup)],
        "columns": [transactions.shape[1], customer_features.shape[1], product_lookup.shape[1]],
    },
    index=["transactions", "customer_features", "product_lookup"],
)
quality_summary
"""
        ),
        code_cell(
            """missingness = (
    transactions[["product_name", "product_family", "country", "unit_price", "quantity", "revenue"]]
    .isna()
    .mean()
    .sort_values(ascending=False)
    .rename("missing_share")
    .to_frame()
)
missingness
"""
        ),
        code_cell(
            """monthly_overview = (
    transactions.groupby("month")
    .agg(revenue=("revenue", "sum"), customers=("customer_id", "nunique"), units=("quantity", "sum"))
    .reset_index()
)
px.line(monthly_overview, x="month", y=["revenue", "customers"], markers=True, title="Revenue and active customers over time")
"""
        ),
        code_cell(
            """top_families = (
    transactions.groupby("product_family")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(12)
    .rename("revenue")
    .reset_index()
)
px.bar(top_families, x="product_family", y="revenue", title="Top product families by revenue")
"""
        ),
        markdown_cell(
            """## 2. Customer feature engineering

The segmentation and predictive modules use customer-level features built from transaction history:

- recency and tenure
- order, line, and unit frequency
- monetary value and basket depth
- product-family diversity
- repeat purchase behavior
- mean interpurchase time
- spend concentration across derived product families
- country concentration and seasonal concentration

These features are designed to reflect retail behavior rather than only generic machine learning convenience.
"""
        ),
        code_cell(
            """customer_features.describe(include="all").T.head(25)
"""
        ),
        markdown_cell(
            """## 3. Customer segmentation

The first objective is to identify behaviorally distinct customer segments that can be interpreted in business terms. I compare KMeans and Gaussian Mixture variants across a small cluster range, then inspect the most interpretable solution.
"""
        ),
        code_cell(
            """segment_comparison = compare_segmentation_models(customer_features)
segment_comparison
"""
        ),
        code_cell(
            """best_segment_row = segment_comparison.sort_values(["silhouette", "bic"], ascending=[False, True]).iloc[0]
segmentation = fit_segmentation_model(
    customer_features,
    algorithm=best_segment_row["algorithm"],
    n_clusters=int(best_segment_row["n_clusters"]),
)
segmentation.segment_summary
"""
        ),
        code_cell(
            """px.scatter(
    segmentation.customer_segments,
    x="segment_pca_x",
    y="segment_pca_y",
    color="segment_name",
    size="spend_total",
    hover_data=["customer_id", "order_count", "avg_order_value", "recency_days", "primary_country"],
    title="Customer segment map",
)
"""
        ),
        markdown_cell(
            """The segment labels are intentionally business-friendly rather than abstract cluster IDs. The useful output is not the cluster itself, but the fact that each segment implies a different action: protect loyal accounts, reactivate quieter buyers, or treat seasonal buyers with more precise timing.
"""
        ),
        markdown_cell(
            """## 4. Cohort and retention analysis

Cohort analysis reframes the customer base by first-purchase month. This highlights how quickly customers return after acquisition and whether some cohorts monetise more effectively over time.
"""
        ),
        code_cell(
            """retention_matrix = build_retention_matrix(cohort_retention)
retention_matrix.head()
"""
        ),
        code_cell(
            """px.imshow(
    retention_matrix,
    aspect="auto",
    color_continuous_scale="YlGnBu",
    labels={"x": "Months since first purchase", "y": "Cohort month", "color": "Retention"},
    title="Monthly retention heatmap",
)
"""
        ),
        code_cell(
            """cohort_revenue["cohort_month"] = cohort_revenue["cohort_month"].astype(str)
selected_cohorts = cohort_revenue["cohort_month"].drop_duplicates().sort_values().tail(6)
px.line(
    cohort_revenue.loc[cohort_revenue["cohort_month"].isin(selected_cohorts)],
    x="cohort_index",
    y="revenue",
    color="cohort_month",
    markers=True,
    title="Revenue curves for recent cohorts",
)
"""
        ),
        markdown_cell(
            """## 5. Future customer value modeling

The platform uses rolling customer snapshots to inspect label density before choosing the primary target. When near-term spend is dense enough, the preferred target is 90-day future spend. If not, the pipeline falls back to 90-day repeat purchase propensity.
"""
        ),
        code_cell(
            """snapshots = build_snapshot_dataset(transactions)
target_choice = choose_target_definition(snapshots)
target_choice
"""
        ),
        code_cell(
            """future_value = train_future_value_models(snapshots)
future_value.metrics
"""
        ),
        code_cell(
            """model_name = future_value.metrics.iloc[0]["model_name"]
pred_slice = future_value.predictions.loc[future_value.predictions["model_name"] == model_name]
plot_y = "score" if "score" in pred_slice.columns else "prediction"
px.scatter(
    pred_slice,
    x="actual",
    y=plot_y,
    title=f"Holdout performance for {model_name}",
)
"""
        ),
        markdown_cell(
            """## 6. Recommendation system

Recommendations are evaluated with a simple but instructive comparison:

- a popularity baseline that represents non-personalized merchandising
- a personalized item-item similarity recommender built from observed co-purchase patterns
"""
        ),
        code_cell(
            """recommender = train_recommender(transactions, product_lookup)
recommender.metrics
"""
        ),
        code_cell(
            """example_customer = list(recommender.customer_history.keys())[0]
history_items = recommender.customer_history[example_customer][-10:]
popular_recs = recommend_popular(recommender.popularity_rank, history_items, 10)
personal_recs = recommend_personalized(
    customer_id=example_customer,
    customer_history=recommender.customer_history,
    item_similarity=recommender.item_similarity,
    item_index=recommender.item_index,
    popularity_rank=recommender.popularity_rank,
    product_lookup=product_lookup,
    n_items=10,
)

display(product_lookup.loc[product_lookup["product_id"].isin(history_items)])
display(popular_recs.head(10))
display(personal_recs.head(10))
"""
        ),
        markdown_cell(
            """## 7. Demand and trend insights

At the demand layer, the goal is not full-store forecasting. Instead, the analysis tracks which derived product families are rising or fading, how strongly demand varies over time, and whether the data supports a simple short-horizon demand view for leading product families.
"""
        ),
        code_cell(
            """top_categories = monthly_demand.groupby("category")["revenue"].sum().sort_values(ascending=False).head(6).index
px.line(
    monthly_demand.loc[monthly_demand["category"].isin(top_categories)],
    x="month",
    y="revenue",
    color="category",
    markers=True,
    title="Demand trajectories for leading product families",
)
"""
        ),
        code_cell(
            """top_category_shifts(monthly_demand)
"""
        ),
        code_cell(
            """forecast_category = list(top_categories)[0]
forecast_category_demand(monthly_demand, forecast_category)
"""
        ),
        markdown_cell(
            """## 8. Final synthesis

The project now brings together customer structure, lifecycle dynamics, value prediction, personalization, and demand signals:

- Segmentation reveals whether revenue is concentrated in a loyal core or spread across emerging and seasonal buyers.
- Cohort analysis shows where post-acquisition drop-off is steepest and which cohorts preserve value over time.
- The future value model provides a prioritization layer for customer outreach.
- The recommender demonstrates the incremental value of tailoring assortments to purchase history.
- Demand analysis highlights which product families deserve closer monitoring during seasonal peaks or rapid trend shifts.
"""
        ),
        markdown_cell(
            """## Next steps

- Replace heuristic product-family derivation with a richer taxonomy if a cleaner product hierarchy becomes available.
- Add margin-aware prioritization so customer value and demand views reflect profitability, not revenue alone.
- Replace item similarity with an implicit factorization workflow for larger-scale personalization.
- Validate whether segment assignments remain stable over time or drift materially across calendar periods.
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    ensure_directories()
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    output_path = NOTEBOOKS_DIR / "retail_customer_intelligence_platform.ipynb"
    output_path.write_text(json.dumps(notebook, indent=2))
    print(f"Wrote {output_path}")
