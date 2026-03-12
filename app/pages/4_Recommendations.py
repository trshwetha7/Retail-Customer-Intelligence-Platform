from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.shared import artifacts_ready, load_metrics, load_models, load_tables, render_missing_artifacts_message
from src.models.recommender import recommend_personalized, recommend_popular


st.title("Recommendations")
if not artifacts_ready():
    render_missing_artifacts_message()
    st.stop()

tables = load_tables()
models = load_models()
metrics = load_metrics()["recommender"]
transactions = tables["transactions"]
recommender = models["recommender"]

product_lookup = (
    transactions[["product_id", "product_name", "product_family"]]
    .drop_duplicates("product_id")
    .sort_values("product_name")
)

sample_customers = sorted(list(recommender.customer_history.keys()))[:500]
customer_id = st.selectbox("Customer", sample_customers)
history_items = recommender.customer_history.get(customer_id, [])[-15:]
history_table = product_lookup.loc[product_lookup["product_id"].isin(history_items)].copy()

st.subheader("Recent purchase history")
st.dataframe(history_table, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Popularity baseline")
    popular = recommend_popular(recommender.popularity_rank, history_items, 10)
    st.dataframe(popular[["product_id", "product_name", "product_family", "purchases"]], use_container_width=True)
with col2:
    st.subheader("Personalized item similarity")
    personalized = recommend_personalized(
        customer_id=customer_id,
        customer_history=recommender.customer_history,
        item_similarity=recommender.item_similarity,
        item_index=recommender.item_index,
        popularity_rank=recommender.popularity_rank,
        product_lookup=product_lookup,
        n_items=10,
    )
    columns = [column for column in ["product_id", "product_name", "product_family", "score"] if column in personalized.columns]
    st.dataframe(personalized[columns], use_container_width=True)

st.subheader("Offline comparison")
st.dataframe(metrics, use_container_width=True)
