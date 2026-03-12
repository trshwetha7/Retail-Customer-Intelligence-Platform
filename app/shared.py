from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ARTIFACT_PATHS
from src.models.cohort import build_retention_matrix


def artifacts_ready() -> bool:
    required = [
        ARTIFACT_PATHS["transactions_enriched"],
        ARTIFACT_PATHS["customer_features"],
        ARTIFACT_PATHS["cohort_retention"],
        ARTIFACT_PATHS["monthly_demand"],
        ARTIFACT_PATHS["segmentation_model"],
        ARTIFACT_PATHS["future_value_model"],
        ARTIFACT_PATHS["recommender_model"],
    ]
    return all(path.exists() for path in required)


def render_missing_artifacts_message() -> None:
    st.warning(
        "Prepared artifacts are not available yet. Place the UCI Online Retail file under `data/raw/` "
        "and run `python scripts/train_models.py` from the repository root."
    )


@st.cache_data(show_spinner=False)
def load_tables() -> dict:
    return {
        "transactions": pd.read_parquet(ARTIFACT_PATHS["transactions_enriched"]),
        "customer_features": pd.read_parquet(ARTIFACT_PATHS["customer_features"]),
        "cohort_retention": pd.read_parquet(ARTIFACT_PATHS["cohort_retention"]),
        "cohort_revenue": pd.read_parquet(ARTIFACT_PATHS["cohort_revenue"]),
        "monthly_demand": pd.read_parquet(ARTIFACT_PATHS["monthly_demand"]),
        "segment_summary": pd.read_parquet(ARTIFACT_PATHS["segment_summary"]),
        "customer_segments": pd.read_parquet(ARTIFACT_PATHS["segment_summary"].with_name("customer_segments.parquet")),
        "snapshots": pd.read_parquet(ARTIFACT_PATHS["snapshot_features"]),
    }


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    return {
        "segmentation": joblib.load(ARTIFACT_PATHS["segmentation_model"]),
        "future_value": joblib.load(ARTIFACT_PATHS["future_value_model"]),
        "recommender": joblib.load(ARTIFACT_PATHS["recommender_model"]),
    }


def load_metrics() -> dict:
    future_value_metrics = json.loads(ARTIFACT_PATHS["future_value_metrics"].read_text())
    recommender_metrics = json.loads(ARTIFACT_PATHS["recommender_metrics"].read_text())
    return {
        "future_value": pd.DataFrame(future_value_metrics),
        "recommender": pd.DataFrame(recommender_metrics),
    }


def load_retention_matrix() -> pd.DataFrame:
    tables = load_tables()
    return build_retention_matrix(tables["cohort_retention"])
