from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import ARTIFACT_PATHS, DEFAULT_CUSTOMER_SAMPLE_FRAC, ensure_directories
from src.data.io import load_dataset, raw_data_available
from src.data.preprocess import clean_transactions
from src.features.customer_features import attach_customer_attributes, build_customer_features
from src.models.cohort import build_cohort_frames
from src.models.demand import build_monthly_demand
from src.models.future_value import build_snapshot_dataset, train_future_value_models
from src.models.recommender import train_recommender
from src.models.segmentation import compare_segmentation_models, fit_segmentation_model


def prepare_data_assets(sample_frac: float = DEFAULT_CUSTOMER_SAMPLE_FRAC) -> dict:
    ensure_directories()
    if not raw_data_available():
        raise FileNotFoundError("Raw UCI Online Retail data is missing under data/raw/.")

    bundle = load_dataset(sample_frac=sample_frac)
    transactions = clean_transactions(bundle.transactions)
    customer_features = attach_customer_attributes(build_customer_features(transactions), transactions)
    retained, revenue = build_cohort_frames(transactions)
    monthly_demand = build_monthly_demand(transactions)
    product_lookup = transactions[["product_id", "product_name", "product_family"]].drop_duplicates("product_id")

    transactions.to_parquet(ARTIFACT_PATHS["transactions_enriched"], index=False)
    customer_features.to_parquet(ARTIFACT_PATHS["customer_features"], index=False)
    retained.to_parquet(ARTIFACT_PATHS["cohort_retention"], index=False)
    revenue.to_parquet(ARTIFACT_PATHS["cohort_revenue"], index=False)
    monthly_demand.to_parquet(ARTIFACT_PATHS["monthly_demand"], index=False)

    metadata_path = Path(ARTIFACT_PATHS["transactions_enriched"]).with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(bundle.metadata, indent=2))
    return {
        "transactions": transactions,
        "customer_features": customer_features,
        "cohort_retention": retained,
        "cohort_revenue": revenue,
        "monthly_demand": monthly_demand,
        "product_lookup": product_lookup,
    }


def load_prepared_assets() -> dict:
    return {
        "transactions": pd.read_parquet(ARTIFACT_PATHS["transactions_enriched"]),
        "customer_features": pd.read_parquet(ARTIFACT_PATHS["customer_features"]),
        "cohort_retention": pd.read_parquet(ARTIFACT_PATHS["cohort_retention"]),
        "cohort_revenue": pd.read_parquet(ARTIFACT_PATHS["cohort_revenue"]),
        "monthly_demand": pd.read_parquet(ARTIFACT_PATHS["monthly_demand"]),
    }


def train_model_artifacts(sample_frac: float = DEFAULT_CUSTOMER_SAMPLE_FRAC) -> dict:
    prepared = prepare_data_assets(sample_frac=sample_frac)
    transactions = prepared["transactions"]
    customer_features = prepared["customer_features"]
    product_lookup = prepared["product_lookup"]

    comparison = compare_segmentation_models(customer_features)
    best_row = comparison.sort_values(["silhouette", "bic"], ascending=[False, True]).iloc[0]
    segmentation = fit_segmentation_model(
        customer_features=customer_features,
        algorithm=str(best_row["algorithm"]),
        n_clusters=int(best_row["n_clusters"]),
    )
    segmentation.customer_segments.to_parquet(
        ARTIFACT_PATHS["segment_summary"].with_name("customer_segments.parquet"),
        index=False,
    )
    segmentation.segment_summary.to_parquet(ARTIFACT_PATHS["segment_summary"], index=False)
    joblib.dump(segmentation, ARTIFACT_PATHS["segmentation_model"])

    snapshots = build_snapshot_dataset(transactions)
    snapshots.to_parquet(ARTIFACT_PATHS["snapshot_features"], index=False)
    future_value = train_future_value_models(snapshots)
    best_future_model_name = future_value.metrics.iloc[0]["model_name"]
    joblib.dump(
        {
            "artifacts": future_value,
            "best_model_name": best_future_model_name,
            "best_model": future_value.models[best_future_model_name],
        },
        ARTIFACT_PATHS["future_value_model"],
    )
    ARTIFACT_PATHS["future_value_metrics"].write_text(
        json.dumps(future_value.metrics.to_dict(orient="records"), indent=2)
    )

    recommender = train_recommender(transactions, product_lookup)
    joblib.dump(recommender, ARTIFACT_PATHS["recommender_model"])
    ARTIFACT_PATHS["recommender_metrics"].write_text(
        json.dumps(recommender.metrics.to_dict(orient="records"), indent=2)
    )

    return {
        "segmentation": segmentation,
        "future_value": future_value,
        "recommender": recommender,
        "comparison": comparison,
        "snapshots": snapshots,
    }
