from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_RANDOM_STATE, SEGMENT_FEATURE_COLUMNS


@dataclass
class SegmentationArtifacts:
    model: object
    feature_columns: List[str]
    algorithm: str
    n_clusters: int
    scores: Dict[str, float]
    segment_summary: pd.DataFrame
    customer_segments: pd.DataFrame


def _prepare_matrix(customer_features: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    existing = [column for column in feature_columns if column in customer_features.columns]
    matrix = customer_features[existing].copy()
    return matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compare_segmentation_models(
    customer_features: pd.DataFrame,
    feature_columns: Iterable[str] = SEGMENT_FEATURE_COLUMNS,
    cluster_range: range = range(3, 7),
) -> pd.DataFrame:
    X = _prepare_matrix(customer_features, feature_columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rows = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=DEFAULT_RANDOM_STATE, n_init=20)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        rows.append(
            {
                "algorithm": "kmeans",
                "n_clusters": n_clusters,
                "silhouette": silhouette_score(X_scaled, kmeans_labels),
                "bic": np.nan,
            }
        )

        gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=DEFAULT_RANDOM_STATE)
        gmm_labels = gmm.fit_predict(X_scaled)
        rows.append(
            {
                "algorithm": "gmm",
                "n_clusters": n_clusters,
                "silhouette": silhouette_score(X_scaled, gmm_labels),
                "bic": gmm.bic(X_scaled),
            }
        )

    return pd.DataFrame(rows).sort_values(["silhouette", "bic"], ascending=[False, True]).reset_index(drop=True)


def _label_segments(summary: pd.DataFrame) -> Dict[int, str]:
    labels = {}
    for _, row in summary.iterrows():
        cluster_id = int(row["segment_id"])
        if row["spend_total"] >= summary["spend_total"].quantile(0.75) and row["recency_days"] <= summary["recency_days"].quantile(0.25):
            labels[cluster_id] = "High-Value Loyalists"
        elif row["order_count"] >= summary["order_count"].quantile(0.75) and row["avg_basket_units"] >= summary["avg_basket_units"].median():
            labels[cluster_id] = "Basket Builders"
        elif row["recency_days"] <= summary["recency_days"].quantile(0.35) and row["spend_total"] < summary["spend_total"].median():
            labels[cluster_id] = "Recent Emerging Buyers"
        elif row["seasonal_concentration"] >= summary["seasonal_concentration"].quantile(0.75):
            labels[cluster_id] = "Seasonal Buyers"
        else:
            labels[cluster_id] = "Low-Engagement Accounts"
    return labels


def fit_segmentation_model(
    customer_features: pd.DataFrame,
    feature_columns: Iterable[str] = SEGMENT_FEATURE_COLUMNS,
    algorithm: str = "kmeans",
    n_clusters: int = 5,
) -> SegmentationArtifacts:
    X = _prepare_matrix(customer_features, feature_columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if algorithm == "gmm":
        cluster_model = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=DEFAULT_RANDOM_STATE)
        labels = cluster_model.fit_predict(X_scaled)
    else:
        cluster_model = KMeans(n_clusters=n_clusters, random_state=DEFAULT_RANDOM_STATE, n_init=20)
        labels = cluster_model.fit_predict(X_scaled)

    projection = PCA(n_components=2, random_state=DEFAULT_RANDOM_STATE).fit_transform(X_scaled)
    customer_segments = customer_features.copy()
    customer_segments["segment_id"] = labels.astype(int)
    customer_segments["segment_pca_x"] = projection[:, 0]
    customer_segments["segment_pca_y"] = projection[:, 1]

    summary = (
        customer_segments.groupby("segment_id")
        .agg(
            customers=("customer_id", "count"),
            recency_days=("recency_days", "median"),
            order_count=("order_count", "median"),
            spend_total=("spend_total", "median"),
            avg_order_value=("avg_order_value", "median"),
            avg_basket_units=("avg_basket_units", "median"),
            category_diversity=("category_diversity", "median"),
            seasonal_concentration=("seasonal_concentration", "median"),
            repeat_purchase_rate=("repeat_purchase_rate", "median"),
        )
        .reset_index()
    )
    label_map = _label_segments(summary)
    customer_segments["segment_name"] = customer_segments["segment_id"].map(label_map)
    summary["segment_name"] = summary["segment_id"].map(label_map)
    summary["share_of_customers"] = summary["customers"] / summary["customers"].sum()

    wrapped_model = Pipeline([("scaler", scaler), ("clusterer", cluster_model)])
    scores = {"silhouette": silhouette_score(X_scaled, labels)}
    if algorithm == "gmm":
        scores["bic"] = cluster_model.bic(X_scaled)

    return SegmentationArtifacts(
        model=wrapped_model,
        feature_columns=list(X.columns),
        algorithm=algorithm,
        n_clusters=n_clusters,
        scores=scores,
        segment_summary=summary.sort_values("customers", ascending=False),
        customer_segments=customer_segments.sort_values("segment_id"),
    )
