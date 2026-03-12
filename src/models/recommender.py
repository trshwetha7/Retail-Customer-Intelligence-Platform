from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RecommendationArtifacts:
    popularity_rank: pd.DataFrame
    item_similarity: sparse.csr_matrix
    item_index: Dict[str, int]
    customer_history: Dict[str, List[str]]
    train_interactions: pd.DataFrame
    test_interactions: pd.DataFrame
    metrics: pd.DataFrame


def temporal_interaction_split(transactions: pd.DataFrame, holdout_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = transactions["t_dat"].max().normalize() - pd.Timedelta(days=holdout_days)
    train = transactions.loc[transactions["t_dat"] < cutoff].copy()
    test = transactions.loc[transactions["t_dat"] >= cutoff].copy()
    return train, test


def fit_popularity_baseline(train_interactions: pd.DataFrame) -> pd.DataFrame:
    popularity = (
        train_interactions.groupby(["product_id", "product_name", "product_family"], dropna=False)
        .agg(
            purchases=("customer_id", "count"),
            unique_customers=("customer_id", "nunique"),
        )
        .reset_index()
        .sort_values(["purchases", "unique_customers"], ascending=False)
    )
    return popularity


def _build_sparse_matrix(train_interactions: pd.DataFrame):
    customer_codes = pd.Categorical(train_interactions["customer_id"])
    item_codes = pd.Categorical(train_interactions["product_id"])
    matrix = sparse.csr_matrix(
        (
            np.ones(len(train_interactions), dtype=np.float32),
            (customer_codes.codes, item_codes.codes),
        ),
        shape=(len(customer_codes.categories), len(item_codes.categories)),
    )
    customer_index = {customer: idx for idx, customer in enumerate(customer_codes.categories.astype(str))}
    item_index = {item: idx for idx, item in enumerate(item_codes.categories.astype(str))}
    return matrix, customer_index, item_index


def fit_item_similarity_recommender(
    train_interactions: pd.DataFrame,
    max_items: int = 4_000,
) -> tuple[sparse.csr_matrix, Dict[str, int], Dict[str, int], pd.DataFrame]:
    top_items = train_interactions["product_id"].value_counts().head(max_items).index
    filtered = train_interactions.loc[train_interactions["product_id"].isin(top_items)].copy()
    if filtered.empty:
        filtered = train_interactions.copy()
    interaction_matrix, customer_index, item_index = _build_sparse_matrix(filtered)
    item_similarity = cosine_similarity(interaction_matrix.T, dense_output=False)
    return item_similarity.tocsr(), customer_index, item_index, filtered


def recommend_popular(
    popularity_rank: pd.DataFrame,
    seen_items: List[str],
    n_items: int = 10,
) -> pd.DataFrame:
    return popularity_rank.loc[~popularity_rank["product_id"].isin(seen_items)].head(n_items).copy()


def recommend_personalized(
    customer_id: str,
    customer_history: Dict[str, List[str]],
    item_similarity: sparse.csr_matrix,
    item_index: Dict[str, int],
    popularity_rank: pd.DataFrame,
    product_lookup: pd.DataFrame,
    n_items: int = 10,
) -> pd.DataFrame:
    seen_items = customer_history.get(customer_id, [])
    if not seen_items:
        return recommend_popular(popularity_rank, seen_items, n_items)

    scores = np.zeros(item_similarity.shape[0], dtype=np.float32)
    for product_id in seen_items[-20:]:
        item_idx = item_index.get(product_id)
        if item_idx is None:
            continue
        scores += item_similarity[item_idx].toarray().ravel()

    for product_id in seen_items:
        item_idx = item_index.get(product_id)
        if item_idx is not None:
            scores[item_idx] = -np.inf

    top_indices = np.argsort(scores)[::-1][: n_items * 3]
    inverse_item_index = {idx: product for product, idx in item_index.items()}
    candidates = [inverse_item_index[idx] for idx in top_indices if np.isfinite(scores[idx])]

    if not candidates:
        return recommend_popular(popularity_rank, seen_items, n_items)

    recommendations = product_lookup.loc[product_lookup["product_id"].isin(candidates)].copy()
    recommendations["score"] = recommendations["product_id"].map(
        {inverse_item_index[idx]: float(scores[idx]) for idx in top_indices if idx in inverse_item_index}
    )
    recommendations = recommendations.sort_values("score", ascending=False).drop_duplicates("product_id")
    if recommendations.empty:
        return recommend_popular(popularity_rank, seen_items, n_items)
    return recommendations.head(n_items)


def evaluate_recommenders(
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    popularity_rank: pd.DataFrame,
    item_similarity: sparse.csr_matrix,
    item_index: Dict[str, int],
    product_lookup: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    train_history = train_interactions.groupby("customer_id")["product_id"].apply(list).to_dict()
    test_baskets = test_interactions.groupby("customer_id")["product_id"].apply(lambda s: list(pd.unique(s))).to_dict()

    rows = []
    sampled_customers = list(test_baskets.keys())[:250]
    for customer_id in sampled_customers:
        seen = train_history.get(customer_id, [])
        actual = set(test_baskets[customer_id])
        if not actual:
            continue
        popularity_recs = recommend_popular(popularity_rank, seen, k)["product_id"].astype(str).tolist()
        personalized_recs = recommend_personalized(
            customer_id,
            train_history,
            item_similarity,
            item_index,
            popularity_rank,
            product_lookup,
            k,
        )["product_id"].astype(str).tolist()
        for model_name, recs in [("popularity", popularity_recs), ("item_similarity", personalized_recs)]:
            hits = len(actual.intersection(recs))
            rows.append(
                {
                    "customer_id": customer_id,
                    "model_name": model_name,
                    "precision_at_k": hits / k,
                    "recall_at_k": hits / len(actual),
                    "hit_rate": float(hits > 0),
                }
            )
    metrics = pd.DataFrame(rows)
    return metrics.groupby("model_name").mean(numeric_only=True).reset_index()


def train_recommender(transactions: pd.DataFrame, product_lookup: pd.DataFrame) -> RecommendationArtifacts:
    train_interactions, test_interactions = temporal_interaction_split(transactions)
    popularity_rank = fit_popularity_baseline(train_interactions)
    item_similarity, _, item_index, filtered_train = fit_item_similarity_recommender(train_interactions)
    history = filtered_train.groupby("customer_id")["product_id"].apply(list).to_dict()
    filtered_test = test_interactions.loc[test_interactions["product_id"].isin(filtered_train["product_id"].unique())]
    metrics = evaluate_recommenders(
        train_interactions=filtered_train,
        test_interactions=filtered_test,
        popularity_rank=popularity_rank,
        item_similarity=item_similarity,
        item_index=item_index,
        product_lookup=product_lookup[["product_id", "product_name", "product_family"]].drop_duplicates(),
    )
    return RecommendationArtifacts(
        popularity_rank=popularity_rank,
        item_similarity=item_similarity,
        item_index=item_index,
        customer_history=history,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        metrics=metrics,
    )


def save_metrics(metrics: pd.DataFrame, path) -> None:
    path.write_text(json.dumps(metrics.to_dict(orient="records"), indent=2))
