from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _safe_hhi(values: pd.Series) -> float:
    total = values.sum()
    if total <= 0:
        return 0.0
    shares = values / total
    return float((shares**2).sum())


def _safe_entropy(values: pd.Series) -> float:
    total = values.sum()
    if total <= 0:
        return 0.0
    shares = values / total
    shares = shares[shares > 0]
    return float(-(shares * np.log(shares)).sum())


def _average_gap_days(order_dates: Iterable[pd.Timestamp]) -> float:
    order_dates = pd.Series(sorted(pd.to_datetime(list(order_dates)).unique()))
    if len(order_dates) <= 1:
        return np.nan
    return float(order_dates.diff().dropna().dt.days.mean())


def build_customer_features(
    transactions: pd.DataFrame,
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()

    tx = transactions.copy()
    as_of_date = pd.to_datetime(as_of_date or tx["t_dat"].max()) + pd.Timedelta(days=1)
    tx["order_date"] = pd.to_datetime(tx["order_date"])
    tx["season"] = tx["season"].fillna("Unknown")
    tx["product_family"] = tx["product_family"].fillna("Other")

    order_level = (
        tx.groupby(["customer_id", "order_id", "order_date"], as_index=False)
        .agg(
            order_value=("revenue", "sum"),
            basket_units=("quantity", "sum"),
            distinct_products=("product_id", "nunique"),
            weekend_order=("is_weekend", "max"),
        )
        .sort_values(["customer_id", "order_date", "order_id"])
    )

    customer_base = (
        tx.groupby("customer_id")
        .agg(
            first_purchase_date=("t_dat", "min"),
            last_purchase_date=("t_dat", "max"),
            unit_count=("quantity", "sum"),
            line_count=("product_id", "count"),
            unique_product_count=("product_id", "nunique"),
            spend_total=("revenue", "sum"),
            mean_price=("unit_price", "mean"),
            category_diversity=("product_family", "nunique"),
            country_order_share=("country", lambda s: float(s.value_counts(normalize=True).iloc[0]) if len(s) else 0.0),
        )
        .reset_index()
    )

    order_summary = (
        order_level.groupby("customer_id")
        .agg(
            order_count=("order_id", "nunique"),
            avg_order_value=("order_value", "mean"),
            avg_basket_units=("basket_units", "mean"),
            max_basket_units=("basket_units", "max"),
            avg_distinct_products=("distinct_products", "mean"),
            weekend_order_share=("weekend_order", "mean"),
        )
        .reset_index()
    )

    seasonal = (
        tx.groupby(["customer_id", "season"])["revenue"]
        .sum()
        .reset_index()
        .pivot(index="customer_id", columns="season", values="revenue")
        .fillna(0)
    )
    seasonal_concentration = seasonal.apply(_safe_hhi, axis=1).rename("seasonal_concentration")
    seasonal_entropy = seasonal.apply(_safe_entropy, axis=1).rename("seasonal_entropy")

    spend_concentration = (
        tx.groupby(["customer_id", "product_family"])["revenue"]
        .sum()
        .groupby(level=0)
        .apply(_safe_hhi)
        .rename("spend_concentration_hhi")
        .reset_index()
    )

    interpurchase = (
        order_level.groupby("customer_id")["order_date"]
        .apply(_average_gap_days)
        .rename("purchase_frequency_days")
        .reset_index()
    )

    features = customer_base.merge(order_summary, on="customer_id", how="left")
    features = features.merge(interpurchase, on="customer_id", how="left")
    features = features.merge(spend_concentration, on="customer_id", how="left")
    features = features.merge(seasonal_concentration.reset_index(), on="customer_id", how="left")
    features = features.merge(seasonal_entropy.reset_index(), on="customer_id", how="left")

    features["recency_days"] = (as_of_date - features["last_purchase_date"]).dt.days.astype(float)
    features["customer_tenure_days"] = (
        features["last_purchase_date"] - features["first_purchase_date"]
    ).dt.days.astype(float)
    features["repeat_product_rate"] = 1 - (
        features["unique_product_count"] / features["line_count"].clip(lower=1)
    )
    features["repeat_purchase_rate"] = (
        (features["order_count"] - 1).clip(lower=0) / features["order_count"].clip(lower=1)
    )
    features["monetary_per_day"] = features["spend_total"] / (features["customer_tenure_days"] + 1)

    numeric_columns = [
        "unit_count",
        "line_count",
        "unique_product_count",
        "spend_total",
        "mean_price",
        "category_diversity",
        "country_order_share",
        "order_count",
        "avg_order_value",
        "avg_basket_units",
        "max_basket_units",
        "avg_distinct_products",
        "weekend_order_share",
        "purchase_frequency_days",
        "spend_concentration_hhi",
        "seasonal_concentration",
        "seasonal_entropy",
        "recency_days",
        "customer_tenure_days",
        "repeat_product_rate",
        "repeat_purchase_rate",
        "monetary_per_day",
    ]
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)

    return features.sort_values("customer_id").reset_index(drop=True)


def attach_customer_attributes(customer_features: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    customer_attributes = (
        transactions.groupby("customer_id")
        .agg(
            primary_country=("country", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
            countries_seen=("country", "nunique"),
        )
        .reset_index()
    )
    return customer_features.merge(customer_attributes, on="customer_id", how="left")
