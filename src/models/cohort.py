from __future__ import annotations

import pandas as pd


def build_cohort_frames(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_base = transactions[["customer_id", "t_dat", "revenue"]].copy()
    cohort_base["order_month"] = cohort_base["t_dat"].dt.to_period("M")
    first_purchase = cohort_base.groupby("customer_id")["order_month"].min().rename("cohort_month")
    cohort_base = cohort_base.merge(first_purchase, on="customer_id", how="left")
    cohort_base["cohort_index"] = (
        cohort_base["order_month"].dt.year - cohort_base["cohort_month"].dt.year
    ) * 12 + (cohort_base["order_month"].dt.month - cohort_base["cohort_month"].dt.month)

    cohort_sizes = cohort_base.groupby("cohort_month")["customer_id"].nunique().rename("cohort_size")

    retained = (
        cohort_base.groupby(["cohort_month", "cohort_index"])["customer_id"]
        .nunique()
        .rename("customers")
        .reset_index()
        .merge(cohort_sizes.reset_index(), on="cohort_month", how="left")
    )
    retained["retention_rate"] = retained["customers"] / retained["cohort_size"]

    revenue = (
        cohort_base.groupby(["cohort_month", "cohort_index"])["revenue"]
        .sum()
        .rename("revenue")
        .reset_index()
    )
    return retained, revenue


def build_retention_matrix(retained: pd.DataFrame) -> pd.DataFrame:
    retention_matrix = retained.pivot(index="cohort_month", columns="cohort_index", values="retention_rate").sort_index()
    retention_matrix.index = retention_matrix.index.astype(str)
    return retention_matrix
