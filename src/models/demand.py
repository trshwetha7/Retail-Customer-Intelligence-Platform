from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def build_monthly_demand(transactions: pd.DataFrame, category_column: str = "product_family") -> pd.DataFrame:
    demand = (
        transactions.groupby(["month", category_column], dropna=False)
        .agg(
            revenue=("revenue", "sum"),
            units=("quantity", "sum"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
        .rename(columns={category_column: "category"})
    )
    return demand.sort_values(["month", "revenue"], ascending=[True, False])


def top_category_shifts(monthly_demand: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    latest_month = monthly_demand["month"].max()
    prior_month = latest_month - pd.offsets.MonthBegin(1)
    latest = monthly_demand.loc[monthly_demand["month"] == latest_month, ["category", "revenue"]].rename(
        columns={"revenue": "latest_revenue"}
    )
    prior = monthly_demand.loc[monthly_demand["month"] == prior_month, ["category", "revenue"]].rename(
        columns={"revenue": "prior_revenue"}
    )
    shifts = latest.merge(prior, on="category", how="left").fillna(0.0)
    shifts["absolute_change"] = shifts["latest_revenue"] - shifts["prior_revenue"]
    shifts["pct_change"] = shifts["absolute_change"] / shifts["prior_revenue"].replace(0, np.nan)
    return shifts.sort_values("absolute_change", ascending=False).head(top_n)


def purchase_timing_profile(transactions: pd.DataFrame) -> pd.DataFrame:
    profile = (
        transactions.assign(
            weekday=transactions["t_dat"].dt.day_name(),
            month_name=transactions["t_dat"].dt.month_name(),
        )
        .groupby(["weekday", "month_name"], dropna=False)
        .agg(revenue=("revenue", "sum"), units=("quantity", "sum"))
        .reset_index()
    )
    return profile


def forecast_category_demand(monthly_demand: pd.DataFrame, category: str, periods: int = 3) -> pd.DataFrame:
    series = monthly_demand.loc[monthly_demand["category"] == category].sort_values("month").copy()
    if len(series) < 6:
        return pd.DataFrame()

    series["time_idx"] = np.arange(len(series))
    series["month_num"] = series["month"].dt.month
    X = pd.get_dummies(series[["time_idx", "month_num"]].astype({"month_num": "category"}), drop_first=True)
    model = LinearRegression()
    model.fit(X, series["revenue"])

    future_rows = []
    last_month = series["month"].max()
    for step in range(1, periods + 1):
        month = last_month + pd.offsets.MonthBegin(step)
        future_rows.append({"month": month, "time_idx": len(series) + step - 1, "month_num": month.month})
    future = pd.DataFrame(future_rows)
    X_future = pd.get_dummies(future[["time_idx", "month_num"]].astype({"month_num": "category"}), drop_first=True)
    X_future = X_future.reindex(columns=X.columns, fill_value=0)
    future["forecast_revenue"] = model.predict(X_future)
    future["category"] = category
    return future[["month", "category", "forecast_revenue"]]
