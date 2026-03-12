from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_RANDOM_STATE, OBSERVATION_WINDOW_DAYS, PREDICTION_HORIZON_DAYS
from src.features.customer_features import build_customer_features


@dataclass
class FutureValueArtifacts:
    task: str
    target_column: str
    feature_columns: List[str]
    models: Dict[str, Pipeline]
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    snapshots: pd.DataFrame
    recommendation: str


def _safe_roc_auc(y_true, scores) -> float:
    return roc_auc_score(y_true, scores) if pd.Series(y_true).nunique() > 1 else np.nan


def generate_snapshot_cutoffs(
    transactions: pd.DataFrame,
    observation_window_days: int = OBSERVATION_WINDOW_DAYS,
    prediction_horizon_days: int = PREDICTION_HORIZON_DAYS,
    max_snapshots: int = 8,
) -> List[pd.Timestamp]:
    max_date = transactions["t_dat"].max().normalize()
    min_date = transactions["t_dat"].min().normalize()
    earliest_cutoff = min_date + pd.Timedelta(days=observation_window_days)
    latest_cutoff = max_date - pd.Timedelta(days=prediction_horizon_days)
    if latest_cutoff <= earliest_cutoff:
        return []
    monthly = pd.date_range(earliest_cutoff, latest_cutoff, freq="MS")
    return list(monthly[-max_snapshots:])


def build_snapshot_dataset(
    transactions: pd.DataFrame,
    observation_window_days: int = OBSERVATION_WINDOW_DAYS,
    prediction_horizon_days: int = PREDICTION_HORIZON_DAYS,
) -> pd.DataFrame:
    snapshots = []
    for cutoff in generate_snapshot_cutoffs(transactions, observation_window_days, prediction_horizon_days):
        history_start = cutoff - pd.Timedelta(days=observation_window_days)
        future_end = cutoff + pd.Timedelta(days=prediction_horizon_days)

        history = transactions.loc[(transactions["t_dat"] >= history_start) & (transactions["t_dat"] < cutoff)].copy()
        future = transactions.loc[(transactions["t_dat"] >= cutoff) & (transactions["t_dat"] < future_end)].copy()
        if history.empty:
            continue

        features = build_customer_features(history, as_of_date=cutoff)
        future_labels = (
            future.groupby("customer_id")
            .agg(
                future_spend_90d=("revenue", "sum"),
                future_orders_90d=("order_date", "nunique"),
                future_units_90d=("quantity", "sum"),
            )
            .reset_index()
        )
        snapshot = features.merge(future_labels, on="customer_id", how="left")
        snapshot["future_spend_90d"] = snapshot["future_spend_90d"].fillna(0.0)
        snapshot["future_orders_90d"] = snapshot["future_orders_90d"].fillna(0).astype(int)
        snapshot["future_units_90d"] = snapshot["future_units_90d"].fillna(0).astype(int)
        snapshot["repeat_purchase_90d"] = (snapshot["future_orders_90d"] > 0).astype(int)
        positive_spend = snapshot.loc[snapshot["future_spend_90d"] > 0, "future_spend_90d"]
        threshold = positive_spend.quantile(0.8) if not positive_spend.empty else np.inf
        snapshot["high_value_return_90d"] = (
            (snapshot["future_spend_90d"] > 0) & (snapshot["future_spend_90d"] >= threshold)
        ).astype(int)
        snapshot["snapshot_date"] = cutoff
        snapshots.append(snapshot)

    if not snapshots:
        return pd.DataFrame()
    return pd.concat(snapshots, ignore_index=True)


def choose_target_definition(snapshot_df: pd.DataFrame) -> Dict[str, object]:
    if snapshot_df.empty:
        raise ValueError("Snapshot dataframe is empty. Generate snapshots before choosing a target.")

    non_zero_spend_share = float((snapshot_df["future_spend_90d"] > 0).mean())
    high_value_share = float(snapshot_df["high_value_return_90d"].mean())
    repeat_share = float(snapshot_df["repeat_purchase_90d"].mean())

    if 0.15 <= non_zero_spend_share <= 0.8:
        return {
            "task": "regression",
            "target_column": "future_spend_90d",
            "recommendation": (
                "Primary target: 90-day future spend. The dataset provides enough temporal coverage to frame "
                "customer value as a continuous near-term outcome while preserving a meaningful share of zeros."
            ),
            "target_diagnostics": {
                "non_zero_spend_share": non_zero_spend_share,
                "repeat_share": repeat_share,
                "high_value_share": high_value_share,
            },
        }

    return {
        "task": "classification",
        "target_column": "repeat_purchase_90d",
        "recommendation": (
            "Primary target: repeat purchase in the next 90 days. The spend distribution is too sparse for a stable "
            "regression target, so the more defensible task is reactivation propensity."
        ),
        "target_diagnostics": {
            "non_zero_spend_share": non_zero_spend_share,
            "repeat_share": repeat_share,
            "high_value_share": high_value_share,
        },
    }


def _split_snapshot_data(snapshot_df: pd.DataFrame, target_column: str):
    ordered_dates = sorted(snapshot_df["snapshot_date"].dropna().unique())
    if len(ordered_dates) < 2:
        raise ValueError("Need at least two snapshot dates for a temporal train/test split.")
    cutoff_date = ordered_dates[-2]
    train = snapshot_df.loc[snapshot_df["snapshot_date"] <= cutoff_date].copy()
    test = snapshot_df.loc[snapshot_df["snapshot_date"] > cutoff_date].copy()
    drop_columns = {
        "customer_id",
        "snapshot_date",
        "future_spend_90d",
        "future_orders_90d",
        "future_units_90d",
        "repeat_purchase_90d",
        "high_value_return_90d",
    }
    candidate_columns = [column for column in snapshot_df.columns if column not in drop_columns]
    feature_columns = [
        column for column in candidate_columns if pd.api.types.is_numeric_dtype(snapshot_df[column])
    ]
    X_train = train[feature_columns]
    X_test = test[feature_columns]
    y_train = train[target_column]
    y_test = test[target_column]
    return train, test, X_train, X_test, y_train, y_test, feature_columns


def train_future_value_models(snapshot_df: pd.DataFrame) -> FutureValueArtifacts:
    choice = choose_target_definition(snapshot_df)
    task = choice["task"]
    target_column = choice["target_column"]
    train, test, X_train, X_test, y_train, y_test, feature_columns = _split_snapshot_data(snapshot_df, target_column)

    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer([("numeric", numeric_preprocessor, feature_columns)])

    if task == "regression":
        models = {
            "baseline": Pipeline([("prep", preprocessor), ("model", DummyRegressor(strategy="median"))]),
            "hist_gbm": Pipeline(
                [("prep", preprocessor), ("model", HistGradientBoostingRegressor(random_state=DEFAULT_RANDOM_STATE))]
            ),
        }
    else:
        models = {
            "baseline": Pipeline([("prep", preprocessor), ("model", DummyClassifier(strategy="prior"))]),
        }
        if pd.Series(y_train).nunique() > 1:
            models["logistic"] = Pipeline(
                [("prep", preprocessor), ("model", LogisticRegression(max_iter=500, random_state=DEFAULT_RANDOM_STATE))]
            )
            models["hist_gbm"] = Pipeline(
                [("prep", preprocessor), ("model", HistGradientBoostingClassifier(random_state=DEFAULT_RANDOM_STATE))]
            )

    metric_rows = []
    prediction_frames = []
    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        prediction_frame = test[["customer_id", "snapshot_date"]].copy()
        prediction_frame["model_name"] = model_name
        prediction_frame["actual"] = y_test.to_numpy()
        prediction_frame["prediction"] = preds

        if task == "regression":
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            metric_rows.append(
                {
                    "model_name": model_name,
                    "rmse": rmse,
                    "mae": mean_absolute_error(y_test, preds),
                    "r2": r2_score(y_test, preds),
                }
            )
        else:
            scores = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else preds
            prediction_frame["score"] = scores
            metric_rows.append(
                {
                    "model_name": model_name,
                    "roc_auc": _safe_roc_auc(y_test, scores),
                    "pr_auc": average_precision_score(y_test, scores),
                    "f1": f1_score(y_test, preds, zero_division=0),
                    "precision": precision_score(y_test, preds, zero_division=0),
                    "recall": recall_score(y_test, preds, zero_division=0),
                }
            )
        prediction_frames.append(prediction_frame)

    metrics = pd.DataFrame(metric_rows)
    if task == "regression":
        metrics = metrics.sort_values(["rmse", "mae"], ascending=[True, True])
    else:
        metrics = metrics.sort_values(["pr_auc", "roc_auc", "f1"], ascending=[False, False, False])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return FutureValueArtifacts(
        task=task,
        target_column=target_column,
        feature_columns=feature_columns,
        models=models,
        metrics=metrics,
        predictions=predictions,
        snapshots=snapshot_df,
        recommendation=choice["recommendation"],
    )


def save_metrics(metrics: pd.DataFrame, path) -> None:
    path.write_text(json.dumps(metrics.to_dict(orient="records"), indent=2))
