from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

EXPECTED_RAW_FILES = {
    "transactions": [
        "Online Retail.xlsx",
        "online_retail.xlsx",
        "online_retail.csv",
        "Online Retail.csv",
    ]
}

DEFAULT_RANDOM_STATE = 42
DEFAULT_CUSTOMER_SAMPLE_FRAC = 1.0
DEFAULT_MAX_CHUNKSIZE = 500_000

OBSERVATION_WINDOW_DAYS = 180
PREDICTION_HORIZON_DAYS = 90
RETENTION_CADENCE = "M"

SEGMENT_FEATURE_COLUMNS = [
    "recency_days",
    "order_count",
    "unit_count",
    "spend_total",
    "avg_order_value",
    "avg_basket_units",
    "purchase_frequency_days",
    "category_diversity",
    "repeat_product_rate",
    "spend_concentration_hhi",
    "seasonal_concentration",
    "weekend_order_share",
    "country_order_share",
    "mean_price",
]

ARTIFACT_PATHS = {
    "customer_features": PROCESSED_DATA_DIR / "customer_features.parquet",
    "transactions_enriched": PROCESSED_DATA_DIR / "transactions_enriched.parquet",
    "cohort_retention": PROCESSED_DATA_DIR / "cohort_retention.parquet",
    "cohort_revenue": PROCESSED_DATA_DIR / "cohort_revenue.parquet",
    "monthly_demand": PROCESSED_DATA_DIR / "monthly_demand.parquet",
    "segment_summary": PROCESSED_DATA_DIR / "segment_summary.parquet",
    "snapshot_features": PROCESSED_DATA_DIR / "customer_snapshots.parquet",
    "segmentation_model": MODELS_DIR / "segmentation.joblib",
    "future_value_model": MODELS_DIR / "future_value.joblib",
    "recommender_model": MODELS_DIR / "recommender.joblib",
    "future_value_metrics": MODELS_DIR / "future_value_metrics.json",
    "recommender_metrics": MODELS_DIR / "recommender_metrics.json",
}


def ensure_directories() -> None:
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
