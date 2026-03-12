from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import (
    DEFAULT_CUSTOMER_SAMPLE_FRAC,
    EXPECTED_RAW_FILES,
    RAW_DATA_DIR,
)


@dataclass
class DatasetBundle:
    transactions: pd.DataFrame
    metadata: Dict[str, object]


def _find_file(filenames: list[str], search_root: Path = RAW_DATA_DIR) -> Path:
    for filename in filenames:
        direct = search_root / filename
        if direct.exists():
            return direct
        matches = list(search_root.rglob(filename))
        if matches:
            return matches[0]

    joined = ", ".join(filenames)
    raise FileNotFoundError(
        f"Could not find any of [{joined}]. Place the UCI Online Retail file under {search_root}."
    )


def discover_raw_files(search_root: Path = RAW_DATA_DIR) -> Dict[str, Path]:
    return {name: _find_file(filenames, search_root) for name, filenames in EXPECTED_RAW_FILES.items()}


def raw_data_available(search_root: Path = RAW_DATA_DIR) -> bool:
    try:
        discover_raw_files(search_root)
    except FileNotFoundError:
        return False
    return True


def _hash_sample_mask(values: pd.Series, sample_frac: float) -> pd.Series:
    hashed = pd.util.hash_pandas_object(values.fillna("missing"), index=False)
    return (hashed % 10_000) < int(sample_frac * 10_000)


def read_transactions(
    path: Path,
    sample_frac: Optional[float] = None,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    sample_frac = sample_frac if sample_frac is not None else 1.0
    sample_frac = min(max(sample_frac, 0.0), 1.0)

    loader = pd.read_excel if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv
    read_kwargs = {
        "dtype": {
            "InvoiceNo": "string",
            "StockCode": "string",
            "Description": "string",
            "Quantity": "float32",
            "UnitPrice": "float32",
            "CustomerID": "string",
            "Country": "string",
        },
        "parse_dates": ["InvoiceDate"],
    }
    transactions = loader(path, **read_kwargs)
    raw_rows = len(transactions)
    if sample_frac < 1.0:
        transactions = transactions.loc[_hash_sample_mask(transactions["CustomerID"], sample_frac)].copy()
    kept_rows = len(transactions)
    metadata = {
        "raw_transaction_rows": raw_rows,
        "loaded_transaction_rows": kept_rows,
        "customer_sample_frac": sample_frac,
    }
    return transactions, metadata


def load_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    sample_frac: Optional[float] = DEFAULT_CUSTOMER_SAMPLE_FRAC,
) -> DatasetBundle:
    files = discover_raw_files(raw_dir)
    transactions, metadata = read_transactions(files["transactions"], sample_frac=sample_frac)
    metadata["source_paths"] = {name: str(path) for name, path in files.items()}
    return DatasetBundle(transactions=transactions, metadata=metadata)
