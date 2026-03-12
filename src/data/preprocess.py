from __future__ import annotations

import re

import pandas as pd


STOPWORDS = {
    "SET",
    "OF",
    "THE",
    "AND",
    "WITH",
    "IN",
    "TO",
    "ON",
    "FOR",
    "SMALL",
    "LARGE",
    "MEDIUM",
    "PINK",
    "BLUE",
    "WHITE",
    "RED",
    "GREEN",
    "BLACK",
    "VINTAGE",
    "RETRO",
    "WOODEN",
    "METAL",
    "ASSORTED",
    "PACK",
    "HANGING",
    "ROUND",
    "MINI",
}

CATEGORY_KEYWORDS = [
    "BAG",
    "BOX",
    "BOTTLE",
    "BOWL",
    "CANDLE",
    "CARD",
    "CLOCK",
    "CUP",
    "DECORATION",
    "FRAME",
    "GARLAND",
    "HOLDER",
    "JAR",
    "LANTERN",
    "LIGHT",
    "MUG",
    "PLATE",
    "SIGN",
    "TIN",
    "TRAY",
    "WRAP",
]


def _derive_product_family(description: str) -> str:
    if not isinstance(description, str) or not description.strip():
        return "Other"
    tokens = re.findall(r"[A-Z]+", description.upper())
    for token in tokens:
        if token in CATEGORY_KEYWORDS:
            return token.title()
    informative = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    return informative[-1].title() if informative else "Other"


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions = transactions.copy()
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])
    transactions = transactions.dropna(subset=["CustomerID", "StockCode", "InvoiceDate", "Description"])
    transactions = transactions.loc[~transactions["InvoiceNo"].str.startswith("C", na=False)].copy()
    transactions = transactions.loc[(transactions["Quantity"] > 0) & (transactions["UnitPrice"] > 0)].copy()
    transactions = transactions.rename(
        columns={
            "InvoiceNo": "order_id",
            "StockCode": "product_id",
            "Description": "product_name",
            "Quantity": "quantity",
            "InvoiceDate": "t_dat",
            "UnitPrice": "unit_price",
            "CustomerID": "customer_id",
            "Country": "country",
        }
    )
    transactions["order_id"] = transactions["order_id"].astype("string")
    transactions["product_id"] = transactions["product_id"].astype("string")
    transactions["product_name"] = transactions["product_name"].str.strip().fillna("Unknown").astype("string")
    transactions["customer_id"] = transactions["customer_id"].astype("string")
    transactions["country"] = transactions["country"].fillna("Unknown").astype("string")
    transactions["quantity"] = transactions["quantity"].astype("int32")
    transactions["unit_price"] = transactions["unit_price"].astype("float32")
    transactions["revenue"] = (transactions["quantity"] * transactions["unit_price"]).astype("float32")
    transactions["product_family"] = transactions["product_name"].map(_derive_product_family).astype("string")
    transactions["order_date"] = transactions["t_dat"].dt.normalize()
    transactions["month"] = transactions["t_dat"].dt.to_period("M").dt.to_timestamp()
    transactions["year_month"] = transactions["t_dat"].dt.strftime("%Y-%m")
    transactions["weekday"] = transactions["t_dat"].dt.day_name()
    transactions["hour"] = transactions["t_dat"].dt.hour
    transactions["is_weekend"] = transactions["t_dat"].dt.dayofweek.ge(5).astype("int8")
    transactions["season"] = transactions["t_dat"].dt.month.map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }
    )
    return transactions.sort_values(["customer_id", "t_dat", "order_id", "product_id"]).reset_index(drop=True)
