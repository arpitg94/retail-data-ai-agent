from __future__ import annotations

from typing import Iterable

import pandas as pd


DELIVERED_STATUSES = {
    "Shipped",
    "Shipped - Delivered to Buyer",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [
        str(col).strip().replace(" ", "_").replace("-", "_").replace(".", "")
        for col in cleaned.columns
    ]
    return cleaned


def to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for col in columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    return cleaned


def parse_amazon_date(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if "Date" in cleaned.columns:
        cleaned["Date"] = pd.to_datetime(cleaned["Date"], format="%m-%d-%y", errors="coerce")
    return cleaned


def parse_international_date(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if "DATE" in cleaned.columns:
        cleaned["DATE"] = pd.to_datetime(cleaned["DATE"], format="%m-%d-%y", errors="coerce")
    return cleaned


def normalize_text_fields(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for col in columns:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()
    return cleaned


def standardize_amazon(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = parse_amazon_date(df)
    cleaned = to_numeric(cleaned, ["Qty", "Amount"])
    cleaned = normalize_text_fields(cleaned, ["Category", "Status", "ship_state", "ship_city"])
    if "Category" in cleaned.columns:
        cleaned["Category"] = cleaned["Category"].str.lower()
    if "Status" in cleaned.columns:
        cleaned["is_delivered"] = cleaned["Status"].isin(DELIVERED_STATUSES)
        cleaned["is_cancelled"] = cleaned["Status"].eq("Cancelled")
    return cleaned


def standardize_international(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = parse_international_date(df)
    cleaned = to_numeric(cleaned, ["PCS", "RATE", "GROSS_AMT"])
    cleaned = normalize_text_fields(cleaned, ["Months", "CUSTOMER", "Style", "SKU", "Size"])
    return cleaned


def standardize_inventory(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = to_numeric(df, ["Stock"])
    cleaned = normalize_text_fields(cleaned, ["Category", "Size", "Color"])
    return cleaned

