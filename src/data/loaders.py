from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.cleaning import (
    normalize_columns,
    standardize_amazon,
    standardize_international,
    standardize_inventory,
    to_numeric,
)


@dataclass
class DatasetBundle:
    amazon_orders: pd.DataFrame
    international_sales: pd.DataFrame
    inventory_stock: pd.DataFrame
    pricing_may_2022: pd.DataFrame
    pricing_mar_2021: pd.DataFrame
    expense_statement_text: str
    warehouse_comparison_text: str


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _text_from_csv_without_commas(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return raw.replace(",", " ")


def load_dataset_bundle(dataset_dir: str | Path) -> DatasetBundle:
    base = Path(dataset_dir)

    amazon_orders = _load_csv(base / "Amazon Sale Report.csv")
    amazon_orders = normalize_columns(amazon_orders)
    amazon_orders = standardize_amazon(amazon_orders)

    international_sales = _load_csv(base / "International sale Report.csv")
    international_sales = normalize_columns(international_sales)
    international_sales = standardize_international(international_sales)

    inventory_stock = _load_csv(base / "Sale Report.csv")
    inventory_stock = normalize_columns(inventory_stock)
    inventory_stock = standardize_inventory(inventory_stock)

    pricing_may_2022 = _load_csv(base / "May-2022.csv")
    pricing_may_2022 = normalize_columns(pricing_may_2022)
    pricing_may_2022 = to_numeric(
        pricing_may_2022,
        [
            "TP",
            "MRP_Old",
            "Final_MRP_Old",
            "Ajio_MRP",
            "Amazon_MRP",
            "Amazon_FBA_MRP",
            "Flipkart_MRP",
            "Limeroad_MRP",
            "Myntra_MRP",
            "Paytm_MRP",
            "Snapdeal_MRP",
        ],
    )

    pricing_mar_2021 = _load_csv(base / "P  L March 2021.csv")
    pricing_mar_2021 = normalize_columns(pricing_mar_2021)
    pricing_mar_2021 = to_numeric(
        pricing_mar_2021,
        [
            "TP_1",
            "TP_2",
            "MRP_Old",
            "Final_MRP_Old",
            "Ajio_MRP",
            "Amazon_MRP",
            "Amazon_FBA_MRP",
            "Flipkart_MRP",
            "Limeroad_MRP",
            "Myntra_MRP",
            "Paytm_MRP",
            "Snapdeal_MRP",
        ],
    )

    expense_statement_text = _text_from_csv_without_commas(base / "Expense IIGF.csv")
    warehouse_comparison_text = _text_from_csv_without_commas(
        base / "Cloud Warehouse Compersion Chart.csv"
    )

    return DatasetBundle(
        amazon_orders=amazon_orders,
        international_sales=international_sales,
        inventory_stock=inventory_stock,
        pricing_may_2022=pricing_may_2022,
        pricing_mar_2021=pricing_mar_2021,
        expense_statement_text=expense_statement_text,
        warehouse_comparison_text=warehouse_comparison_text,
    )

