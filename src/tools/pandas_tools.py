from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.data.loaders import DatasetBundle


@dataclass
class ToolResult:
    tool_name: str
    summary: str
    records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PandasTools:
    def __init__(self, bundle: DatasetBundle):
        self.bundle = bundle

    @staticmethod
    def _parse_date(value: str | None) -> pd.Timestamp | None:
        if not value:
            return None
        return pd.to_datetime(value, errors="coerce")

    @staticmethod
    def _apply_date_range(
        df: pd.DataFrame, date_col: str, date_from: str | None, date_to: str | None
    ) -> pd.DataFrame:
        result = df.copy()
        start = PandasTools._parse_date(date_from)
        end = PandasTools._parse_date(date_to)
        if start is not None:
            result = result[result[date_col] >= start]
        if end is not None:
            result = result[result[date_col] <= end]
        return result

    @staticmethod
    def _records(df: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
        if df.empty:
            return []
        return df.head(limit).to_dict(orient="records")

    def _amazon_sales_base(
        self, date_from: str | None = None, date_to: str | None = None, delivered_only: bool = True
    ) -> pd.DataFrame:
        df = self.bundle.amazon_orders.copy()
        df = self._apply_date_range(df, "Date", date_from, date_to)
        if delivered_only and "is_delivered" in df.columns:
            df = df[df["is_delivered"]]
        return df

    def total_sales_amazon(
        self, date_from: str | None = None, date_to: str | None = None, delivered_only: bool = True
    ) -> ToolResult:
        df = self._amazon_sales_base(date_from, date_to, delivered_only)
        total = float(df["Amount"].fillna(0).sum())
        return ToolResult(
            tool_name="total_sales_amazon",
            summary=f"Total Amazon sales amount is {total:.2f}.",
            records=[{"total_sales_amount": round(total, 2), "rows_considered": int(len(df))}],
        )

    def units_sold_by_category(
        self,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        delivered_only: bool = True,
    ) -> ToolResult:
        df = self._amazon_sales_base(date_from, date_to, delivered_only)
        if category:
            df = df[df["Category"].astype(str).str.contains(category, case=False, na=False)]
        grouped = (
            df.groupby("Category", dropna=False)["Qty"].sum(min_count=1).fillna(0).reset_index()
            .sort_values("Qty", ascending=False)
        )
        return ToolResult(
            tool_name="units_sold_by_category",
            summary="Units sold by category computed from Amazon orders.",
            records=self._records(grouped),
        )

    def category_sales_rank(self, top_n: int = 10, delivered_only: bool = True) -> ToolResult:
        df = self._amazon_sales_base(delivered_only=delivered_only)
        grouped = (
            df.groupby("Category", dropna=False)["Amount"]
            .sum(min_count=1)
            .fillna(0)
            .reset_index()
            .sort_values("Amount", ascending=False)
            .head(top_n)
        )
        return ToolResult(
            tool_name="category_sales_rank",
            summary=f"Top {top_n} categories by sales amount.",
            records=self._records(grouped, limit=top_n),
        )

    def state_sales_rank(self, top_n: int = 10, delivered_only: bool = True) -> ToolResult:
        df = self._amazon_sales_base(delivered_only=delivered_only)
        grouped = (
            df.groupby("ship_state", dropna=False)["Amount"]
            .sum(min_count=1)
            .fillna(0)
            .reset_index()
            .sort_values("Amount", ascending=False)
            .head(top_n)
        )
        return ToolResult(
            tool_name="state_sales_rank",
            summary=f"Top {top_n} states by Amazon sales.",
            records=self._records(grouped, limit=top_n),
        )

    def city_sales_rank(self, top_n: int = 10, delivered_only: bool = True) -> ToolResult:
        df = self._amazon_sales_base(delivered_only=delivered_only)
        grouped = (
            df.groupby("ship_city", dropna=False)["Amount"]
            .sum(min_count=1)
            .fillna(0)
            .reset_index()
            .sort_values("Amount", ascending=False)
            .head(top_n)
        )
        return ToolResult(
            tool_name="city_sales_rank",
            summary=f"Top {top_n} cities by Amazon sales.",
            records=self._records(grouped, limit=top_n),
        )

    def order_status_breakdown(self) -> ToolResult:
        df = self.bundle.amazon_orders.copy()
        grouped = (
            df.groupby("Status", dropna=False)["Order_ID"]
            .count()
            .reset_index(name="order_count")
            .sort_values("order_count", ascending=False)
        )
        total = grouped["order_count"].sum()
        grouped["share_pct"] = (grouped["order_count"] / total * 100).round(2) if total else 0
        return ToolResult(
            tool_name="order_status_breakdown",
            summary="Order status distribution computed.",
            records=self._records(grouped),
        )

    def cancellation_rate(self) -> ToolResult:
        df = self.bundle.amazon_orders.copy()
        total_orders = len(df)
        cancelled_orders = int(df["is_cancelled"].sum()) if "is_cancelled" in df.columns else 0
        rate = (cancelled_orders / total_orders * 100) if total_orders else 0
        return ToolResult(
            tool_name="cancellation_rate",
            summary=f"Cancellation rate is {rate:.2f}% ({cancelled_orders}/{total_orders}).",
            records=[
                {
                    "total_orders": total_orders,
                    "cancelled_orders": cancelled_orders,
                    "cancellation_rate_pct": round(rate, 2),
                }
            ],
        )

    def b2b_vs_b2c_summary(self) -> ToolResult:
        df = self.bundle.amazon_orders.copy()
        if "B2B" not in df.columns:
            return ToolResult("b2b_vs_b2c_summary", "B2B column not found.", [])
        grouped = (
            df.assign(B2B=df["B2B"].astype(str))
            .groupby("B2B", dropna=False)
            .agg(orders=("Order_ID", "count"), sales=("Amount", "sum"), qty=("Qty", "sum"))
            .reset_index()
        )
        return ToolResult(
            tool_name="b2b_vs_b2c_summary",
            summary="B2B vs B2C comparison computed.",
            records=self._records(grouped),
        )

    def international_total_sales(self, month: str | None = None) -> ToolResult:
        df = self.bundle.international_sales.copy()
        if month:
            df = df[df["Months"].astype(str).str.lower() == month.lower()]
        total = float(df["GROSS_AMT"].fillna(0).sum())
        return ToolResult(
            tool_name="international_total_sales",
            summary=f"International total sales amount is {total:.2f}.",
            records=[{"month": month or "all", "total_sales_amount": round(total, 2)}],
        )

    def top_customers_international(self, top_n: int = 10) -> ToolResult:
        df = self.bundle.international_sales.copy()
        grouped = (
            df.groupby("CUSTOMER", dropna=False)["GROSS_AMT"]
            .sum(min_count=1)
            .fillna(0)
            .reset_index()
            .sort_values("GROSS_AMT", ascending=False)
            .head(top_n)
        )
        return ToolResult(
            tool_name="top_customers_international",
            summary=f"Top {top_n} international customers by sales.",
            records=self._records(grouped, limit=top_n),
        )

    def top_styles_international(self, top_n: int = 10) -> ToolResult:
        df = self.bundle.international_sales.copy()
        grouped = (
            df.groupby("Style", dropna=False)
            .agg(total_sales=("GROSS_AMT", "sum"), total_units=("PCS", "sum"))
            .reset_index()
            .sort_values("total_sales", ascending=False)
            .head(top_n)
        )
        return ToolResult(
            tool_name="top_styles_international",
            summary=f"Top {top_n} styles by international sales.",
            records=self._records(grouped, limit=top_n),
        )

    def total_stock_by_category(self) -> ToolResult:
        df = self.bundle.inventory_stock.copy()
        grouped = (
            df.groupby("Category", dropna=False)["Stock"]
            .sum(min_count=1)
            .fillna(0)
            .reset_index()
            .sort_values("Stock", ascending=False)
        )
        return ToolResult(
            tool_name="total_stock_by_category",
            summary="Total stock by category computed.",
            records=self._records(grouped),
        )

    def low_stock_items(self, threshold: int = 5) -> ToolResult:
        df = self.bundle.inventory_stock.copy()
        low = df[df["Stock"].fillna(0) < threshold].sort_values("Stock", ascending=True)
        slim = low[["SKU_Code", "Design_No", "Stock", "Category", "Size", "Color"]]
        return ToolResult(
            tool_name="low_stock_items",
            summary=f"Items with stock lower than {threshold}.",
            records=self._records(slim, limit=50),
        )

    def stock_by_color(self, category_contains: str | None = None) -> ToolResult:
        df = self.bundle.inventory_stock.copy()
        if category_contains:
            df = df[df["Category"].astype(str).str.contains(category_contains, case=False, na=False)]
        grouped = (
            df.groupby("Color", dropna=False)["Stock"]
            .sum(min_count=1)
            .fillna(0)
            .reset_index()
            .sort_values("Stock", ascending=False)
        )
        return ToolResult(
            tool_name="stock_by_color",
            summary="Stock by color computed.",
            records=self._records(grouped),
        )

    def channel_price_comparison_may2022(
        self, channel_a: str = "Amazon_MRP", channel_b: str = "Flipkart_MRP"
    ) -> ToolResult:
        df = self.bundle.pricing_may_2022.copy()
        if channel_a not in df.columns or channel_b not in df.columns:
            return ToolResult(
                tool_name="channel_price_comparison_may2022",
                summary=f"Invalid channels: {channel_a}, {channel_b}.",
                records=[],
            )
        out = df[["Sku", channel_a, channel_b]].copy()
        out["difference"] = out[channel_a] - out[channel_b]
        summary = (
            out["difference"].mean(skipna=True) if not out["difference"].dropna().empty else float("nan")
        )
        return ToolResult(
            tool_name="channel_price_comparison_may2022",
            summary=f"Average {channel_a} - {channel_b} difference is {summary:.2f}.",
            records=self._records(out, limit=30),
        )

    def price_snapshot_march2021(self, metric: str = "Final_MRP_Old") -> ToolResult:
        df = self.bundle.pricing_mar_2021.copy()
        if metric not in df.columns:
            return ToolResult(
                tool_name="price_snapshot_march2021",
                summary=f"Metric {metric} not found.",
                records=[],
            )
        grouped = (
            df.groupby("Category", dropna=False)[metric]
            .mean()
            .reset_index(name=f"avg_{metric.lower()}")
            .sort_values(by=f"avg_{metric.lower()}", ascending=False)
        )
        return ToolResult(
            tool_name="price_snapshot_march2021",
            summary=f"Average {metric} by category computed.",
            records=self._records(grouped),
        )

    def expense_statement_text(self) -> ToolResult:
        text = self.bundle.expense_statement_text
        return ToolResult(
            tool_name="expense_statement_text",
            summary="Raw Expense IIGF text loaded.",
            records=[{"text": text[:10000]}],
        )

    def warehouse_comparison_text(self) -> ToolResult:
        text = self.bundle.warehouse_comparison_text
        return ToolResult(
            tool_name="warehouse_comparison_text",
            summary="Raw warehouse comparison text loaded.",
            records=[{"text": text[:10000]}],
        )

    def list_available_tools(self) -> list[str]:
        return [
            "total_sales_amazon",
            "units_sold_by_category",
            "category_sales_rank",
            "state_sales_rank",
            "city_sales_rank",
            "order_status_breakdown",
            "cancellation_rate",
            "b2b_vs_b2c_summary",
            "international_total_sales",
            "top_customers_international",
            "top_styles_international",
            "total_stock_by_category",
            "low_stock_items",
            "stock_by_color",
            "channel_price_comparison_may2022",
            "price_snapshot_march2021",
            "expense_statement_text",
            "warehouse_comparison_text",
        ]

    def run_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        if not hasattr(self, tool_name):
            return ToolResult(tool_name, f"Unknown tool {tool_name}.", []).to_dict()
        fn = getattr(self, tool_name)
        sig = inspect.signature(fn)
        accepted = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        result = fn(**filtered_kwargs)
        if isinstance(result, ToolResult):
            return result.to_dict()
        return ToolResult(tool_name, "Tool returned unsupported type.", []).to_dict()

