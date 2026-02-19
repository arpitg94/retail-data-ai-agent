from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def heuristic_plan(query: str) -> dict[str, Any]:
    q = query.lower()

    if "penalty" in q or "warehouse" in q or "fill rate" in q:
        return {
            "intent": "warehouse_text_lookup",
            "tool_calls": [{"tool_name": "warehouse_comparison_text", "kwargs": {}}],
            "notes": "Text lookup from warehouse comparison file.",
        }
    if "expense" in q or "iigf" in q:
        return {
            "intent": "expense_text_lookup",
            "tool_calls": [{"tool_name": "expense_statement_text", "kwargs": {}}],
            "notes": "Text lookup from expense statement file.",
        }
    if "customer" in q and ("top" in q or "highest" in q):
        return {
            "intent": "top_customers_international",
            "tool_calls": [{"tool_name": "top_customers_international", "kwargs": {"top_n": 5}}],
            "notes": "Top international customers by revenue.",
        }
    if "international" in q and ("sales" in q or "revenue" in q):
        month = "Jun-21" if "jun" in q else None
        return {
            "intent": "international_total_sales",
            "tool_calls": [{"tool_name": "international_total_sales", "kwargs": {"month": month}}],
            "notes": "International sales summary.",
        }
    if "low stock" in q or "running low" in q:
        return {
            "intent": "low_stock_items",
            "tool_calls": [{"tool_name": "low_stock_items", "kwargs": {"threshold": 5}}],
            "notes": "Inventory low stock report.",
        }
    if "stock" in q and "category" in q:
        return {
            "intent": "stock_by_category",
            "tool_calls": [{"tool_name": "total_stock_by_category", "kwargs": {}}],
            "notes": "Inventory by category.",
        }
    if "blouse" in q and "sold" in q:
        return {
            "intent": "blouse_units",
            "tool_calls": [{"tool_name": "units_sold_by_category", "kwargs": {}}],
            "notes": "Units sold by category; filter blouse in narrative.",
        }
    if "category" in q and ("most" in q or "top" in q) and "unit" in q:
        return {
            "intent": "top_category_by_units",
            "tool_calls": [{"tool_name": "units_sold_by_category", "kwargs": {}}],
            "notes": "Category ranking by units.",
        }
    if "state" in q and ("top" in q or "highest" in q):
        return {
            "intent": "state_sales_rank",
            "tool_calls": [{"tool_name": "state_sales_rank", "kwargs": {"top_n": 5}}],
            "notes": "State ranking by sales.",
        }
    if "cancel" in q:
        return {
            "intent": "cancellation_rate",
            "tool_calls": [{"tool_name": "cancellation_rate", "kwargs": {}}],
            "notes": "Cancellation KPI.",
        }
    if "b2b" in q or "b2c" in q:
        return {
            "intent": "b2b_vs_b2c",
            "tool_calls": [{"tool_name": "b2b_vs_b2c_summary", "kwargs": {}}],
            "notes": "B2B/B2C comparison.",
        }
    if "amazon" in q and "flipkart" in q and ("price" in q or "difference" in q):
        return {
            "intent": "channel_price_comparison",
            "tool_calls": [
                {
                    "tool_name": "channel_price_comparison_may2022",
                    "kwargs": {"channel_a": "Amazon_MRP", "channel_b": "Flipkart_MRP"},
                }
            ],
            "notes": "Channel price difference comparison.",
        }

    return {
        "intent": "default_sales_overview",
        "tool_calls": [
            {"tool_name": "total_sales_amazon", "kwargs": {}},
            {"tool_name": "order_status_breakdown", "kwargs": {}},
        ],
        "notes": "Default overview plan.",
    }

