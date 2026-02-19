from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.agents.crew_setup import RetailInsightsCrew
from src.data.loaders import DatasetBundle, load_dataset_bundle
from src.tools.pandas_tools import PandasTools


PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "sales-dataset"
load_dotenv(PROJECT_ROOT / ".env")

QUICK_QUERIES = [
    "How many blouse were sold?",
    "Which category sold the most units?",
    "Which 5 states generated the highest sales?",
    "What is the cancellation rate?",
    "Compare B2B and B2C sales.",
    "Who are the top 5 customers by revenue?",
    "What were total international sales in Jun-21?",
    "Which items are low in stock (below 5)?",
    "How much stock do we have by category?",
    "What is the price difference between Amazon and Flipkart in May 2022?",
    "Summarize the Expense IIGF statement.",
    "What does the warehouse file say about fill-rate penalty?",
]


@st.cache_resource(show_spinner=False)
def init_runtime() -> tuple[DatasetBundle, PandasTools, RetailInsightsCrew]:
    bundle = load_dataset_bundle(DATASET_DIR)
    tools = PandasTools(bundle)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    crew = RetailInsightsCrew(tools=tools, model=model)
    return bundle, tools, crew


def _dataset_status(bundle: DatasetBundle) -> list[tuple[str, int | str]]:
    return [
        ("Amazon Sale Report", len(bundle.amazon_orders)),
        ("International sale Report", len(bundle.international_sales)),
        ("Sale Report", len(bundle.inventory_stock)),
        ("May-2022", len(bundle.pricing_may_2022)),
        ("P  L March 2021", len(bundle.pricing_mar_2021)),
        ("Expense IIGF text chars", len(bundle.expense_statement_text)),
        ("Warehouse text chars", len(bundle.warehouse_comparison_text)),
    ]


def _generate_dataset_summary(tools: PandasTools) -> str:
    top_category = tools.run_tool("category_sales_rank", top_n=1)
    top_state = tools.run_tool("state_sales_rank", top_n=1)
    cancel = tools.run_tool("cancellation_rate")
    stock = tools.run_tool("total_stock_by_category")

    parts = []
    if top_category["records"]:
        row = top_category["records"][0]
        parts.append(
            f"Top Amazon category by sales is {row.get('Category', 'N/A')} "
            f"with amount {row.get('Amount', 0):,.2f}."
        )
    if top_state["records"]:
        row = top_state["records"][0]
        parts.append(
            f"Top state by Amazon sales is {row.get('ship_state', 'N/A')} "
            f"with amount {row.get('Amount', 0):,.2f}."
        )
    if cancel["records"]:
        row = cancel["records"][0]
        parts.append(
            f"Cancellation rate is {row.get('cancellation_rate_pct', 0)}% "
            f"({row.get('cancelled_orders', 0)}/{row.get('total_orders', 0)} orders)."
        )
    if stock["records"]:
        row = stock["records"][0]
        parts.append(
            f"Highest inventory category is {row.get('Category', 'N/A')} with stock "
            f"{row.get('Stock', 0):,.0f}."
        )

    return " ".join(parts) if parts else "No summary could be generated from loaded data."


def main() -> None:
    st.set_page_config(page_title="Retail Insights Assistant", layout="wide")
    st.title("Retail Insights Assistant")

    try:
        bundle, tools, crew = init_runtime()
    except Exception as exc:
        st.error(f"Failed to initialize app: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Mode")
        mode = st.radio("Select interaction mode", ["Q&A", "Summarization"], index=0)

        st.subheader("Data Status")
        for label, count in _dataset_status(bundle):
            st.caption(f"{label}: {count}")

        st.subheader("Quick Demo Queries")
        for idx, query in enumerate(QUICK_QUERIES):
            if st.button(query, key=f"quick_{idx}", use_container_width=True):
                st.session_state.pending_query = query

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if mode == "Summarization":
        if st.button("Generate data summary", type="primary"):
            summary = _generate_dataset_summary(tools)
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.rerun()
        st.stop()

    prompt = st.chat_input("Ask a business question about sales, stock, pricing, or statements...")
    if not prompt and st.session_state.get("pending_query"):
        prompt = st.session_state.pop("pending_query")

    if not prompt:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running CrewAI workflow..."):
            history = st.session_state.messages[:-1]
            result = crew.answer(query=prompt, history=history, mode=mode)
        st.markdown(result["final_answer"])

        with st.expander("Execution trace", expanded=False):
            st.write("Plan:")
            st.json(result["plan"])
            st.write("Tool outputs:")
            st.json(result["tool_results"])
            st.write("Extraction summary:")
            st.markdown(result["extraction_summary"])

    st.session_state.messages.append({"role": "assistant", "content": result["final_answer"]})


if __name__ == "__main__":
    main()

