# Retail Insights Assistant (Demo-First)

This project is a Streamlit + CrewAI assignment implementation for retail insights on the provided `sales-dataset`.
Unzip sales-dataset here.

## What this app does

- Supports two modes:
  - **Summarization**: quick overall business summary.
  - **Q&A**: natural language analytical questions.
- Uses a **3-agent flow**:
  - Language-to-query resolution agent
  - Data extraction agent
  - Validation agent
- Uses deterministic **Pandas functions** (no generated Python code).

## Data handling choices

- Tabular files are loaded with Pandas:
  - `Amazon Sale Report.csv`
  - `International sale Report.csv`
  - `Sale Report.csv`
  - `May-2022.csv`
  - `P  L March 2021.csv`
- Statement-style files are treated as text only (comma-stripped):
  - `Expense IIGF.csv`
  - `Cloud Warehouse Compersion Chart.csv`

This was intentionally done for a quick assignment demo and to match requested constraints.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your OpenAI key (either `.env` or shell export):

`.env` example in project root:

```bash
OPENAI_API_KEY=your_key_here
```

Or export in shell:

```bash
export OPENAI_API_KEY="your_key_here"
```

4. Run Streamlit:

```bash
streamlit run app.py
```

## Suggested demo questions

Use these in the recording:

1. How many blouse were sold?
2. Which category sold the most units?
3. Which 5 states generated the highest sales?
4. What is the cancellation rate?
5. Compare B2B and B2C sales.
6. Who are the top 5 customers by revenue?
7. What were total international sales in Jun-21?
8. Which items are low in stock (below 5)?
9. How much stock do we have by category?
10. What is the price difference between Amazon and Flipkart in May 2022?
11. Summarize the Expense IIGF statement.
12. What does the warehouse file say about fill-rate penalty?

## Limitations

- Built for demo reliability, not production deployment.
- No RAG/vector retrieval.
- Statement-like CSV files are consumed as raw text, not deeply parsed.
- Intent routing includes LLM planning plus heuristic fallback for stability.

## Possible Improvements

### Data & Storage
- **Migrate to a cloud data warehouse** (BigQuery / Snowflake) for scalable querying beyond in-memory Pandas limits.
- **Use Parquet / Delta Lake** for columnar storage with partition pruning on date, category, and region.
- **Add a data lake layer** (S3 / GCS) for raw and curated data zones with versioning.
- **Implement incremental ingestion** via Apache Airflow + PySpark instead of loading all CSVs at startup.

### Retrieval & Intelligence
- **Add RAG (Retrieval-Augmented Generation)** with vector embeddings (FAISS / Pinecone) for unstructured text queries — expense statements, warehouse comparisons, and future document types.
- **Implement Text-to-SQL** so the LLM generates SQL directly against the warehouse for ad-hoc analytical queries not covered by the 18 pre-built tools.
- **Semantic query caching** — embed user queries and return cached answers for semantically similar questions (cosine similarity > 0.95).

### Model & Cost Optimization
- **Model tiering** — route simple KPI lookups to a local/smaller model (e.g., Llama 3 8B), use GPT-4.1-mini for standard queries, and reserve GPT-4.1 for complex multi-table analysis.
- **Prompt caching and result caching** via Redis to reduce redundant LLM calls.
- **Token budget enforcement** — cap input/output tokens per agent call to control costs at scale.
- **Streaming responses** for better perceived latency in the Streamlit UI.

### Production Readiness
- **Separate API layer** (FastAPI) from the Streamlit frontend to support multiple clients and horizontal scaling.
- **Add authentication and rate limiting** for multi-user deployment.
- **Containerize with Docker** and deploy on Cloud Run / ECS Fargate with autoscaling.
- **CI/CD pipeline** for automated testing, linting, and deployment.

### Monitoring & Quality
- **Add observability** — Prometheus + Grafana for latency, error rate, and cache hit rate dashboards.
- **LLM evaluation pipeline** — automated accuracy checks comparing LLM answers to ground-truth SQL results.
- **Hallucination detection** — flag answers that reference data not present in tool results.
- **Human-in-the-loop feedback** — queue low-confidence answers for human review and feed corrections back into prompt templates.

### UX Enhancements
- **Chart and visualization generation** — return Plotly/Altair charts alongside text answers for ranking and trend queries.
- **Multi-turn conversation memory** — persist chat history across sessions with a database-backed store.
- **Export to PDF/Excel** — let users download query results as formatted reports.
- **Voice input** — integrate speech-to-text for hands-free querying in warehouse/retail settings.

