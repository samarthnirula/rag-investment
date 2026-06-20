# Atticus

Atticus is a legal document intelligence app for small firms, solo lawyers, legal-aid teams, and public-record investigations. Users upload PDFs, organize them into cases, ask source-grounded questions, and receive answers with citations, risk flags, and follow-up actions.

The product is designed around one business rule: **keep at least a 60% gross margin**. Public pricing can stay simple while query limits, upload limits, page caps, and cost telemetry prevent heavy document usage from turning a low-price plan unprofitable.

## What It Does

- Upload legal PDFs and public-record documents.
- Parse, chunk, embed, and store documents in PostgreSQL with pgvector.
- Run hybrid retrieval with vector search, BM25 keyword search, RRF fusion, version scoring, chunk-type scoring, reranking, and deduplication.
- Ask questions in a chat UI with citations tied to source pages.
- Render structured legal answers with summary, risk flags, detailed markdown analysis, and follow-up actions.
- Organize documents into cases/matters.
- Track query usage, estimated AI cost, and plan cost caps.
- Gate access through Firebase authentication and optional access codes.
- Run database migrations through `scripts/setup_database.py`.
- Offer a no-login public demo mode for the shared public-record case workspace.
- Generate first-pass case insights: timeline items, entities, contradiction candidates, and client-summary exports.
- Provide subscription/org/background-job database foundations for Stripe, teams, and async processing.

## Current Stack

| Layer | Tooling |
|---|---|
| UI | Streamlit |
| Auth | Firebase Auth |
| App language | Python 3.10+ |
| Database | PostgreSQL + pgvector |
| Embeddings | `sentence-transformers` local model |
| Retrieval | pgvector + BM25 + cross-encoder reranking |
| Generation | Anthropic Claude |
| PDF parsing | PyMuPDF, pdfplumber, optional Tesseract OCR |
| Deployment | Docker, docker-compose, nginx |

## Margin Guardrails

The default plan is controlled by environment variables:

```bash
ATTICUS_TARGET_GROSS_MARGIN=0.60
ATTICUS_STARTER_PRICE_USD=29
ATTICUS_MONTHLY_QUERY_LIMIT=300
ATTICUS_MONTHLY_UPLOAD_LIMIT=20
ATTICUS_MAX_UPLOAD_MB=25
ATTICUS_MAX_PAGES_PER_PDF=250
```

At `$29/mo`, a 60% gross margin means the variable cost cap is `$11.60/user/month`. The app now tracks estimated query cost in `query_log.estimated_cost_usd` and shows a cost-cap signal in the Data Usage page.

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Create `.env`:

```bash
DATABASE_URL=postgresql://user:password@host:5432/dbname
ANTHROPIC_API_KEY=sk-ant-...
ZEP_API_KEY=
ZEP_ENABLED=true
EMBEDDING_MODEL=all-MiniLM-L6-v2
GENERATION_MODEL=claude-sonnet-4-6
RETRIEVAL_TOP_K=8

FIREBASE_API_KEY=...
FIREBASE_PROJECT_ID=...
FIREBASE_AUTH_DOMAIN=...
APP_URL=http://localhost:8501
```

Initialize the database:

```bash
python scripts/setup_database.py
```

This applies the base schema and every SQL file in `migrations/`, recording completed migrations in `schema_migrations`.

Run the app:

```bash
./run.sh
```

Open `http://localhost:8501`.

## Important Files

- `src/insightlens/ui/streamlit_app.py` - main Atticus app shell and chat UI.
- `src/insightlens/ui/cases_page.py` - case and document upload management.
- `src/insightlens/billing.py` - plan limits, 60% margin rule, cost estimates.
- `src/insightlens/analysis/case_insights.py` - first-pass timeline/entity/contradiction extraction.
- `src/insightlens/retrieval/hybrid_search.py` - main retrieval pipeline.
- `src/insightlens/ingestion/ingest_service.py` - in-app PDF ingestion.
- `src/insightlens/storage/schema.sql` - PostgreSQL/pgvector schema.
- `src/insightlens/storage/migrations.py` - migration runner.
- `src/insightlens/generation/prompts.py` - legal answer prompt contracts.
- `PROJECT.md` - full system guide, file map, architecture, and roadmap.

## Tests

```bash
pytest
```

Some integration tests require a configured database and API keys; offline unit tests cover chunking, hybrid retrieval behavior, and metadata extraction.
