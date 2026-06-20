# Atticus Project Guide

This document is the full project map for Atticus. A new engineer, founder, contractor, or investor should be able to read it and understand what the website does, how the backend works, how the frontend is assembled, where the files live, and what still needs to be built.

## 1. Product Summary

Atticus is a legal document intelligence workspace. It lets users upload PDFs, organize them into cases, ask source-grounded questions, and receive structured legal research outputs with citations, risk flags, and follow-up actions.

The near-term wedge is not to copy Harvey for BigLaw. The better path is to serve solo lawyers, small firms, legal-aid teams, and public-record researchers with transparent pricing, public demos, fast onboarding, local/practical workflows, and strict cost guardrails.

The operating rule is: **every plan should preserve at least 60% gross margin**. That means variable AI, embedding, OCR, storage, and database costs should stay below 40% of monthly revenue.

## 2. What Exists Today

Implemented today:

- Authenticated Streamlit web app branded as Atticus.
- Firebase email/password and Google sign-in flow.
- Optional access-code gating.
- PostgreSQL + pgvector storage.
- PDF ingestion from the UI.
- Local sentence-transformer embeddings.
- Hybrid retrieval using vector search and BM25.
- RRF fusion, version scoring, chunk-type scoring, reranking, and deduplication.
- Claude answer generation.
- Structured legal answer renderer with summary, risk flags, answer body, and follow-up actions.
- Persistent chat history.
- Case/matter collections.
- User-owned documents and shared system documents.
- Query audit log.
- Usage analytics.
- Monthly query limits, upload size limits, page limits, and estimated cost telemetry.
- Database migrations.
- Monthly upload event tracking.
- Public no-login demo mode for the shared public-record case workspace.
- First-pass case insight extraction: timelines, entities, contradiction candidates, and client-summary exports.
- Subscription, organization, and background-job database foundations.
- Docker/nginx deployment files.

Not fully implemented yet:

- True autonomous legal workflow agents.
- Full contradiction extraction as a production-grade async workflow.
- Full timeline visualization for user matters.
- Entity relationship graph.
- DOCX/PDF export pipeline.
- Live Stripe webhook syncing.
- Clio/MyCase integrations.
- Microsoft Word/Outlook add-ins.
- Full organization/team permission model.
- Production-grade provider billing reconciliation.

## 3. User-Facing Website

The website is currently a Streamlit app. The frontend entry point is:

- `src/insightlens/ui/streamlit_app.py`

Main pages:

- Landing page: `src/insightlens/ui/landing_page.py`
- Chat app shell: `src/insightlens/ui/streamlit_app.py`
- Sidebar navigation: `src/insightlens/ui/sidebar.py`
- Cases and document management: `src/insightlens/ui/cases_page.py`
- Usage analytics: `src/insightlens/ui/data_page.py`
- Profile and account deletion: `src/insightlens/ui/profile_page.py`
- Terms/privacy pages: `src/insightlens/ui/legal_page.py`
- Product/about page: `src/insightlens/ui/about_page.py`

Important UI behavior:

- Unauthenticated visitors see the landing page.
- Authenticated users enter the app shell.
- Chat pages maintain persistent chats when the user is signed in.
- Case pages let users create cases and upload PDFs.
- Data page shows usage metrics and estimated cost relative to the 60% margin cost cap.
- Legal answers render as structured cards when Claude returns the JSON contract from `CASE_SYSTEM_PROMPT`.

## 4. Backend Architecture

Atticus is a Python package under:

- `src/insightlens/`

The current architecture is modular even though the UI and backend run in one Streamlit process.

### Configuration

- `src/insightlens/config.py`

Loads:

- `DATABASE_URL`
- `ANTHROPIC_API_KEY`
- `EMBEDDING_MODEL`
- `GENERATION_MODEL`
- `CHUNK_SIZE_TOKENS`
- `CHUNK_OVERLAP_TOKENS`
- `RETRIEVAL_TOP_K`

The config class still exposes `SnowflakeConfig` as a backwards-compatible alias, but the actual database is PostgreSQL.

### Billing And Margin Guardrails

- `src/insightlens/billing.py`

This file contains the 60% gross-margin rule and plan defaults.

Environment variables:

- `ATTICUS_TARGET_GROSS_MARGIN` default `0.60`
- `ATTICUS_STARTER_PRICE_USD` default `29`
- `ATTICUS_MONTHLY_QUERY_LIMIT` default `300`
- `ATTICUS_MONTHLY_UPLOAD_LIMIT` default `20`
- `ATTICUS_MAX_UPLOAD_MB` default `25`
- `ATTICUS_MAX_PAGES_PER_PDF` default `250`
- `ATTICUS_EST_COST_PER_TOKEN` default `0.000003`
- `ATTICUS_EST_COST_PER_SOURCE` default `0.00001`
- `ATTICUS_EST_INGEST_COST_PER_PAGE` default `0.002`
- `ATTICUS_EST_STORAGE_COST_PER_MB_MONTH` default `0.0005`

The goal is not perfect cost accounting yet. The goal is early warning when a user is becoming unprofitable.

### Database Connection

- `src/insightlens/storage/snowflake_client.py`

Despite the filename, this is a PostgreSQL connection pool. It keeps the old public API names to avoid touching every repository import.

Key functions:

- `open_connection(cfg.db)`
- `execute_script(conn, sql_text)`

### Database Schema

- `src/insightlens/storage/schema.sql`

Tables:

- `schema_migrations` - applied SQL migration versions.
- `documents` - one row per PDF/document.
- `chunks` - text chunks and pgvector embeddings.
- `images` - extracted images and optional AI descriptions.
- `query_log` - immutable usage/audit log with estimated cost.
- `upload_events` - monthly upload tracking and estimated ingestion cost.
- `cases` - user-created matter/case workspaces.
- `case_documents` - many-to-many case/document links.
- `case_insights` - extracted timeline/entity/contradiction rows.
- `generated_artifacts` - client summaries and other generated outputs.
- `subscriptions` - Stripe/subscription sync target.
- `organizations` - team/org accounts.
- `organization_members` - org membership and roles.
- `background_jobs` - async job queue table.
- `chats` - persisted chat sessions.
- `chat_messages` - persisted messages.
- `discussion_posts` - community/discussion feature.
- `consent_log` - terms/privacy acceptance log.
- `access_codes` - invitation/access-code gating.

Vector storage:

- `chunks.embedding vector(384)`
- IVFFlat cosine index on `chunks.embedding`.

### Storage Repositories

Files:

- `src/insightlens/storage/chunk_repository.py`
- `src/insightlens/storage/cases_repository.py`
- `src/insightlens/storage/image_repository.py`
- `src/insightlens/storage/insights_repository.py`
- `src/insightlens/storage/usage_repository.py`
- `src/insightlens/storage/billing_repository.py`
- `src/insightlens/storage/jobs_repository.py`
- `src/insightlens/storage/org_repository.py`
- `src/insightlens/storage/chat_repository_persistent.py`
- `src/insightlens/storage/audit_repository.py`
- `src/insightlens/storage/consent_repository.py`
- `src/insightlens/storage/access_code_repository.py`
- `src/insightlens/storage/discussion_repository.py`

Important recent change:

- `ChunkRepository.delete_document(document_id, user_id=None)` now supports safe deletion for both user-owned and system documents.
- If `user_id` is provided, the document must belong to that user.
- If `user_id` is omitted, only shared/system documents with `user_id IS NULL` can be deleted.
- Deletion also removes dependent `images`, `case_documents`, and `chunks`.

### Ingestion

Main file:

- `src/insightlens/ingestion/ingest_service.py`

Supporting files:

- `src/insightlens/ingestion/pdf_parser.py`
- `src/insightlens/ingestion/chunker.py`
- `src/insightlens/ingestion/document_metadata.py`
- `src/insightlens/ingestion/image_extractor.py`
- `src/insightlens/ingestion/ocr_extractor.py`
- `src/insightlens/ingestion/vision_extractor.py`

Pipeline:

1. User uploads a PDF in `cases_page.py`.
2. UI checks max file size from `billing.py`.
3. `IngestService.ingest()` parses the PDF.
4. If user-owned and page count exceeds plan max, ingestion is skipped.
5. Metadata is extracted.
6. Pages are chunked.
7. Chunks are embedded.
8. Document and chunks are written to PostgreSQL/pgvector.
9. Images are extracted and stored as metadata.
10. UI reports page count, chunk count, image count, and estimated variable cost.
11. Successful uploads are logged to `upload_events` for monthly plan enforcement.

### Case Insight Analysis

Main file:

- `src/insightlens/analysis/case_insights.py`

Current extractors:

- Timeline extraction from date-bearing sentences.
- Entity extraction from repeated capitalized person/entity names.
- Contradiction candidates from conflicting money/number values in similar statements.
- Client summary markdown generation.

The UI entry point is the selected case's Insights expander in:

- `src/insightlens/ui/cases_page.py`

This is a deterministic first pass. The future production version should move extraction to `background_jobs` and add LLM verification.

### Embeddings

Main file:

- `src/insightlens/embeddings/embedder.py`

Current default:

- `all-MiniLM-L6-v2`
- 384-dimensional vectors.

Reason:

- Local embeddings reduce variable cost and help protect the 60% margin rule.

### Retrieval

Main files:

- `src/insightlens/retrieval/hybrid_search.py`
- `src/insightlens/retrieval/vector_search.py`
- `src/insightlens/retrieval/reranker.py`

Hybrid retrieval pipeline:

1. Embed query locally.
2. Run pgvector cosine search.
3. Run BM25 keyword search over cached corpus chunks.
4. Fuse rankings with Reciprocal Rank Fusion.
5. Drop weak vector candidates below similarity floor.
6. Apply version scoring.
7. Apply chunk-type scoring.
8. Apply per-document quota for broad queries.
9. Cross-encoder rerank.
10. Deduplicate by document/page.

This is one of the strongest parts of the current codebase.

### Generation

Files:

- `src/insightlens/generation/llm_client.py`
- `src/insightlens/generation/prompts.py`
- `src/insightlens/generation/answer_builder.py`

Claude is called through `ClaudeClient`.

Legal responses use `CASE_SYSTEM_PROMPT`, which requires this JSON shape:

```json
{
  "summary": "short plain-English summary",
  "risk_flags": [],
  "answer": "markdown answer with citations",
  "follow_up_actions": []
}
```

The UI parser is in `streamlit_app.py` and renders this JSON as polished legal answer cards.

## 5. Authentication

Main file:

- `src/insightlens/ui/auth.py`

Auth supports:

- Email/password sign-in.
- Registration.
- Password reset.
- Google sign-in.
- Firebase token refresh.
- Email verification gate.
- Optional access codes through `access_codes`.

Required Firebase variables:

- `FIREBASE_API_KEY` or `FIREBASE_WEB_API_KEY`
- `FIREBASE_PROJECT_ID`
- `FIREBASE_AUTH_DOMAIN`
- `APP_URL`

## 6. Rate Limits And Cost Controls

Files:

- `src/insightlens/ui/rate_limiter.py`
- `src/insightlens/billing.py`
- `src/insightlens/storage/audit_repository.py`
- `src/insightlens/ui/data_page.py`
- `src/insightlens/ui/cases_page.py`

Current controls:

- Monthly query cap.
- Hourly query cap.
- Per-minute burst cap.
- Max file size.
- Max PDF page count.
- Estimated query cost.
- Estimated ingestion cost.
- Usage dashboard with estimated cost versus cost cap.

These controls address the concern that `$29/mo` can become unprofitable if users upload bulk documents or run many model calls.

## 7. Scripts

Setup:

- `scripts/setup_database.py` - creates schema from `schema.sql`.
- `migrations/*.sql` - incremental schema updates recorded in `schema_migrations`.

Document ingestion and data:

- `scripts/ingest_documents.py` - ingests PDFs from local folders.
- `scripts/ingest_epstein.py` - ingests Epstein public-record data.
- `scripts/ingest_images.py` - ingests extracted images.
- `scripts/backfill_image_descriptions.py` - backfills image descriptions.
- `scripts/fetch_epstein_data.py` - fetches dataset.
- `scripts/fetch_epstein_pdfs.py` - fetches PDFs.
- `scripts/scrape_epstein_files.py` - scrapes source files.

Evaluation/admin:

- `scripts/eval_retrieval.py` - retrieval evaluation.
- `scripts/manage_access_codes.py` - access-code management.

## 8. Tests

Test files:

- `tests/test_chunker.py`
- `tests/test_hybrid_search.py`
- `tests/test_metadata_extraction.py`
- `tests/test_retrieval_eval.py`
- `tests/golden_qa.json`

Run:

```bash
pytest
```

Some tests are offline. Integration-style tests require database/API configuration.

## 9. Deployment Files

Files:

- `Dockerfile`
- `docker-compose.yml`
- `nginx/nginx.conf`
- `runtime.txt`
- `run.sh`
- `requirements.txt`
- `requirements-dev.txt`
- `pyproject.toml`

Local app command:

```bash
./run.sh
```

## 10. Environment Variables

Minimum local `.env`:

```bash
DATABASE_URL=postgresql://user:password@host:5432/dbname
ANTHROPIC_API_KEY=sk-ant-...
ZEP_API_KEY=
ZEP_ENABLED=true
EMBEDDING_MODEL=all-MiniLM-L6-v2
GENERATION_MODEL=claude-sonnet-4-6
RETRIEVAL_TOP_K=8
```

Recommended product variables:

```bash
ATTICUS_TARGET_GROSS_MARGIN=0.60
ATTICUS_STARTER_PRICE_USD=29
ATTICUS_MONTHLY_QUERY_LIMIT=300
ATTICUS_MONTHLY_UPLOAD_LIMIT=20
ATTICUS_MAX_UPLOAD_MB=25
ATTICUS_MAX_PAGES_PER_PDF=250
```

Firebase variables:

```bash
FIREBASE_API_KEY=...
FIREBASE_PROJECT_ID=...
FIREBASE_AUTH_DOMAIN=...
APP_URL=http://localhost:8501
```

Optional database pool variables:

```bash
PG_POOL_SIZE=5
```

## 11. Product Checklist Status

Done or partially done:

- [x] Public-facing landing page.
- [x] Public no-login demo with sandbox/public-record corpus.
- [x] Authenticated app.
- [x] Case/matter workspace foundation.
- [x] PDF upload.
- [x] Source-cited legal Q&A.
- [x] Structured legal answer rendering.
- [x] Follow-up action buttons.
- [x] Persistent chats.
- [x] Usage analytics.
- [x] Query rate limits.
- [x] Upload size guardrails.
- [x] PDF page-count guardrails.
- [x] Monthly upload count from database.
- [x] Estimated query cost tracking.
- [x] Estimated ingestion cost reporting.
- [x] Migration framework.
- [x] Contradiction detection workflow foundation.
- [x] Timeline extraction foundation.
- [x] Entity extraction foundation.
- [x] Client-ready markdown export foundation.
- [x] Stripe/subscription table foundation.
- [x] Admin margin dashboard foundation.
- [x] Background job queue table foundation.
- [x] Organization/team table foundation.
- [x] README updated for Atticus.
- [x] `PROJECT.md` created.
- [x] Stale `SECURITY.md` removed.

Still to build:

- [ ] Live Stripe webhook syncing.
- [ ] Real provider cost reconciliation.
- [ ] Async contradiction verification.
- [ ] Rich timeline visualization for user-uploaded matters.
- [ ] Entity relationship graph.
- [ ] Client-ready PDF/DOCX exports.
- [ ] Review-table workflow for bulk document sets.
- [ ] Texas jurisdiction pack.
- [ ] Organization/team UI and enforcement.
- [ ] Matter-level RBAC.
- [ ] Admin alerts/notifications for heavy users.
- [ ] Clio/MyCase billing integration.
- [ ] Word/Outlook add-ins.
- [ ] Async worker process for ingestion/analysis jobs.
- [ ] Object storage for original PDFs and images.

## 12. Suggested Next Build Order

1. Replace synchronous insight generation with background jobs.
2. Add live Stripe webhook handling and plan-specific limits.
3. Add organization/team UI and matter-level RBAC enforcement.
4. Add rich timeline and entity graph views.
5. Add PDF/DOCX exports.
6. Add Texas jurisdiction pack.
7. Add review tables for bulk document sets.
8. Add admin alerts for users approaching margin caps.
9. Add object storage for original PDFs and images.
10. Rewrite or remove the old `docs/architecture.md`.

## 13. Known Technical Debt

- The package is still named `insightlens`; product branding is Atticus.
- `snowflake_client.py` is now a PostgreSQL client with compatibility names.
- `docs/architecture.md` still describes the older investment/Snowflake version and should either be rewritten or replaced by this file.
- Some prompts and tests still include investment examples because the project began as an investment-doc RAG app.
- Streamlit is fast for iteration but will eventually constrain a polished SaaS UX; a Next.js/FastAPI split is the likely future architecture.

## 14. Competitive Direction

Harvey is strong for enterprise legal teams, but Atticus should win on:

- Public demo access.
- Transparent pricing.
- Lower entry price.
- Strong source citations.
- Small-firm workflows.
- Local jurisdiction awareness.
- Plain-English client outputs.
- Strict cost controls that let low pricing survive.

The best product promise is:

**Affordable legal AI for small firms: upload matter documents, find the facts, flag risks, build timelines, and produce client-ready outputs with citations.**
