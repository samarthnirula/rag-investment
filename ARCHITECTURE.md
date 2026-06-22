# Atticus — Full System Architecture & Replication Guide

This document describes the **current, real** architecture of Atticus as it exists in this repository today — not the historical Streamlit/Snowflake prototype described in `README.md`/`PROJECT.md`, and not the original investment-RAG build guide in `docs/architecture.md`. Those files are stale. This document supersedes them for understanding or rebuilding the system.

Atticus is a legal document intelligence platform: users upload case documents (PDFs, PPTX), the system parses/chunks/embeds them, retrieval pulls relevant passages with a hybrid search pipeline, and Claude generates source-grounded legal answers with citations.

---

## 1. High-Level Architecture

```
┌─────────────────────────────┐         ┌──────────────────────────────────┐
│   Next.js Frontend          │  HTTPS  │   FastAPI Backend                 │
│   (Vercel)                  │ ──────► │   (Render, Docker)                 │
│   frontend/                 │  JSON / │   backend/main.py                  │
│                              │  SSE    │   backend/demo_router.py           │
└──────────────┬───────────────┘         └───────────────┬────────────────────┘
               │                                          │
               │ Firebase ID tokens                       │ imports
               ▼                                          ▼
      ┌──────────────────┐                    ┌─────────────────────────────┐
      │ Firebase Auth     │  verified by      │  src/insightlens/ (core lib) │
      │ (client SDK)      │◄──────────────────│  config, ingestion,          │
      └──────────────────┘  Firebase Admin SDK│  embeddings, retrieval,      │
                                               │  generation, storage,        │
                                               │  billing, analysis           │
                                               └───────────────┬───────────────┘
                                                                │
                          ┌─────────────────────────────────────┼───────────────────┐
                          ▼                                     ▼                   ▼
                 ┌────────────────┐                  ┌────────────────────┐  ┌─────────────┐
                 │ PostgreSQL +    │                  │  Voyage AI /        │  │  Anthropic  │
                 │ pgvector        │                  │  local SentenceTfmr │  │  Claude API │
                 │ (Supabase)      │                  │  (embeddings)       │  │ (generation)│
                 └────────────────┘                  └────────────────────┘  └─────────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │ Redis (rate    │
                 │ limiting,      │
                 │ optional)      │
                 └────────────────┘
```

**Two front doors into the backend:**
- **Authenticated app** — Firebase Auth (email/password + Google), real user accounts, persistent cases/chats, billing/plan limits.
- **Public demo** — access-code gated, stateless JWT (not Firebase), scoped to a shared public-record corpus plus one upload-able demo case, rate-limited and cost-tracked separately.

**Deployment topology:**
- Frontend: Next.js app deployed to **Vercel**.
- Backend: FastAPI app, containerized with Docker, deployed to **Render** (free tier — cold starts are expected; the frontend has timeout/retry handling for this).
- Database: PostgreSQL with the `pgvector` extension, hosted on **Supabase**.
- Auth: **Firebase** project (Authentication only — no Firestore/Storage used).

---

## 2. Repository Layout

```
rag-invetment/
├── frontend/                     # Next.js app (App Router)
│   └── src/
│       ├── app/                  # routes (see §6)
│       ├── components/           # shared UI components
│       ├── contexts/AuthContext.tsx
│       └── lib/
│           ├── api.ts             # authenticated backend client
│           ├── demo-api.ts        # demo-mode backend client
│           └── firebase.ts        # Firebase client SDK init
│
├── backend/                      # FastAPI HTTP layer (thin — delegates to src/insightlens)
│   ├── main.py                    # main app: auth, chats, cases, query, upload
│   ├── demo_router.py             # public demo endpoints
│   └── rate_limiter.py            # Redis-backed rate limiting
│
├── src/insightlens/               # Core Python library — all business logic lives here
│   ├── config.py                  # env var loading
│   ├── billing.py                 # plan limits / 60% margin guardrail
│   ├── ingestion/                 # PDF/PPTX parsing → chunking → embedding pipeline
│   ├── embeddings/embedder.py     # Voyage AI or local SentenceTransformer
│   ├── retrieval/                 # hybrid search: vector + BM25 + RRF + rerank
│   ├── generation/                # Claude client, prompts, answer assembly
│   ├── analysis/case_insights.py  # timeline/entity/contradiction extraction
│   ├── storage/                   # Postgres client + one repository class per table group
│   └── memory/zep_memory.py       # optional long-term chat memory (Zep)
│
├── migrations/*.sql               # incremental schema changes, tracked in schema_migrations
├── scripts/                       # one-off ingestion/admin/eval scripts
├── tests/                         # pytest suite (offline + integration)
├── docker-compose.yml, Dockerfile, render.yaml, nginx/
└── data/raw_pdfs/, data/processed/, data/images/
```

**Why this split matters for replication:** the FastAPI layer in `backend/` is intentionally thin — it does auth/HTTP/validation/ownership-checks and then calls into `src/insightlens`, which has zero FastAPI/HTTP dependencies. You could swap the HTTP framework without touching the RAG pipeline. This is also why a frontend rewrite (Streamlit → Next.js) didn't require touching ingestion/retrieval/generation code.

---

## 3. Backend — FastAPI (`backend/main.py`, `backend/demo_router.py`)

### 3.1 Bootstrap

On startup, `create_app()`:
1. Loads config via `insightlens.config.load_config()` (reads env vars, see §9).
2. Constructs shared singleton services: `Embedder`, `ClaudeClient`, `Reranker`, optional `ZepMemory`.
3. If any service fails to construct, the app still boots but sets an internal flag so protected routes return `503 Service not ready` instead of crashing — this matters on Render's free tier where misconfigured env vars shouldn't take down health checks.
4. Configures CORS: allowed origins include `localhost` for dev plus a regex matching Vercel preview deployment URLs (`^https://rag-investment(?:-[a-z0-9-]+)*\.vercel\.app$`).
5. Sets up a rotating file logger (`error.log`, 10MB × 5 backups).

### 3.2 Authentication model

Two completely separate auth systems coexist:

**Firebase (authenticated app):**
- Firebase Admin SDK initialized from one of three sources, in priority order: `FIREBASE_SERVICE_ACCOUNT_JSON` (raw JSON string, used on Render), `FIREBASE_SERVICE_ACCOUNT_PATH` (file path, used in Docker), or Application Default Credentials.
- Every protected request carries `Authorization: Bearer <firebase_id_token>`.
- `_verify_firebase_token()` calls `firebase_admin.auth.verify_id_token()`; invalid/expired tokens → `401`.
- A `current_user` dependency extracts `{uid, email, display_name}`; `require_user` enforces it's present; `require_active_subscription` additionally checks the user's trial/plan hasn't expired (queries `users` table).
- Custom claims (`{plan, plan_updated_at}`) are written to the Firebase user at registration so the ID token itself carries plan info.

**Demo JWT (public demo):**
- Independent of Firebase entirely. Access codes (`DEMO_ACCESS_CODE_USER1/2/3` env vars) are bcrypt-hashed and checked against `demo.sessions`.
- On successful redemption, the backend mints its own JWT containing `user_slug`; this token does not expire and is not refreshed — if a demo user wants a new session they re-enter the access code.
- Demo routes verify this JWT directly (not via Firebase Admin SDK).

### 3.3 Route map — authenticated app (`/api/...`)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/health` | none | health check, `{"status":"ok","service":"atticus-api"}` |
| POST | `/api/auth/session` | Firebase | fetch `{uid, email, display_name}` |
| POST | `/api/auth/refresh` | none | exchange Firebase refresh token for new ID token |
| POST | `/api/auth/register` | Firebase | idempotent upsert into Postgres `users` + Zep; sets custom claims; returns `{uid, plan, is_new}` |
| POST | `/api/admin/users/{uid}/plan` | admin key header | force-set a user's plan (trial/starter/pro/enterprise) |
| GET / POST | `/api/chats` | Firebase | list / create chat sessions |
| GET / POST / PATCH / DELETE | `/api/chats/{chat_id}`(`/messages`) | Firebase | message history, rename, delete |
| GET | `/api/companies` | Firebase | distinct companies across indexed documents |
| POST | `/api/query` | Firebase + active sub | synchronous RAG query → `{answer, sources, source_details, images, image_note, confidence}` |
| POST | `/api/query/stream` | Firebase + active sub | same, but Server-Sent Events: `{token}` chunks then a final `{done, text, sources, source_details, images, confidence}` |
| GET | `/api/usage` | Firebase | query counts/cost for the current billing period |
| POST | `/api/consent` | Firebase | log GDPR/ToS acceptance |
| GET / POST / DELETE | `/api/cases`, `/api/cases/{id}` | Firebase | case CRUD (trial accounts capped at 2 cases) |
| GET / POST / DELETE | `/api/cases/{id}/documents(/{doc_id})` | Firebase | attach/detach documents to a case |
| POST | `/api/cases/bulk-upload` | Firebase + active sub | multipart upload of many files → creates async ingest jobs |
| GET | `/api/cases/{id}/jobs` | Firebase | aggregate status of those async jobs |
| GET | `/api/cases/{id}/overview` / `/timeline` | Firebase | AI-generated case summary / chronological event list |
| GET / DELETE | `/api/documents`(`/{id}`) | Firebase | list/delete user + shared documents |
| POST | `/api/upload` | Firebase + active sub | single-file synchronous ingest |
| GET / DELETE | `/api/images/{id}`, `/api/demo/images/{id}` | Firebase / none | serve extracted images, ownership-checked |

All case/document/chat handlers run explicit **ownership checks** (`_assert_case_owned_by`, `_assert_document_owned_by`, `_assert_chat_owned_by`) before reading/writing — this is the IDOR defense layer, since IDs are plain strings, not scoped per-user in the URL.

### 3.4 Route map — public demo (`/api/demo/...`)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/demo/auth` | none | redeem access code (+ optional contact info) → `{user_slug, token}` |
| GET | `/api/demo/me` | demo JWT | session info, query count |
| GET / POST | `/api/demo/cases`, `/api/demo/cases/upload` | demo JWT | one demo case per session, max 8 files |
| POST | `/api/demo/query` | demo JWT | query the shared corpus or the demo case |
| GET | `/api/demo/timeline`, `/api/demo/overview` | demo JWT | AI-generated case views, lazily computed |
| GET | `/api/demo/admin/costs` | admin key header | per-user/per-model cost dashboard |

Demo queries are rate-limited at 20/hour and 3/minute per `user_slug`, and every query's token usage and `$` cost is recorded in `demo.usage` keyed by model name (cost table in `demo_router.py` covers Claude Haiku/Sonnet/Opus and `voyage-law-2`).

### 3.5 Rate limiting (`backend/rate_limiter.py`)

Redis sliding-window counters (`rl:q:hour:{uid}`, `rl:q:min:{uid}`, `rl:demo:hour:{ip}`, etc.) using sorted sets keyed by timestamp; if Redis is unreachable, falls back to in-memory counters and **fails open** (allows the request) rather than blocking users on infra issues.

---

## 4. Core Library — `src/insightlens/`

This is where every algorithmic decision lives. The FastAPI layer is just plumbing on top of this.

### 4.1 Configuration (`config.py`)

Loads and validates environment variables into a frozen `AppConfig` dataclass. Required: `DATABASE_URL`, `ANTHROPIC_API_KEY`. Optional with defaults: `EMBEDDING_MODEL`, `GENERATION_MODEL`, `RETRIEVAL_TOP_K`, `CHUNK_SIZE_TOKENS` (400), `CHUNK_OVERLAP_TOKENS` (50), `VOYAGE_API_KEY`, `ZEP_API_KEY`/`ZEP_ENABLED`, `REDIS_URL`. The database config struct is still internally named `PostgresConfig` but historically `SnowflakeConfig` — the storage client filename (`storage/snowflake_client.py`) is also a holdover name; it is a plain `psycopg2` connection pool today, not Snowflake.

### 4.2 Ingestion pipeline (`ingestion/`)

Order of operations for every uploaded document:

1. **Parse** — `pdf_parser.py` (PyMuPDF/`fitz` for text, `pdfplumber` for tables) or a PPTX parser (slide XML + speaker notes) → `ParsedDocument` of `ParsedPage`s, each flagged `is_likely_visual` if text density is low.
2. **Metadata extraction** — `document_metadata.py` infers company, document type, version label/date from filename + first-page heuristics (regex-based, no LLM call).
3. **Chunking** — `chunker.py`'s `SlideAwareChunker`: recursive token-bounded splitting using `tiktoken` (`cl100k_base`), trying separators `\n\n → \n → ". " → " " → hard split` in order, default 400 tokens with 50 overlap. Footnotes (bottom-quarter-of-page lines matching footnote patterns) are tagged `[FOOTNOTE]` so retrieval/generation can treat them as authoritative overrides.
4. **Embedding** — see §4.3.
5. **Image extraction** (PDFs) — `image_extractor.py` pulls raster images via `fitz`, stores them to `data/images/`, and calls Claude's vision API (Haiku) to generate a searchable text description per image, which is itself embedded.
6. **OCR/vision fallback** — for low-text pages: if borderline, Claude Sonnet vision (`vision_extractor.py`) describes the page; if vision is disabled, falls back to Tesseract OCR (`ocr_extractor.py`, requires the `tesseract-ocr` binary).
7. **Persistence** — document row → `documents`, chunk rows (with embeddings) → `chunks`, image rows (with description embeddings) → `images`, all via the repository classes in `storage/`.

Limits enforced before/during ingestion (`billing.py`): max file size, max pages per file, monthly upload count — different caps for authenticated users vs. demo uploads (demo: 8 files, 25MB each).

### 4.3 Embeddings (`embeddings/embedder.py`)

Dual-mode, selected automatically by whether `VOYAGE_API_KEY` is set:
- **Voyage AI** (`voyage-law-2`, 1024-dim) — REST API at `api.voyageai.com/v1/embeddings`, batches of 128 texts, `input_type="document"` for ingestion vs `"query"` for search. This is the production path — a domain-specific legal embedding model.
- **Local SentenceTransformer** (`all-MiniLM-L6-v2`, 384-dim) — fallback with no API cost, lazy-loaded on first use specifically to defer the `torch` import (this was a memory-fix for Render's free tier, see git history: "Defer CrossEncoder/torch load... to first use, not app startup").

This is why `migrations/007_vector_dim_1024.sql` exists — the schema migrated from 384-dim to 1024-dim vectors when the project moved from local-only embeddings to Voyage. **The vector column dimension and the embedding model in use must always match**; switching one requires re-ingesting everything and altering the column.

### 4.4 Retrieval (`retrieval/hybrid_search.py`, `vector_search.py`, `reranker.py`)

This is the most engineered part of the system. Pipeline, in exact order:

1. **Vector search** — embed the query, pgvector cosine search over `chunks.embedding` (IVFFlat index, `lists=100`).
2. **BM25 keyword search** — `rank_bm25`'s `BM25Okapi` over the same candidate corpus, query tokenized/lowercased with a ~60-word stopword list and a 3-character minimum token length.
3. **Reciprocal Rank Fusion (RRF)** — combine the two rankings: `score = 1/(k + rank_vector) + 1/(k + rank_bm25)`, `k = 60`.
4. **Similarity floor** — drop vector-only candidates below cosine similarity `0.35`.
5. **Version scoring** — boost chunks from the current/latest document version (`×1.15`), penalize superseded versions (`×0.80`), using the `supersedes_document_id` chain in `documents`.
6. **Chunk-type scoring** — classify the query as numeric (mentions revenue/EBITDA/%/currency/etc.) vs narrative, then reweight `financial_table`/`chart_caption`/`body` chunk types accordingly (e.g. numeric query → financial tables `×1.25`, body text `×0.88`).
7. **Cross-encoder rerank** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores `(query, chunk_text)` pairs directly; results below `0.3` are dropped. Disabled via `ATTICUS_LOCAL_RERANKER=false` (used in the Render deploy to save memory — the cross-encoder also lazy-loads `torch`/`sentence-transformers` only on first real query, not at boot).
8. **Deduplication** — keep only the top-ranked chunk per `(document_id, page_number)` so one page can't dominate the result set.

Query preprocessing on top of this: heuristic **query expansion** (e.g. "timeline/chronology" queries get "date/event/report" appended) and **compound-query splitting** at `?` boundaries for multi-part questions.

`RetrievalRequest` supports scoping: `case_id`, `system_only`/`user_only` (demo corpus vs user uploads), `company_filter`, `org_member_ids`.

### 4.5 Generation (`generation/llm_client.py`, `prompts.py`, `answer_builder.py`)

- `ClaudeClient` wraps the Anthropic SDK with both blocking `generate()` and streaming `stream()` methods (the streaming path is what powers `/api/query/stream`'s SSE).
- The **system prompt** (`SYSTEM_PROMPT`, ~125 lines) is the core of answer quality: it enforces (a) grounding every factual claim in retrieved sources cited as `[Source N]`, (b) a two-tier evidence model — Tier 1 = document evidence, Tier 2 = general legal knowledge explicitly labeled `[General Legal Context]` — so the model never silently blends "what the documents say" with "what I know about law in general," (c) Bluebook-style legal citation formatting, (d) a 1–5 confidence score the model must emit in a trailing `<CONFIDENCE>{"score":N,"rationale":"..."}</CONFIDENCE>` block, which the backend parses and then **caps downward** server-side if the evidence was thin (broad question answered from <3 primary docs, or only secondary/general context used), and (e) treating `[FOOTNOTE]`-tagged text as an authoritative override of body text.
- An alternative `CASE_SYSTEM_PROMPT` asks for a structured JSON contract instead of freeform markdown: `{summary, risk_flags[], answer, citations[], follow_up_actions[], confidence}` — used for structured legal-answer card rendering.
- `answer_builder.py` glues retrieval + generation + citation parsing into one `AnswerService.answer()` call producing an `AnswerWithSources`.

### 4.6 Billing / margin guardrail (`billing.py`)

The product's core business constraint: every plan must keep variable cost (AI + storage) under 40% of revenue (`ATTICUS_TARGET_GROSS_MARGIN=0.60`). `PlanLimits` (price, query/upload caps, max file size, max pages) is defined per tier, and `estimate_query_cost_usd()` / `estimate_ingestion_cost_usd()` give a rough per-action cost so usage can be checked against `monthly_price × (1 - margin)`. This isn't exact provider billing reconciliation — it's an early-warning system.

### 4.7 Case analysis (`analysis/case_insights.py`)

Deterministic-first extraction (regex/heuristics) for timelines (date-bearing sentences), entities (repeated capitalized names), and contradiction candidates (conflicting numbers in similar statements), with an LRU cache to avoid recomputing. Optionally LLM-verified. Outputs are persisted to `case_overviews`/`case_timelines`/`case_insights` so the frontend's Overview/Timeline tabs can lazily trigger generation once and then read cached results.

### 4.8 Storage layer (`storage/`)

`snowflake_client.py` (legacy name) is a thread-safe `psycopg2` connection pool (default size 5, `PG_POOL_SIZE` env override) with autocommit and stale-connection eviction, talking to Supabase Postgres. One repository class per concern: `ChunkRepository`, `ImageRepository`, `CasesRepository`, `UserRepository`, `AuditRepository`, `JobsRepository`, `OrgRepository`, persistent chat repos, etc. — each wraps raw SQL behind a typed Python API; no ORM.

**Key tables** (see `storage/schema.sql` + `migrations/*.sql` for the authoritative DDL): `documents`, `chunks` (`embedding vector(1024)`, IVFFlat cosine index), `images` (`description_embedding vector(1024)`), `query_log`, `upload_events`, `cases`, `case_documents`, `case_overviews`, `case_timelines`, `case_insights`, `chats`, `chat_messages`, `users`, `subscriptions`, `organizations`, `organization_members`, `background_jobs`, `demo.sessions`, `demo.usage`, `schema_migrations`.

### 4.9 Memory (`memory/zep_memory.py`, optional)

If `ZEP_ENABLED=true` and `ZEP_API_KEY` set, conversation context and system events (account creation, plan changes, uploads) are pushed to Zep under thread IDs shaped `{uid}:{workspace}:{chat_id}` (workspace = `case:{id}` or `page:{name}`) and `{uid}:events`, and retrieved context is folded into the system prompt. This is additive — the app works without it.

---

## 5. Database Schema Summary

PostgreSQL + `pgvector`. Apply with `python scripts/setup_database.py`, which runs `storage/schema.sql` then every file in `migrations/` in order, recording each in `schema_migrations` so re-runs are idempotent.

Vectors are `vector(1024)` (Voyage `voyage-law-2` dimension) with IVFFlat cosine-similarity indexes. **If you swap in the local 384-dim embedder, the column dimension and index must be migrated and all existing rows re-embedded** — there is no dual-dimension support.

---

## 6. Frontend — Next.js (`frontend/`)

### 6.1 Stack

Next.js 16 (App Router) + React 19, TypeScript, Tailwind CSS v4, Firebase JS SDK v12 (Auth only), Framer Motion (animation), Three.js/`@react-three/fiber`/`@react-three/drei` (the landing page's 3D "Scales of Justice" hero), `react-dropzone` (file upload), `lucide-react` (icons).

### 6.2 Routes

```
/                       landing page (public)
/demo                   demo access-code entry (public)
/demo/chat              demo chat (public, gated by localStorage token)
/demo/admin             demo cost dashboard (admin key)
/about /privacy /terms  static pages (public)

/(app)/chat             main authenticated chat — requires Firebase auth
/(app)/cases            case list
/(app)/cases/[caseId]   case detail: documents + overview + timeline tabs
/(app)/epstein          pinned public-record demo case, inside the authed app
/(app)/org              organization/team management
/(app)/data             usage analytics dashboard
/(app)/profile          account settings, deletion
/(app)/discussion       community discussion feature
```

### 6.3 Talking to the backend — the authenticated client (`frontend/src/lib/api.ts`)

- **Base URL**: `process.env.NEXT_PUBLIC_API_URL`, with a safety net — in a production build, if that var still points at `localhost`, it's ignored in favor of a hardcoded Render URL fallback (prevents accidentally shipping a build wired to a dev backend).
- **Every request**: fetches a fresh Firebase ID token via `auth.currentUser.getIdToken(false)` (Firebase caches/refreshes internally — tokens last 1 hour, SDK refreshes silently under the hood), attaches `Authorization: Bearer <token>` and `Content-Type: application/json`, and sends `credentials: "include"`.
- **401 handling**: on a 401 response, force-refreshes the token (`getIdToken(true)`) and retries the request exactly once, then gives up — guards against a request firing right as a token expires.
- **Streaming**: `runQueryStream()` opens `POST /api/query/stream`, reads the response body as a stream of `data: {...}` lines (SSE), and invokes an `onToken` callback per incremental token, finally resolving with the full `{text, sources, source_details, images, confidence}` payload once a `{done: true}` line arrives.

### 6.4 Talking to the backend — the demo client (`frontend/src/lib/demo-api.ts`)

- Same base-URL logic, but auth is a JWT stored in `localStorage`/`sessionStorage` (key `demo_token`), not Firebase — attached the same way as a bearer token, but never refreshed (it doesn't expire; re-auth means re-entering the access code).
- **Cold-start handling**: request timeout was raised from 20s to **120s** (10 minutes for uploads) specifically because Render's free tier can take >20s to wake a sleeping instance — this was a recent fix (commit `35c0ea0`, "Allow Render cold starts in demo client"). Implemented via `AbortController` + a wrapping `withTimeout()` helper, not via `fetch`'s native timeout (there isn't one).

### 6.5 Firebase Auth wiring (`frontend/src/lib/firebase.ts`, `frontend/src/contexts/AuthContext.tsx`)

- The Firebase app initializes only if `NEXT_PUBLIC_FIREBASE_API_KEY` is present (lets the app boot in environments without Firebase configured, e.g. demo-only deployments).
- `browserLocalPersistence` is set explicitly so sessions survive refreshes/restarts.
- `AuthContext` subscribes to `onAuthStateChanged`; on login it calls `POST /api/auth/register` (idempotent — syncs the Firebase user into Postgres + Zep) and `GET /api/subscription/status` (trial/plan info), and exposes `{user, idToken, plan, subscriptionStatus, isTrialExpired, daysRemaining}` to the whole app via `useAuth()`. If `is_new` comes back true from registration, it force-refreshes the ID token once so newly-set custom claims (plan) are reflected immediately.

### 6.6 Authenticated chat page flow (`frontend/src/app/(app)/chat/page.tsx`)

1. Redirects unauthenticated users back to `/`.
2. Loads chat list (`GET /api/chats`, filtered by `page`), the active chat's messages, and the company-filter dropdown (`GET /api/companies`).
3. Tracks a "workspace" selection (a specific case, "all uploaded documents", or the pinned public demo case) persisted to `localStorage` and broadcast across tabs via a custom `atticus-workspace-change` event.
4. On submit: appends the user message locally, lazily creates a chat if none is active (`POST /api/chats`), then calls `runQueryStream()`; tokens stream into the last assistant message in place; once `done`, sources/images/confidence are attached to that message and it's persisted server-side with `POST /api/chats/{id}/messages` (fire-and-forget — UI doesn't block on this).
5. Renders Markdown answers, a confidence gauge, extracted `⚠️ HIGH RISK` flags, clickable source citations (opens the underlying chunk text + page), and any extracted images.

### 6.7 Demo flow (`frontend/src/app/demo/page.tsx`, `frontend/src/app/demo/chat/page.tsx`)

Entry page collects name/email/phone + access code → `POST /api/demo/auth` → stores the returned JWT → redirects to `/demo/chat`, which loads session info (`GET /api/demo/me`), the demo case list, and supports three tabs: **Chat** (same streaming-less `POST /api/demo/query`, no SSE), **Timeline**, and **Overview** (both lazily trigger generation on first visit and poll/display a "pending" state). A lightweight achievement/gamification layer tracks milestones client-side (first query, fifth query, visited all tabs, etc.) purely for engagement — no backend involvement.

---

## 7. End-to-End Request Walkthrough

**A signed-in user asks a question:**

1. Browser: `useAuth()` confirms a logged-in Firebase user; chat page already has `activeChatId` and `workspace` (e.g. a specific case).
2. `runQueryStream({query, chat_id, case_id, company_filter})` fetches a fresh Firebase ID token, opens `POST /api/query/stream` with `Authorization: Bearer <token>`.
3. Backend: `require_active_subscription` dependency verifies the Firebase token (Admin SDK) and checks the user's plan hasn't expired (Postgres `users` row).
4. Ownership check: if `case_id` present, confirm the case belongs to this `uid`.
5. `Embedder.embed_query()` — calls Voyage AI (or local model) to embed the question.
6. `HybridSearchService` runs vector search + BM25 → RRF fusion → version/chunk-type scoring → cross-encoder rerank → dedup, scoped to the case/user/system documents as appropriate.
7. `build_user_prompt()` assembles retrieved chunks (with citation labels, page numbers, section headers) into the Claude prompt alongside `SYSTEM_PROMPT`.
8. `ClaudeClient.stream()` streams tokens back from Anthropic; the FastAPI route forwards each token as an SSE `{token}` event.
9. On completion, the backend parses the `<CONFIDENCE>` block, applies the evidence-based confidence cap, collects any associated images, and emits a final `{done: true, text, sources, source_details, images, confidence}` SSE event; it also writes an audit row to `query_log` with estimated cost.
10. Frontend renders tokens incrementally as they arrive, then attaches sources/confidence/images to the finished message and persists it via `POST /api/chats/{id}/messages`.

**A demo visitor uploads a folder of documents:**

1. `/demo` → access code redeemed → JWT stored client-side.
2. `/demo/chat` → upload modal → `POST /api/demo/cases/upload` (multipart, ≤8 files) with `Authorization: Bearer <demo_jwt>`.
3. Backend verifies the demo JWT, enforces the one-case-per-session limit, and synchronously (no job queue, `fast_mode=True`) runs the full ingestion pipeline (§4.2) for each file.
4. Response includes per-file `{document_id, page_count, chunks_inserted}` plus any skipped files (e.g. unsupported type, over size limit).
5. Subsequent `POST /api/demo/query` calls with that `case_id` retrieve only from this case's chunks.

---

## 8. Replicating This From Scratch

If rebuilding from zero, the dependency order is:

1. **Provision infra**: a Postgres database with `pgvector` enabled (Supabase is easiest), a Firebase project (Auth → Email/Password + Google enabled), an Anthropic API key, a Voyage AI API key (optional — local embeddings work without it), and optionally Redis and Zep.
2. **Core library first** (`src/insightlens`): config → storage schema/migrations → embedder → ingestion pipeline → retrieval pipeline → generation/prompts → billing. Get this working end-to-end via the `scripts/ingest_documents.py` and a small CLI/test harness before touching HTTP at all — this is the part worth getting right, and it has no web framework dependency.
3. **FastAPI layer** (`backend/`): wrap the library in routes, wire Firebase Admin SDK token verification, add ownership checks per resource, add rate limiting.
4. **Demo layer** (`backend/demo_router.py`): a parallel, simpler auth path (access code → JWT) scoped to a shared read-only corpus.
5. **Frontend** (`frontend/`): Next.js app, Firebase client SDK for auth, a thin typed API client per backend (one for authed routes, one for demo routes) that always re-fetches the Firebase token per request rather than caching it locally.
6. **Deploy**: containerize the backend (Dockerfile already present), deploy to Render or similar with health checks; deploy the frontend to Vercel with `NEXT_PUBLIC_*` env vars pointing at the backend URL and Firebase config.

The single most important design decision to preserve: **keep all RAG/business logic in a framework-agnostic library, and keep the HTTP layer as a thin adapter that only does auth, validation, and ownership checks.** That's what made this project's frontend migration (Streamlit → Next.js/FastAPI) possible without touching ingestion, retrieval, or generation code at all.

---

## 9. Environment Variables Reference

**Backend (`src/insightlens/config.py`, `backend/`):**

```bash
# Required
DATABASE_URL=postgresql://user:password@host:5432/dbname
ANTHROPIC_API_KEY=sk-ant-...

# Generation / retrieval tuning
GENERATION_MODEL=claude-sonnet-4-6
EMBEDDING_MODEL=voyage-law-2          # or all-MiniLM-L6-v2 if no Voyage key
RETRIEVAL_TOP_K=8
CHUNK_SIZE_TOKENS=400
CHUNK_OVERLAP_TOKENS=50
VOYAGE_API_KEY=                        # optional; unlocks 1024-dim Voyage embeddings
ATTICUS_LOCAL_RERANKER=true            # set false to disable cross-encoder (saves memory)

# Firebase Admin (pick one path)
FIREBASE_SERVICE_ACCOUNT_JSON=         # raw service-account JSON, for PaaS envs
FIREBASE_SERVICE_ACCOUNT_PATH=         # file path, for Docker/local
FIREBASE_API_KEY=

# Demo mode
DEMO_ACCESS_CODE_USER1=
DEMO_ACCESS_CODE_USER2=
DEMO_ACCESS_CODE_USER3=

# Infra
REDIS_URL=redis://localhost:6379/0     # optional, falls back to in-memory rate limiting
PG_POOL_SIZE=5
ADMIN_API_KEY=                         # protects /api/admin/* and demo cost dashboard
CORS_ORIGINS=https://your-frontend.vercel.app
CORS_ORIGIN_REGEX=^https://your-frontend(?:-[a-z0-9-]+)*\.vercel\.app$

# Optional long-term memory
ZEP_API_KEY=
ZEP_ENABLED=false

# Margin guardrails (billing.py)
ATTICUS_TARGET_GROSS_MARGIN=0.60
ATTICUS_STARTER_PRICE_USD=29
ATTICUS_MONTHLY_QUERY_LIMIT=300
ATTICUS_MONTHLY_UPLOAD_LIMIT=20
ATTICUS_MAX_UPLOAD_MB=50
ATTICUS_MAX_PAGES_PER_PDF=500
```

**Frontend (`frontend/.env.local`):**

```bash
NEXT_PUBLIC_FIREBASE_API_KEY=
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=
NEXT_PUBLIC_FIREBASE_PROJECT_ID=
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=
NEXT_PUBLIC_FIREBASE_APP_ID=
NEXT_PUBLIC_API_URL=http://localhost:8000   # backend base URL
```

---

## 10. Known Inconsistencies in This Repo (worth knowing if you dig into the code)

- `src/insightlens/storage/snowflake_client.py` is a **PostgreSQL** client; the name is a holdover from the original Snowflake-based prototype.
- `docs/architecture.md` and parts of `PROJECT.md`/`README.md` describe an **earlier Streamlit + Snowflake/local-embeddings** version of this product. The real, current stack is Next.js + FastAPI + Postgres/pgvector + Voyage AI, as documented above. `MIGRATION_CHECKLIST.md` documents the cutover.
- The Python package is still named `insightlens` (the product's original name) even though the product is branded **Atticus**.
- Embedding dimension is `1024` (Voyage `voyage-law-2`) in the current schema, not the `384` (local MiniLM) quoted in the older docs — see `migrations/007_vector_dim_1024.sql`.
