-- InsightLens — PostgreSQL + pgvector schema
-- Run once to initialise the database.
-- Requires: CREATE EXTENSION IF NOT EXISTS vector;  (done first below)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    applied_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    document_id            TEXT PRIMARY KEY,
    file_name              TEXT NOT NULL,
    company                TEXT,
    document_type          TEXT,
    version_label          TEXT,
    version_date           DATE,
    page_count             INTEGER,
    supersedes_document_id TEXT,
    user_id                TEXT,
    ingested_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id           TEXT PRIMARY KEY,
    document_id        TEXT NOT NULL REFERENCES documents(document_id),
    page_number        INTEGER NOT NULL,
    chunk_index        INTEGER NOT NULL,
    chunk_text         TEXT NOT NULL,
    token_count        INTEGER,
    -- 1024 dims = voyage-law-2 (see migrations/007_vector_dim_1024.sql). Was 384
    -- (all-MiniLM-L6-v2) before; kept in sync here so fresh installs match
    -- post-migration installs.
    embedding          vector(1024),
    section_header     TEXT,
    chunk_type         TEXT DEFAULT 'body',
    structured_content TEXT
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS images (
    image_id              TEXT PRIMARY KEY,
    document_id           TEXT NOT NULL REFERENCES documents(document_id),
    page_number           INTEGER NOT NULL,
    image_index           INTEGER NOT NULL,
    file_path             TEXT NOT NULL,
    media_type            TEXT,
    width                 INTEGER,
    height                INTEGER,
    ai_description        TEXT,
    description_embedding vector(1024),
    ingested_at           TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE images ADD COLUMN IF NOT EXISTS description_embedding vector(1024);

CREATE INDEX IF NOT EXISTS images_desc_embedding_idx
    ON images USING ivfflat (description_embedding vector_cosine_ops)
    WITH (lists = 10);

CREATE TABLE IF NOT EXISTS query_log (
    log_id           TEXT PRIMARY KEY,
    user_id          TEXT,
    page             TEXT,
    query_text       TEXT NOT NULL,
    chunks_retrieved INTEGER,
    model_used       TEXT,
    response_length  INTEGER,
    estimated_cost_usd NUMERIC(12, 6) DEFAULT 0,
    logged_at        TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS estimated_cost_usd NUMERIC(12, 6) DEFAULT 0;

CREATE TABLE IF NOT EXISTS upload_events (
    upload_id          TEXT PRIMARY KEY,
    user_id            TEXT NOT NULL,
    document_id        TEXT,
    file_name          TEXT NOT NULL,
    file_size_bytes    BIGINT DEFAULT 0,
    page_count         INTEGER DEFAULT 0,
    chunks_inserted    INTEGER DEFAULT 0,
    estimated_cost_usd NUMERIC(12, 6) DEFAULT 0,
    uploaded_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS upload_events_user_uploaded_idx
    ON upload_events (user_id, uploaded_at DESC);

-- Moved up from later in the file: case_insights and generated_artifacts
-- below reference cases(case_id), so cases must be created first.
CREATE TABLE IF NOT EXISTS cases (
    case_id     TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    case_name   TEXT NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS case_insights (
    insight_id         TEXT PRIMARY KEY,
    case_id            TEXT NOT NULL REFERENCES cases(case_id),
    insight_type       TEXT NOT NULL,
    title              TEXT NOT NULL,
    body               TEXT,
    severity           TEXT,
    document_id        TEXT,
    page_number        INTEGER,
    metadata_json      TEXT,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS case_insights_case_type_idx
    ON case_insights (case_id, insight_type, created_at DESC);

CREATE TABLE IF NOT EXISTS generated_artifacts (
    artifact_id        TEXT PRIMARY KEY,
    user_id            TEXT NOT NULL,
    case_id            TEXT,
    artifact_type      TEXT NOT NULL,
    title              TEXT NOT NULL,
    content            TEXT NOT NULL,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS generated_artifacts_user_created_idx
    ON generated_artifacts (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS subscriptions (
    user_id              TEXT PRIMARY KEY,
    plan_name            TEXT NOT NULL DEFAULT 'Starter',
    stripe_customer_id   TEXT,
    stripe_subscription_id TEXT,
    status               TEXT NOT NULL DEFAULT 'trialing',
    current_period_start TIMESTAMPTZ,
    current_period_end   TIMESTAMPTZ,
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS organizations (
    org_id      TEXT PRIMARY KEY,
    org_name    TEXT NOT NULL,
    owner_id    TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS organization_members (
    org_id      TEXT NOT NULL REFERENCES organizations(org_id),
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'member',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (org_id, user_id)
);

CREATE TABLE IF NOT EXISTS background_jobs (
    job_id       TEXT PRIMARY KEY,
    job_type     TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'queued',
    user_id      TEXT,
    case_id      TEXT,
    payload_json TEXT,
    error        TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS background_jobs_status_created_idx
    ON background_jobs (status, created_at);

CREATE TABLE IF NOT EXISTS case_documents (
    case_id     TEXT NOT NULL,
    document_id TEXT NOT NULL,
    added_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (case_id, document_id)
);

CREATE TABLE IF NOT EXISTS chats (
    chat_id    TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    page       TEXT NOT NULL,
    chat_name  TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id  TEXT PRIMARY KEY,
    chat_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    chunks_json TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS discussion_posts (
    post_id   TEXT PRIMARY KEY,
    author    TEXT NOT NULL,
    post_type TEXT NOT NULL,
    content   TEXT NOT NULL,
    posted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS consent_log (
    consent_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    consent_type TEXT NOT NULL,
    accepted_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (user_id, consent_type)
);

CREATE TABLE IF NOT EXISTS access_codes (
    code        TEXT PRIMARY KEY,
    created_by  TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    max_uses    INTEGER DEFAULT 1,
    uses_count  INTEGER DEFAULT 0,
    used_by     TEXT,
    used_at     TIMESTAMPTZ,
    is_active   BOOLEAN DEFAULT TRUE,
    note        TEXT
);
