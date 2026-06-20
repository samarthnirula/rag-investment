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
