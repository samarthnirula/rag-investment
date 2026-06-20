CREATE TABLE IF NOT EXISTS subscriptions (
    user_id                TEXT PRIMARY KEY,
    plan_name              TEXT NOT NULL DEFAULT 'Starter',
    stripe_customer_id     TEXT,
    stripe_subscription_id TEXT,
    status                 TEXT NOT NULL DEFAULT 'trialing',
    current_period_start   TIMESTAMPTZ,
    current_period_end     TIMESTAMPTZ,
    updated_at             TIMESTAMPTZ DEFAULT NOW()
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
