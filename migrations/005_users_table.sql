-- 005: User registry table
-- Stores Firebase user metadata synced on first sign-in, plus the user's plan.
-- plan mirrors the Firebase custom claim so a single DB query can return it.

CREATE TABLE IF NOT EXISTS users (
    uid             TEXT PRIMARY KEY,
    email           TEXT,
    display_name    TEXT,
    plan            TEXT NOT NULL DEFAULT 'trial',
    plan_updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS users_email_idx ON users (email);
