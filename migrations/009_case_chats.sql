-- Case-isolated chat tabs, auto-generated overview and timeline.

-- Link chats to cases and add a type discriminator.
-- NOTE: cases.case_id is TEXT (see schema.sql), not UUID -- this column must
-- match or the REFERENCES constraint below fails to even create.
ALTER TABLE chats ADD COLUMN IF NOT EXISTS case_id TEXT REFERENCES cases(case_id) ON DELETE CASCADE;
ALTER TABLE chats ADD COLUMN IF NOT EXISTS chat_type TEXT NOT NULL DEFAULT 'chat';

-- Track whether AI summaries have been generated for each case.
ALTER TABLE cases ADD COLUMN IF NOT EXISTS overview_generated BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE cases ADD COLUMN IF NOT EXISTS timeline_generated BOOLEAN NOT NULL DEFAULT FALSE;

-- One-row-per-case AI-generated timeline (list of events as JSONB).
CREATE TABLE IF NOT EXISTS case_timelines (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id      TEXT         NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
    user_id      TEXT         NOT NULL,
    events       JSONB        NOT NULL DEFAULT '[]',
    generated_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (case_id)
);

-- One-row-per-case AI-generated overview (structured summary).
CREATE TABLE IF NOT EXISTS case_overviews (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id      TEXT         NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
    user_id      TEXT         NOT NULL,
    summary      TEXT         NOT NULL DEFAULT '',
    parties      JSONB        NOT NULL DEFAULT '[]',
    key_issues   JSONB        NOT NULL DEFAULT '[]',
    jurisdiction TEXT,
    matter_type  TEXT,
    generated_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (case_id)
);

-- Performance indexes.
CREATE INDEX IF NOT EXISTS idx_case_documents_case_id   ON case_documents (case_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id       ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chats_case_id            ON chats (case_id) WHERE case_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_case_timelines_case_id   ON case_timelines (case_id);
CREATE INDEX IF NOT EXISTS idx_case_overviews_case_id   ON case_overviews (case_id);
