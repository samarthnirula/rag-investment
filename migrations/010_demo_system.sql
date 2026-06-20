CREATE SCHEMA IF NOT EXISTS demo;

CREATE TABLE IF NOT EXISTS demo.sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_slug TEXT NOT NULL UNIQUE,
  access_code_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  last_active TIMESTAMPTZ,
  query_count INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS demo.usage (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_slug TEXT NOT NULL,
  timestamp TIMESTAMPTZ DEFAULT NOW(),
  query_type TEXT NOT NULL,
  model TEXT NOT NULL,
  input_tokens INT DEFAULT 0,
  output_tokens INT DEFAULT 0,
  cost_usd NUMERIC(10,6) DEFAULT 0,
  question TEXT
);

CREATE INDEX IF NOT EXISTS demo_usage_user_idx ON demo.usage(user_slug);
CREATE INDEX IF NOT EXISTS demo_usage_ts_idx ON demo.usage(timestamp);
