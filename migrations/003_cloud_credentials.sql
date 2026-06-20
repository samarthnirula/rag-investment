-- Store per-user OAuth refresh tokens for cloud storage providers.
-- Access tokens are short-lived (1h) and refreshed on demand.

CREATE TABLE IF NOT EXISTS cloud_credentials (
    id              BIGSERIAL PRIMARY KEY,
    user_id         TEXT NOT NULL,
    provider        TEXT NOT NULL,          -- 'google_drive' | 'dropbox' | 'onedrive'
    refresh_token   TEXT NOT NULL,
    access_token    TEXT,
    token_expires_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (user_id, provider)
);

CREATE INDEX IF NOT EXISTS cloud_credentials_user_idx
    ON cloud_credentials (user_id);

-- SECURITY: refresh_token and access_token are encrypted at the application
-- layer before insert and decrypted on read -- see
-- src/insightlens/storage/token_crypto.py (Fernet/AES-128-CBC+HMAC) and its
-- use in cloud_credentials_repository.py's upsert()/get(). The column type
-- stays TEXT because it stores ciphertext, not plaintext. Encryption key
-- comes from the TOKEN_ENCRYPTION_KEY env var -- a leaked DB snapshot alone
-- does NOT grant access to a user's connected cloud storage without that key.
COMMENT ON TABLE cloud_credentials IS
    'OAuth refresh/access tokens for cloud storage integrations. '
    'Encrypted at the application layer (Fernet) before storage -- see '
    'storage/token_crypto.py. Access tokens are refreshed on demand and not '
    'persisted long-term.';