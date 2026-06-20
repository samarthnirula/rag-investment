-- Trial system: per-user trial window and subscription state.
ALTER TABLE users ADD COLUMN IF NOT EXISTS trial_expires_at TIMESTAMPTZ;
ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_active BOOLEAN NOT NULL DEFAULT FALSE;

-- Back-fill: existing users get a 4-day window from their account creation.
UPDATE users
   SET trial_expires_at = created_at + INTERVAL '4 days'
 WHERE trial_expires_at IS NULL;
