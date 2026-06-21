-- demo.sessions gains contact info columns captured at access-code redemption
-- (backend/demo_router.py auth_demo writes these; see commit ef644b7).
ALTER TABLE demo.sessions ADD COLUMN IF NOT EXISTS first_name TEXT;
ALTER TABLE demo.sessions ADD COLUMN IF NOT EXISTS last_name TEXT;
ALTER TABLE demo.sessions ADD COLUMN IF NOT EXISTS email TEXT;
ALTER TABLE demo.sessions ADD COLUMN IF NOT EXISTS phone TEXT;
ALTER TABLE demo.sessions ADD COLUMN IF NOT EXISTS info_submitted_at TIMESTAMPTZ;
