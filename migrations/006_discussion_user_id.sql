-- Add user_id column to discussion_posts for ownership-based delete guards.
ALTER TABLE discussion_posts ADD COLUMN IF NOT EXISTS user_id TEXT;
CREATE INDEX IF NOT EXISTS discussion_posts_user_id_idx ON discussion_posts (user_id);
