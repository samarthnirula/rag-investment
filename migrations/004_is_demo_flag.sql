-- 004: Mark demo/system corpus documents and cases as read-only
-- Adds is_demo flag to documents and cases tables.
-- All documents with user_id IS NULL are system-ingested demo documents
-- (Epstein corpus, public research files, etc.) and must never be user-editable.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_demo BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE cases     ADD COLUMN IF NOT EXISTS is_demo BOOLEAN NOT NULL DEFAULT FALSE;

-- Stamp every document that was ingested without a user_id as a demo document.
-- These are the Epstein/public corpus files loaded by the admin ingestion scripts.
UPDATE documents SET is_demo = TRUE WHERE user_id IS NULL AND is_demo = FALSE;
