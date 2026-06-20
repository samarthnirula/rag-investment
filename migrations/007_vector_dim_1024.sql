-- Migrate embedding column from 384-dim (all-MiniLM-L6-v2) to 1024-dim (voyage-law-2).
-- WARNING: after running this migration you MUST re-ingest all documents because
-- the old 384-dim embeddings are no longer valid for the new 1024-dim column.
-- Run: python scripts/ingest_documents.py --reset  (or your ingest script with --reset)

ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024);
ALTER TABLE images ALTER COLUMN embedding TYPE vector(1024);
