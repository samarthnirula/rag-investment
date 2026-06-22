-- Migrate embeddings from 384-dim all-MiniLM-L6-v2 to 1024-dim voyage-law-2.
--
-- Existing vectors cannot be converted between model dimensions. Preserve the
-- source rows and text, clear only the obsolete vectors, then backfill them
-- with scripts/backfill_embeddings.py after this migration completes.

DROP INDEX IF EXISTS chunks_embedding_idx;
DROP INDEX IF EXISTS images_desc_embedding_idx;

ALTER TABLE chunks
    ALTER COLUMN embedding TYPE vector(1024)
    USING NULL::vector(1024);

ALTER TABLE images
    ALTER COLUMN description_embedding TYPE vector(1024)
    USING NULL::vector(1024);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS images_desc_embedding_idx
    ON images USING ivfflat (description_embedding vector_cosine_ops)
    WITH (lists = 10);
