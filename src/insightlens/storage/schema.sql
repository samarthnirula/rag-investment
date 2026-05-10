CREATE DATABASE IF NOT EXISTS INSIGHTLENS;
USE DATABASE INSIGHTLENS;
USE SCHEMA PUBLIC;

CREATE TABLE IF NOT EXISTS DOCUMENTS (
    document_id     VARCHAR PRIMARY KEY,
    file_name       VARCHAR NOT NULL,
    company         VARCHAR,
    document_type   VARCHAR,
    version_label   VARCHAR,
    version_date    DATE,
    page_count      INTEGER,
    ingested_at     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS CHUNKS (
    chunk_id        VARCHAR PRIMARY KEY,
    document_id     VARCHAR NOT NULL,
    page_number     INTEGER NOT NULL,
    chunk_index     INTEGER NOT NULL,
    chunk_text      VARCHAR NOT NULL,
    token_count     INTEGER,
    embedding       VECTOR(FLOAT, 384),
    FOREIGN KEY (document_id) REFERENCES DOCUMENTS(document_id)
)
