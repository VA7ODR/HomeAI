-- Enable pgvector extension for cosine similarity searches.
CREATE EXTENSION IF NOT EXISTS vector;

-- -----------------------------------------------------------------------------
-- doc_chunks: individual segments of repository files prepared for retrieval.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS doc_chunks (
    id            BIGSERIAL PRIMARY KEY,
    source_kind   TEXT        NOT NULL,
    source_path   TEXT        NOT NULL,
    file_name     TEXT        NOT NULL,
    content_hash  TEXT        NOT NULL,
    chunk_index   INTEGER     NOT NULL,
    content       TEXT        NOT NULL,
    embedding     VECTOR(1024),
    size_bytes    BIGINT      NOT NULL,
    mtime         TIMESTAMPTZ NOT NULL,
    mime_type     TEXT,
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (chunk_index >= 0)
);

COMMENT ON COLUMN doc_chunks.source_kind IS 'Category of document (e.g. file, log, note).';
COMMENT ON COLUMN doc_chunks.source_path IS 'Absolute allowlisted path to the source file.';
COMMENT ON COLUMN doc_chunks.file_name IS 'Basename of the source file for quick display.';
COMMENT ON COLUMN doc_chunks.content_hash IS 'SHA-256 hash of the chunk contents for idempotent ingestion.';
COMMENT ON COLUMN doc_chunks.embedding IS 'Vector embedding (defaults to 1024 dims; migrate before changing).';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'doc_chunks_unique_chunk'
    ) THEN
        ALTER TABLE doc_chunks
            ADD CONSTRAINT doc_chunks_unique_chunk
            UNIQUE (source_path, content_hash, chunk_index);
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS doc_chunks_updated_at_idx
    ON doc_chunks (updated_at DESC);

CREATE INDEX IF NOT EXISTS doc_chunks_mime_type_idx
    ON doc_chunks (mime_type);

-- HNSW offers better recall on PostgreSQL 16+.  Fallback to IVFFlat where HNSW
-- is unavailable by adjusting the method below.
CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx
    ON doc_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- -----------------------------------------------------------------------------
-- messages: raw chat history tied to a logical thread.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    message_id  TEXT        NOT NULL,
    thread_id   TEXT        NOT NULL,
    role        TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    embedding   VECTOR(1024),
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON COLUMN messages.message_id IS 'Stable identifier for the message turn (e.g. UUID).';
COMMENT ON COLUMN messages.thread_id IS 'Conversation / session identifier.';
COMMENT ON COLUMN messages.role IS 'Conversation role (user, assistant, tool).';
COMMENT ON COLUMN messages.embedding IS 'Vector embedding (cosine similarity).';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'messages_message_id_unique'
    ) THEN
        ALTER TABLE messages
            ADD CONSTRAINT messages_message_id_unique UNIQUE (message_id);
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS messages_thread_updated_idx
    ON messages (thread_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS messages_embedding_idx
    ON messages
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
