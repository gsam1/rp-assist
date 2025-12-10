-- RPG Assistant Database Schema for PGVector
-- This schema defines the tables and indexes for the vector database

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Document chunks table
-- Stores document chunks with their embeddings for similarity search
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page INTEGER,
    embedding vector(1536),  -- OpenAI embedding dimension
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
-- Using IVFFlat for faster approximate nearest neighbor search
-- Adjust lists parameter based on dataset size (rule of thumb: sqrt(total_rows))
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for document_id for fast deletion
CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx
ON document_chunks (document_id);

-- Create index for metadata queries
CREATE INDEX IF NOT EXISTS document_chunks_metadata_idx
ON document_chunks USING gin (metadata);

-- Optional: Documents metadata table
-- For storing document-level information
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    type TEXT NOT NULL,  -- 'pdf', 'txt', 'note'
    path TEXT NOT NULL,
    pages INTEGER,
    chunks INTEGER DEFAULT 0,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP,
    file_hash TEXT,
    metadata JSONB
);

-- Create index for filename searches
CREATE INDEX IF NOT EXISTS documents_filename_idx
ON documents (filename);

-- Create index for type filtering
CREATE INDEX IF NOT EXISTS documents_type_idx
ON documents (type);

-- Comments for documentation
COMMENT ON TABLE document_chunks IS 'Stores text chunks from documents with vector embeddings';
COMMENT ON COLUMN document_chunks.embedding IS 'Vector embedding for similarity search';
COMMENT ON COLUMN document_chunks.metadata IS 'Additional metadata (filename, type, etc.)';

COMMENT ON TABLE documents IS 'Metadata for uploaded documents';
COMMENT ON COLUMN documents.file_hash IS 'SHA256 hash for change detection';
