"""Vector store abstraction and implementations"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import psycopg2
from psycopg2.extras import execute_values
import faiss
import numpy as np
import pickle
import os
import json
from pathlib import Path

from app.models.document import DocumentChunk
from app.utils.logger import get_logger
from app.utils.config_loader import get_config

logger = get_logger()


class VectorStore(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the store"""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int) -> List[DocumentChunk]:
        """Search for similar chunks"""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the store"""
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Get total number of chunks"""
        pass


class PGVectorStore(VectorStore):
    """PostgreSQL with pgvector extension"""

    def __init__(self):
        config = get_config()
        self.conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            database=os.getenv('POSTGRES_DB', 'rpg_assistant'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', ''),
        )
        self._ensure_extension()
        self._ensure_table()

    def _ensure_extension(self):
        """Ensure pgvector extension is enabled"""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit()

    def _ensure_table(self):
        """Ensure document_chunks table exists"""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page INTEGER,
                    embedding vector(1536),
                    metadata JSONB
                )
            """)
            # Create index for vector similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            # Create index for document_id
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx
                ON document_chunks (document_id)
            """)
            self.conn.commit()

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to PostgreSQL"""
        if not chunks:
            return

        with self.conn.cursor() as cur:
            data = [
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.content,
                    chunk.chunk_index,
                    chunk.page,
                    chunk.embedding,
                    json.dumps(chunk.metadata)
                )
                for chunk in chunks
            ]

            execute_values(
                cur,
                """
                INSERT INTO document_chunks
                (id, document_id, content, chunk_index, page, embedding, metadata)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                data
            )
            self.conn.commit()

        logger.info(f"Added {len(chunks)} chunks to PGVector")

    def search(self, query_embedding: List[float], top_k: int) -> List[DocumentChunk]:
        """Search for similar chunks using cosine similarity"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, document_id, content, chunk_index, page, metadata,
                       1 - (embedding <=> %s::vector) as similarity
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k)
            )

            results = []
            for row in cur.fetchall():
                chunk = DocumentChunk(
                    id=row[0],
                    document_id=row[1],
                    content=row[2],
                    chunk_index=row[3],
                    page=row[4],
                    metadata=row[5] if row[5] else {}
                )
                results.append(chunk)

            return results

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document"""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
            self.conn.commit()
        logger.info(f"Deleted chunks for document {document_id}")

    def clear(self) -> None:
        """Clear all chunks"""
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE document_chunks")
            self.conn.commit()
        logger.info("Cleared all chunks from PGVector")

    def get_count(self) -> int:
        """Get total number of chunks"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            return cur.fetchone()[0]


class FAISSVectorStore(VectorStore):
    """FAISS-based local vector store"""

    def __init__(self):
        self.index_path = "data/vectordb/faiss.index"
        self.metadata_path = "data/vectordb/metadata.pkl"
        self.dimension = 1536  # OpenAI embedding dimension

        Path("data/vectordb").mkdir(parents=True, exist_ok=True)

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.chunks = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.chunks: List[DocumentChunk] = []

    def _save(self):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.chunks, f)

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to FAISS"""
        if not chunks:
            return

        embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self._save()

        logger.info(f"Added {len(chunks)} chunks to FAISS")

    def search(self, query_embedding: List[float], top_k: int) -> List[DocumentChunk]:
        """Search for similar chunks"""
        if self.index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document (rebuild index)"""
        # FAISS doesn't support deletion, so we rebuild
        self.chunks = [c for c in self.chunks if c.document_id != document_id]

        # Rebuild index
        self.index = faiss.IndexFlatL2(self.dimension)
        if self.chunks:
            embeddings = np.array([c.embedding for c in self.chunks], dtype=np.float32)
            self.index.add(embeddings)

        self._save()
        logger.info(f"Deleted chunks for document {document_id} (rebuilt FAISS index)")

    def clear(self) -> None:
        """Clear all chunks"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self._save()
        logger.info("Cleared all chunks from FAISS")

    def get_count(self) -> int:
        """Get total number of chunks"""
        return self.index.ntotal


def create_vector_store() -> VectorStore:
    """Factory function to create the configured vector store"""
    config = get_config()
    store_type = config.get('vectorstore.type', 'faiss')

    if store_type == 'pgvector':
        logger.info("Initializing PGVector store")
        return PGVectorStore()
    elif store_type == 'faiss':
        logger.info("Initializing FAISS store")
        return FAISSVectorStore()
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
