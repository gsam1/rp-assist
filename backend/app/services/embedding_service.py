"""Embedding service for generating text embeddings"""

from typing import List
import openai
import os

from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger()


class EmbeddingService:
    """Service for generating text embeddings"""

    def __init__(self):
        config = get_config()
        self.provider = config.get('embedding.provider', 'openai')
        self.model = config.get('embedding.model', 'text-embedding-3-large')

        if self.provider == 'openai':
            openai.api_key = os.getenv('OPENAI_API_KEY')
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        if not texts:
            return []

        if self.provider == 'openai':
            return self._embed_openai(texts)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=texts
            )

            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings using OpenAI")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


# Global instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
