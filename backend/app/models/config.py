"""Configuration models"""

from pydantic import BaseModel
from typing import Optional, List


class AppConfig(BaseModel):
    """Application configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    max_file_size_mb: int = 100
    log_level: str = "INFO"


class LLMConfig(BaseModel):
    """LLM configuration"""
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: int = 4096


class ProviderConfig(BaseModel):
    """Individual provider configuration"""
    model: str
    context_window: int
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    endpoint: Optional[str] = None  # For local providers


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    type: str = "pgvector"  # faiss or pgvector
    chunk_size: int = 1000
    chunk_overlap: float = 0.2
    top_k: int = 5


class EmbeddingConfig(BaseModel):
    """Embedding configuration"""
    model: str = "text-embedding-3-large"
    provider: str = "openai"


class ImageConfig(BaseModel):
    """Image generation configuration"""
    size: str = "1024x1024"
    format: str = "jpeg"
    save_directory: str = "data/images"
    provider: str = "openai"
    model: str = "dall-e-3"


class DocumentConfig(BaseModel):
    """Document configuration"""
    upload_directory: str = "data/uploads"
    notes_directory: str = "data/notes"
    metadata_file: str = "data/metadata/index_metadata.json"


class ConversationConfig(BaseModel):
    """Conversation configuration"""
    save_directory: str = "data/conversations"
    auto_save_on_exit: bool = True
