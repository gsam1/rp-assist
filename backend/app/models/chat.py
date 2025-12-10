"""Chat and messaging models"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import uuid4


class Citation(BaseModel):
    """Citation reference for RAG sources"""
    source: str  # Filename
    document_type: str  # "pdf", "txt", "note"
    page: Optional[int] = None  # For PDFs
    chunk_index: int
    relevance_score: float


class ChatMessage(BaseModel):
    """Chat message model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: Optional[str] = None  # Which model generated this (for assistant messages)
    citations: List[Citation] = Field(default_factory=list)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    session_id: Optional[str] = None


class ConversationMetadata(BaseModel):
    """Metadata about a conversation"""
    exists: bool
    timestamp: Optional[datetime] = None
    message_count: Optional[int] = None
    file: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LoadConversationRequest(BaseModel):
    """Request to load a previous conversation"""
    file: str


class LoadConversationResponse(BaseModel):
    """Response for loading a conversation"""
    status: str
    messages: List[ChatMessage]
