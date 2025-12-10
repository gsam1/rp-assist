"""Document models"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4


class DocumentChunk(BaseModel):
    """Document chunk with embedding"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    chunk_index: int
    page: Optional[int] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Document metadata"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    type: str  # "pdf", "txt"
    path: str
    pages: Optional[int] = None
    chunks: int = 0
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: Optional[datetime] = None
    file_hash: str = ""

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    status: str
    document_id: str
    filename: str
    pages: Optional[int] = None
    chunks: int


class DocumentListResponse(BaseModel):
    """Response for listing documents"""
    documents: List[Document]


class NoteMetadata(BaseModel):
    """Metadata for a note file"""
    filename: str
    size_bytes: int
    modified_at: datetime
    indexed: bool = False

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NoteContent(BaseModel):
    """Content of a note file"""
    filename: str
    content: str
    modified_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NoteUpdateRequest(BaseModel):
    """Request to update a note"""
    content: str


class NoteUpdateResponse(BaseModel):
    """Response for note update"""
    status: str
    filename: str
    indexed: bool


class NotesListResponse(BaseModel):
    """Response for listing notes"""
    notes: List[NoteMetadata]
