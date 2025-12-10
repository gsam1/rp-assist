"""Document management API endpoints"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
from pathlib import Path
import uuid
import json
from datetime import datetime

from app.models.document import (
    DocumentUploadResponse,
    DocumentListResponse,
    NotesListResponse,
    NoteContent,
    NoteUpdateRequest,
    NoteUpdateResponse,
    NoteMetadata,
    Document
)
from app.services.document_processor import get_document_processor
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import create_vector_store
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger()
router = APIRouter()


# In-memory document registry (in production, use a database)
documents_registry: List[Document] = []


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload and process a PDF or TXT document"""
    config = get_config()
    upload_dir = config.get('documents.upload_directory', 'data/uploads')
    Path(upload_dir).mkdir(parents=True, exist_ok=True)

    # Validate file type
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    # Save uploaded file
    document_id = str(uuid.uuid4())
    file_path = f"{upload_dir}/{document_id}_{file.filename}"

    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    logger.info(f"File uploaded: {file_path}")

    # Process document
    processor = get_document_processor()

    if file.filename.endswith('.pdf'):
        document, chunks = processor.process_pdf(file_path, document_id)
    else:
        document, chunks = processor.process_txt(file_path, document_id)

    # Generate embeddings
    embedding_service = get_embedding_service()
    texts = [chunk.content for chunk in chunks]
    embeddings = embedding_service.embed_batch(texts)

    # Add embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
        chunk.metadata['filename'] = file.filename

    # Store in vector database
    vector_store = create_vector_store()
    vector_store.add_documents(chunks)

    # Update document metadata
    document.indexed_at = datetime.utcnow()

    # Add to registry
    documents_registry.append(document)

    logger.info(f"Document processed and indexed: {file.filename}")

    return DocumentUploadResponse(
        status="success",
        document_id=document.id,
        filename=document.filename,
        pages=document.pages,
        chunks=document.chunks
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all uploaded documents"""
    return DocumentListResponse(documents=documents_registry)


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its embeddings"""
    global documents_registry

    # Find document
    document = next((d for d in documents_registry if d.id == document_id), None)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from vector store
    vector_store = create_vector_store()
    vector_store.delete_document(document_id)

    # Delete file
    if os.path.exists(document.path):
        os.remove(document.path)

    # Remove from registry
    documents_registry = [d for d in documents_registry if d.id != document_id]

    logger.info(f"Document deleted: {document.filename}")

    return {"status": "success", "message": "Document deleted"}


@router.get("/notes", response_model=NotesListResponse)
async def list_notes() -> NotesListResponse:
    """List all notes"""
    config = get_config()
    notes_dir = config.get('documents.notes_directory', 'data/notes')
    Path(notes_dir).mkdir(parents=True, exist_ok=True)

    notes = []
    for file_path in Path(notes_dir).glob("*.txt"):
        stat = file_path.stat()
        notes.append(NoteMetadata(
            filename=file_path.name,
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            indexed=True  # Assume indexed for now
        ))

    return NotesListResponse(notes=notes)


@router.get("/notes/{filename}", response_model=NoteContent)
async def get_note(filename: str) -> NoteContent:
    """Read a note file"""
    config = get_config()
    notes_dir = config.get('documents.notes_directory', 'data/notes')
    file_path = Path(notes_dir) / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Note not found")

    # Security check: ensure file is within notes directory
    if not str(file_path.resolve()).startswith(str(Path(notes_dir).resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    stat = file_path.stat()

    return NoteContent(
        filename=filename,
        content=content,
        modified_at=datetime.fromtimestamp(stat.st_mtime)
    )


@router.put("/notes/{filename}", response_model=NoteUpdateResponse)
async def update_note(filename: str, request: NoteUpdateRequest) -> NoteUpdateResponse:
    """Create or update a note file"""
    config = get_config()
    notes_dir = config.get('documents.notes_directory', 'data/notes')
    Path(notes_dir).mkdir(parents=True, exist_ok=True)

    file_path = Path(notes_dir) / filename

    # Security check
    if not str(file_path.resolve()).startswith(str(Path(notes_dir).resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    # Write file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(request.content)

    logger.info(f"Note saved: {filename}")

    # Re-index the note
    document_id = f"note_{filename}"

    # Process as text document
    processor = get_document_processor()
    document, chunks = processor.process_txt(str(file_path), document_id)

    # Generate embeddings
    embedding_service = get_embedding_service()
    texts = [chunk.content for chunk in chunks]
    embeddings = embedding_service.embed_batch(texts)

    # Add embeddings
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
        chunk.metadata['filename'] = filename
        chunk.metadata['type'] = 'note'

    # Store in vector database
    vector_store = create_vector_store()

    # Delete old chunks for this note
    try:
        vector_store.delete_document(document_id)
    except:
        pass  # Note might not have been indexed before

    # Add new chunks
    vector_store.add_documents(chunks)

    logger.info(f"Note indexed: {filename} ({len(chunks)} chunks)")

    return NoteUpdateResponse(
        status="success",
        filename=filename,
        indexed=True
    )


@router.delete("/notes/{filename}")
async def delete_note(filename: str):
    """Delete a note file"""
    config = get_config()
    notes_dir = config.get('documents.notes_directory', 'data/notes')
    file_path = Path(notes_dir) / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Note not found")

    # Security check
    if not str(file_path.resolve()).startswith(str(Path(notes_dir).resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    # Delete from vector store
    document_id = f"note_{filename}"
    vector_store = create_vector_store()
    try:
        vector_store.delete_document(document_id)
    except:
        pass

    # Delete file
    os.remove(file_path)

    logger.info(f"Note deleted: {filename}")

    return {"status": "success", "message": "Note deleted"}
