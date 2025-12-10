# RPG Assistant - Technical Specification

## Project Overview

RPG Assistant is a locally-hosted web application that serves as an alternative to NotebookLM, specifically tailored for Tabletop RPG content. It provides RAG (Retrieval Augmented Generation) capabilities with configurable LLM providers, document management, conversation persistence, and image generation.

## Core Requirements

### Target Users
- Single user (local development and personal use)
- TTRPG game masters and players
- Users who want full control over their AI assistant setup

### Key Features
1. Configurable LLM providers (GPT-4, Claude, Gemini, local models)
2. Document upload and management (PDF, TXT)
3. Vector database with RAG capabilities
4. Persistent conversation history with session management
5. Text file editing with auto-embedding
6. Image generation with iteration and saving
7. Source citation in LLM responses
8. Docker-based deployment with Make commands

---

## Technology Stack

### Frontend
- **Framework**: Svelte/SvelteKit
- **Communication**: WebSockets for real-time streaming
- **State Management**: Svelte stores

### Backend
- **Python Dependency Management**: uv
- **Framework**: FastAPI (Python)
- **WebSocket**: FastAPI WebSocket support
- **PDF Processing**: PyPDF2
- **Vector Database**: Abstracted (FAISS for local, PGVector for hosted)
- **Embedding**: Configurable embedding models

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Makefile
- **Database**: PostgreSQL with pgvector extension (optional)

### Development
- Hot-reloading enabled for both frontend and backend
- Volume mounts for local development

---

## Architecture

### High-Level Components

```
┌─────────────────┐
│  Svelte Frontend│
│   (Port 5173)   │
└────────┬────────┘
         │ WebSocket + HTTP
         │
┌────────▼────────┐
│  FastAPI Backend│
│   (Port 8000)   │
└────────┬────────┘
         │
    ┌────┴─────┬──────────┬──────────┐
    │          │          │          │
┌───▼───┐  ┌──▼──┐  ┌────▼─────┐ ┌─▼─────┐
│Vector │  │ LLM │  │ Image    │ │ File  │
│ Store │  │ API │  │ Gen API  │ │System │
└───────┘  └─────┘  └──────────┘ └───────┘
```

### Directory Structure

```
rpg-assistant/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   ├── documents.py
│   │   │   ├── images.py
│   │   │   └── config.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── llm_service.py
│   │   │   ├── embedding_service.py
│   │   │   ├── vector_store.py
│   │   │   ├── document_processor.py
│   │   │   └── image_service.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   ├── document.py
│   │   │   └── config.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── config_loader.py
│   │       └── logger.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── lib/
│   │   │   ├── components/
│   │   │   │   ├── Chat.svelte
│   │   │   │   ├── Sidebar.svelte
│   │   │   │   ├── FileExplorer.svelte
│   │   │   │   ├── TextEditor.svelte
│   │   │   │   ├── ImageGallery.svelte
│   │   │   │   └── DocumentList.svelte
│   │   │   ├── stores/
│   │   │   │   ├── chat.js
│   │   │   │   ├── documents.js
│   │   │   │   └── images.js
│   │   │   └── api/
│   │   │       └── client.js
│   │   ├── routes/
│   │   │   └── +page.svelte
│   │   └── app.html
│   ├── package.json
│   └── Dockerfile
├── data/
│   ├── uploads/          # User-uploaded PDFs and TXT files
│   ├── notes/            # User-created/edited notes
│   ├── images/           # Saved generated images
│   ├── conversations/    # Exported conversation markdown files
│   ├── vectordb/         # Vector database files (FAISS)
│   └── metadata/         # Document indexing metadata
├── logs/
│   └── app.log
├── config.toml
├── config.example.toml
├── .env.example
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Configuration

### `.env` File (Secrets - Not Committed)

```env
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Database (for PGVector)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=rpg_assistant
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password

# Optional: Local LLM endpoints
LOCAL_LLM_ENDPOINT=http://localhost:11434
```

### `config.toml` File (Configuration - Committed)

```toml
[app]
host = "0.0.0.0"
port = 8000
max_file_size_mb = 100  # 0 for unlimited
log_level = "INFO"

[llm]
default_provider = "openai"  # openai, anthropic, google, local
temperature = 0.7
max_tokens = 4096

[llm.context]
max_messages_before_summary = 20  # Keep last N messages in full
context_window_size = 128000      # Tokens available for context

[providers.openai]
model = "gpt-4o"
context_window = 128000
temperature = 0.7
max_tokens = 4096
top_p = 1.0

[providers.anthropic]
model = "claude-sonnet-4-20250514"
context_window = 200000
temperature = 0.7
max_tokens = 4096
top_p = 1.0

[providers.google]
model = "gemini-2.0-flash-exp"
context_window = 1000000
temperature = 0.7
max_tokens = 4096
top_p = 1.0

[providers.local]
endpoint = "http://localhost:11434/v1"  # OpenAI-compatible endpoint
model = "llama3.1:8b"
context_window = 8192
temperature = 0.7
max_tokens = 2048

[vectorstore]
type = "faiss"  # faiss, pgvector
chunk_size = 1000
chunk_overlap = 0.2  # 20% overlap
top_k = 5  # Number of chunks to retrieve

[embedding]
model = "text-embedding-3-large"  # Model name for embeddings
provider = "openai"  # Which provider to use for embeddings

[images]
size = "1024x1024"
format = "jpeg"
save_directory = "data/images"
provider = "openai"  # Which LLM provider handles image generation
model = "dall-e-3"  # Model for image generation

[documents]
upload_directory = "data/uploads"
notes_directory = "data/notes"
metadata_file = "data/metadata/index_metadata.json"

[conversations]
save_directory = "data/conversations"
auto_save_on_exit = true
```

### `config.example.toml`

A complete example configuration file with sensible defaults and comments explaining each option.

---

## API Endpoints

### Chat Endpoints

#### `POST /api/chat`
Send a message and receive streaming LLM response via WebSocket upgrade.

**Request Body:**
```json
{
  "message": "Tell me about dragons in D&D",
  "session_id": "uuid-string"
}
```

**WebSocket Events (Server -> Client):**
```json
// Message start
{ "event": "message_start", "session_id": "..." }

// Streaming chunks
{ "event": "message_chunk", "content": "Dragons are...", "session_id": "..." }

// Complete with citations
{
  "event": "message_complete",
  "session_id": "...",
  "citations": [
    { "source": "monster_manual.pdf", "page": 86, "chunk_index": 3 }
  ]
}

// Error
{ "event": "error", "message": "...", "session_id": "..." }
```

#### `POST /api/conversation/reset`
Clear the current conversation and start fresh.

**Response:**
```json
{
  "status": "success",
  "message": "Conversation reset"
}
```

#### `GET /api/conversation/last`
Get metadata about the last conversation session.

**Response:**
```json
{
  "exists": true,
  "timestamp": "2024-12-09T14:30:00Z",
  "message_count": 42,
  "file": "data/conversations/2024-12-09_143000_abc123.md"
}
```

#### `POST /api/conversation/load`
Load a previous conversation from markdown file.

**Request Body:**
```json
{
  "file": "data/conversations/2024-12-09_143000_abc123.md"
}
```

**Response:**
```json
{
  "status": "success",
  "messages": [
    { "role": "user", "content": "...", "timestamp": "..." },
    { "role": "assistant", "content": "...", "model": "gpt-4o", "citations": [...] }
  ]
}
```

### Document Endpoints

#### `POST /api/documents/upload`
Upload a PDF or TXT file.

**Form Data:**
- `file`: File upload

**WebSocket Events (during processing):**
```json
{ "event": "document_processing", "filename": "rulebook.pdf", "progress": 45, "stage": "extracting" }
{ "event": "document_processing", "filename": "rulebook.pdf", "progress": 100, "stage": "embedded" }
```

**Response:**
```json
{
  "status": "success",
  "document_id": "uuid",
  "filename": "rulebook.pdf",
  "pages": 200,
  "chunks": 150
}
```

#### `GET /api/documents`
List all uploaded documents.

**Response:**
```json
{
  "documents": [
    {
      "id": "uuid",
      "filename": "rulebook.pdf",
      "type": "pdf",
      "pages": 200,
      "chunks": 150,
      "uploaded_at": "2024-12-09T12:00:00Z",
      "indexed_at": "2024-12-09T12:01:30Z"
    }
  ]
}
```

#### `DELETE /api/documents/{document_id}`
Remove a document and its embeddings.

**Response:**
```json
{
  "status": "success",
  "message": "Document deleted"
}
```

#### `GET /api/notes`
List all notes in the notes directory.

**Response:**
```json
{
  "notes": [
    {
      "filename": "campaign_notes.txt",
      "size_bytes": 2048,
      "modified_at": "2024-12-09T13:00:00Z",
      "indexed": true
    }
  ]
}
```

#### `GET /api/notes/{filename}`
Read a note file.

**Response:**
```json
{
  "filename": "campaign_notes.txt",
  "content": "...",
  "modified_at": "2024-12-09T13:00:00Z"
}
```

#### `PUT /api/notes/{filename}`
Create or update a note file (auto-saves, triggers incremental indexing).

**Request Body:**
```json
{
  "content": "Updated note content..."
}
```

**Response:**
```json
{
  "status": "success",
  "filename": "campaign_notes.txt",
  "indexed": true
}
```

#### `DELETE /api/notes/{filename}`
Delete a note file.

**Response:**
```json
{
  "status": "success",
  "message": "Note deleted"
}
```

### Image Endpoints

#### `POST /api/images/generate`
Generate an image from a prompt.

**Request Body:**
```json
{
  "prompt": "A fierce red dragon perched on a mountain",
  "session_id": "uuid-string"
}
```

**WebSocket Events:**
```json
{ "event": "image_generating", "status": "started", "session_id": "..." }
{ "event": "image_generating", "status": "complete", "image_url": "/api/images/temp/uuid.jpg", "session_id": "..." }
```

**Response:**
```json
{
  "status": "success",
  "image_url": "/api/images/temp/uuid.jpg",
  "temp_id": "uuid"
}
```

#### `POST /api/images/save`
Save a generated image permanently.

**Request Body:**
```json
{
  "temp_id": "uuid",
  "custom_filename": "dragon_scene"  // Optional, will be appended to timestamp_hash
}
```

**Response:**
```json
{
  "status": "success",
  "filename": "20241209_143022_abc123_dragon_scene.jpg",
  "path": "data/images/20241209_143022_abc123_dragon_scene.jpg"
}
```

#### `GET /api/images`
List saved images.

**Response:**
```json
{
  "images": [
    {
      "filename": "20241209_143022_abc123_dragon_scene.jpg",
      "url": "/api/images/saved/20241209_143022_abc123_dragon_scene.jpg",
      "created_at": "2024-12-09T14:30:22Z",
      "size_bytes": 245678
    }
  ]
}
```

#### `GET /api/images/session`
List images from current session (saved + temporary).

**Response:**
```json
{
  "saved": [...],  // Same format as /api/images
  "temporary": [
    {
      "temp_id": "uuid",
      "url": "/api/images/temp/uuid.jpg",
      "prompt": "A fierce red dragon...",
      "generated_at": "2024-12-09T14:28:00Z"
    }
  ]
}
```

### Configuration Endpoints

#### `GET /api/config`
Get current configuration (non-sensitive parts).

**Response:**
```json
{
  "llm": {
    "default_provider": "openai",
    "current_model": "gpt-4o",
    "temperature": 0.7
  },
  "vectorstore": {
    "type": "faiss",
    "chunk_size": 1000,
    "top_k": 5
  },
  "available_providers": ["openai", "anthropic", "google", "local"]
}
```

#### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "llm_reachable": true,
  "vectorstore_initialized": true,
  "version": "1.0.0"
}
```

---

## Data Models

### Chat Message

```python
class ChatMessage:
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    model: Optional[str]  # Which model generated this (for assistant messages)
    citations: List[Citation]
```

### Citation

```python
class Citation:
    source: str  # Filename
    document_type: str  # "pdf", "txt", "note"
    page: Optional[int]  # For PDFs
    chunk_index: int
    relevance_score: float
```

### Document

```python
class Document:
    id: str
    filename: str
    type: str  # "pdf", "txt"
    path: str
    pages: Optional[int]
    chunks: int
    uploaded_at: datetime
    indexed_at: datetime
    file_hash: str
```

### Document Chunk

```python
class DocumentChunk:
    id: str
    document_id: str
    content: str
    chunk_index: int
    page: Optional[int]
    embedding: List[float]
    metadata: dict
```

### Index Metadata

```json
{
  "documents": {
    "uuid-1": {
      "filename": "rulebook.pdf",
      "indexed_at": "2024-12-09T12:00:00Z",
      "file_hash": "sha256...",
      "chunks": 150
    }
  },
  "last_reindex": "2024-12-09T12:00:00Z",
  "vectorstore_type": "faiss"
}
```

---

## Vector Store Abstraction

### Interface

```python
class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int) -> List[DocumentChunk]:
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass
    
    @abstractmethod
    def get_count(self) -> int:
        pass
```

### Implementations

#### FAISS Implementation
- Store index in `data/vectordb/faiss.index`
- Store metadata in `data/vectordb/metadata.pkl`
- Load on startup, persist on changes

#### PGVector Implementation
- Connect to PostgreSQL with pgvector extension
- Store embeddings in `document_chunks` table
- Connection details from `.env`

---

## Document Processing Pipeline

### PDF Processing
1. Upload via `/api/documents/upload`
2. Extract text using PyPDF2
3. Preserve page numbers
4. Split into chunks respecting paragraph boundaries
5. Generate embeddings for each chunk
6. Store in vector database
7. Update metadata file
8. Emit WebSocket progress events

### Text File Processing
1. Read file content
2. Split into chunks respecting paragraph boundaries
3. Generate embeddings
4. Store in vector database
5. Update metadata

### Note Processing (Auto-save)
1. User edits note in UI
2. Auto-save triggers after inactivity (2 seconds)
3. Backend receives updated content
4. Check if content changed (hash comparison)
5. If changed: re-embed and update vector store incrementally
6. Update metadata

### Chunking Strategy
1. Split text by paragraph boundaries (`\n\n`)
2. If paragraph > `chunk_size`, split by sentences
3. Apply `chunk_overlap` percentage between consecutive chunks
4. Preserve page numbers in chunk metadata
5. Store original text with each chunk

---

## LLM Integration

### Provider Abstraction

```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate_streaming(
        self, 
        messages: List[ChatMessage], 
        context_chunks: List[DocumentChunk]
    ) -> AsyncIterator[str]:
        pass
    
    @abstractmethod
    async def generate_image(self, prompt: str) -> bytes:
        pass
    
    @abstractmethod
    async def summarize_conversation(
        self, 
        messages: List[ChatMessage]
    ) -> str:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass
```

### Implementations
- OpenAI (GPT-4, GPT-4o, DALL-E)
- Anthropic (Claude Sonnet, Opus)
- Google (Gemini)
- Local (OpenAI-compatible endpoint)

### Context Management
1. Load conversation history from database
2. If total tokens > `context_window_size - buffer`:
   - Keep last `max_messages_before_summary` messages in full
   - Summarize older messages using LLM
   - Prepend summary to context
3. Add retrieved document chunks after summary/history
4. Add current user message
5. Send to LLM for generation

### Streaming Response
1. Client sends message via WebSocket
2. Backend retrieves relevant chunks from vector store
3. Builds context with history + chunks
4. Streams LLM response token by token
5. Collects citations from chunk metadata
6. Sends complete message with citations
7. Saves to conversation history

---

## Image Generation

### Workflow
1. User enters prompt in UI
2. Frontend sends to `/api/images/generate`
3. Backend uses configured image provider (multimodal LLM)
4. Returns temporary image URL
5. Image displayed in gallery (current session)
6. User can iterate (generate again) or save
7. On save: move to permanent storage with timestamp_hash_customname.jpg
8. Unsaved images are cleaned up on session end

### Gallery Display
- Grid of thumbnails
- Show saved images (persistent across sessions)
- Show temporary images (current session only)
- Click image to view full size
- Save button on temporary images
- Custom filename input on save

---

## Session Management

### Conversation Persistence

#### Markdown Format
```markdown
# RPG Assistant Conversation
**Started**: 2024-12-09 14:30:00
**Model**: gpt-4o
**Session ID**: abc123-def456

---

## Message 1
**User** (2024-12-09 14:30:15):
Tell me about dragons in D&D

**Assistant** (2024-12-09 14:30:22) [gpt-4o]:
Dragons are powerful reptilian creatures... [^1][^2]

**Citations**:
- [^1] monster_manual.pdf, page 86, chunk 3
- [^2] campaign_guide.txt, chunk 12

---

## Message 2
...
```

#### Session Lifecycle
1. **App Start**: Check for last conversation file
2. **User Prompt**: "Load last conversation? (Y/n)"
3. **If Yes**: Load markdown, parse into messages, inject into LLM context
4. **If No**: Start fresh session
5. **During Session**: Messages stored in memory
6. **On Exit/Reset**: Save current session to markdown in `data/conversations/`

### Metadata Tracking
File: `data/metadata/index_metadata.json`

Tracks:
- Document indexing timestamps
- File hashes for change detection
- Last reindex time
- Vectorstore type

Updated on:
- Document upload
- Note save
- Incremental index
- Full reindex

---

## Docker Setup

### docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config.toml:/app/config.toml:ro
      - ./.env:/app/.env:ro
    environment:
      - PYTHONUNBUFFERED=1
    depends_on:
      - postgres
    networks:
      - rpg-assistant

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - VITE_API_URL=http://localhost:8000
      - VITE_WS_URL=ws://localhost:8000
    networks:
      - rpg-assistant

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-rpg_assistant}
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - rpg-assistant

volumes:
  postgres_data:

networks:
  rpg-assistant:
    driver: bridge
```

### Backend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### Frontend Dockerfile

```dockerfile
FROM node:20-slim

WORKDIR /app

COPY package*.json .
RUN npm install

COPY . .

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
```

---

## Makefile Commands

```makefile
.PHONY: start stop restart build logs clean reindex index-new shell-backend shell-frontend validate-config help

help:
	@echo "RPG Assistant - Available commands:"
	@echo "  make start          - Start all services"
	@echo "  make stop           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make build          - Build Docker images"
	@echo "  make logs           - Tail logs from all services"
	@echo "  make clean          - Remove containers and volumes"
	@echo "  make reindex        - Full reindex of all documents"
	@echo "  make index-new      - Incremental index of new documents"
	@echo "  make shell-backend  - Open shell in backend container"
	@echo "  make shell-frontend - Open shell in frontend container"
	@echo "  make validate-config- Validate configuration files"

start:
	docker-compose up -d

stop:
	docker-compose down

restart:
	docker-compose restart

build:
	docker-compose build

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	rm -rf data/vectordb/*
	rm -rf data/metadata/*

reindex:
	docker-compose exec backend python -m app.cli reindex --full

index-new:
	docker-compose exec backend python -m app.cli reindex --incremental

shell-backend:
	docker-compose exec backend /bin/bash

shell-frontend:
	docker-compose exec frontend /bin/sh

validate-config:
	docker-compose exec backend python -m app.cli validate-config
```

---

## CLI Commands

### Backend CLI (`app/cli.py`)

```bash
# Full reindex
python -m app.cli reindex --full

# Incremental reindex
python -m app.cli reindex --incremental

# Validate configuration
python -m app.cli validate-config

# Health check
python -m app.cli health
```

---

## Error Handling & Logging

### Logging Configuration
- **Format**: Plain text
- **Level**: INFO (configurable in `config.toml`)
- **Location**: `logs/app.log`
- **Rotation**: Append indefinitely (manual cleanup)

### Log Entries
```
2024-12-09 14:30:00 INFO [app.main] Application started
2024-12-09 14:30:15 INFO [app.api.documents] Document uploaded: rulebook.pdf
2024-12-09 14:30:20 INFO [app.services.embedding] Embedding 150 chunks
2024-12-09 14:30:45 INFO [app.services.vector_store] Added 150 chunks to FAISS
2024-12-09 14:31:00 ERROR [app.services.llm] OpenAI API error: Rate limit exceeded
```

### Error Responses
All API errors return:
```json
{
  "error": "Error message",
  "detail": "Additional context",
  "timestamp": "2024-12-09T14:30:00Z"
}
```

---

## Security Considerations

### Current Scope (Local Only)
- No authentication required
- No rate limiting
- Trusted localhost environment
- Single user

### Future Considerations (If Hosted)
- Add authentication (JWT tokens)
- Implement rate limiting per user
- Add CORS restrictions
- Encrypt API keys in storage
- Add audit logging

---

## Testing Strategy

### Unit Tests
- LLM provider implementations
- Vector store implementations
- Document processing pipeline
- Chunking logic
- Context management

### Integration Tests
- API endpoints
- WebSocket streaming
- Document upload and embedding
- Conversation persistence
- Image generation and saving

### Manual Testing Checklist
- [ ] Upload PDF and verify extraction
- [ ] Upload TXT and verify chunking
- [ ] Create and edit notes
- [ ] Generate images and save
- [ ] Chat with RAG retrieval
- [ ] Verify source citations
- [ ] Test context summarization
- [ ] Load previous conversation
- [ ] Switch LLM providers
- [ ] Full reindex
- [ ] Incremental index

---

## Deployment Steps

### Initial Setup

1. **Clone repository**
   ```bash
   git clone <repo-url>
   cd rpg-assistant
   ```

2. **Copy configuration files**
   ```bash
   cp .env.example .env
   cp config.example.toml config.toml
   ```

3. **Edit `.env` with API keys**
   ```bash
   nano .env
   # Add your API keys
   ```

4. **Edit `config.toml` if needed**
   ```bash
   nano config.toml
   # Adjust models, chunk sizes, etc.
   ```

5. **Build and start**
   ```bash
   make build
   make start
   ```

6. **Validate configuration**
   ```bash
   make validate-config
   ```

7. **Access application**
   - Open browser to `http://localhost:5173`
   - Backend API at `http://localhost:8000`

### Updating

```bash
git pull
make build
make restart
```

### Backing Up

```bash
# Backup data directory
tar -czf rpg-assistant-backup-$(date +%Y%m%d).tar.gz data/

# Backup configuration
cp config.toml config.toml.backup
```

---

## UI Components

### Main Layout

```
┌─────────────────────────────────────────────────────┐
│  RPG Assistant                      [Settings] [?]  │
├─────────┬───────────────────────────────────────────┤
│         │                                           │
│ Sidebar │          Chat Area                        │
│         │                                           │
│ - Docs  │  ┌─────────────────────────────────────┐ │
│ - Notes │  │ User: Tell me about dragons         │ │
│ - Images│  │                                     │ │
│         │  │ Assistant: Dragons are powerful...  │ │
│         │  │ [1] monster_manual.pdf, p. 86       │ │
│         │  └─────────────────────────────────────┘ │
│         │                                           │
│         │  ┌─────────────────────────────────────┐ │
│         │  │ Type your message...                │ │
│         │  └─────────────────────────────────────┘ │
└─────────┴───────────────────────────────────────────┘
```

### Sidebar Components

#### Document List
- Shows uploaded PDFs and TXT files
- Delete button per document
- Upload button at top

#### Notes Section
- File explorer for notes directory
- Click to edit in text editor
- Create new note button

#### Image Gallery
- Grid of thumbnails
- Saved images (persistent)
- Current session images
- Click to enlarge
- Save button on temporary images

### Chat Component
- Single column message list
- User messages aligned right
- Assistant messages aligned left
- Citations as footnotes
- Auto-scroll to bottom
- Loading indicator during streaming

### Text Editor (Modal/Panel)
- Simple textarea
- Auto-save after 2 seconds inactivity
- Save status indicator
- Close button

### Image Generation (Modal)
- Prompt textarea
- Generate button
- Loading spinner
- Image preview
- Regenerate and Save buttons
- Custom filename input

---

## Performance Considerations

### Optimization Strategies
1. **Lazy Loading**: Load conversation history in chunks
2. **Caching**: Cache embedding model in memory
3. **Batching**: Batch embed multiple chunks together
4. **Indexing**: Use efficient vector search (FAISS IVF for large datasets)
5. **Streaming**: Stream LLM responses to reduce perceived latency

### Resource Limits
- Max file upload size: Configurable (default 100MB)
- Max chunks per document: Unlimited (but logged)
- Max conversation history: Configurable (default 20 full messages)
- Vector store size: Monitor and log warnings if > 10k chunks

---

## Troubleshooting

### Common Issues

#### "LLM provider not reachable"
- Check API keys in `.env`
- Verify network connectivity
- Check logs: `make logs`
- Test with: `docker-compose exec backend python -m app.cli health`

#### "Document failed to embed"
- Check PDF is not corrupted
- Verify encoding for TXT files
- Check logs for specific error
- Try smaller chunk size in `config.toml`

#### "Vector store initialization failed"
- For FAISS: Check disk space in `data/vectordb/`
- For PGVector: Verify PostgreSQL is running and accessible
- Check logs: `make logs | grep vector`

#### "WebSocket connection failed"
- Check backend is running: `docker-compose ps`
- Verify ports are not in use
- Check browser console for errors

#### "Out of context window"
- Reduce `max_messages_before_summary` in config
- Check LLM context window size matches config
- Manually reset conversation

---

## Future Enhancements

### Potential Features
1. **Multi-campaign support**: Separate workspaces per campaign
2. **Advanced search**: Full-text search across documents
3. **Export options**: PDF, DOCX export of conversations
4. **Voice input**: Speech-to-text for messages
5. **Collaborative mode**: Multi-user support
6. **Plugin system**: Extend with custom tools
7. **Mobile app**: React Native companion app
8. **Dice roller**: Integrated dice rolling with results in chat
9. **Character sheets**: Store and reference character data
10. **Map integration**: Upload and annotate maps

### Architecture Improvements
1. **Background workers**: Celery for async processing
2. **Queue system**: RabbitMQ for document processing
3. **Caching layer**: Redis for session management
4. **CDN**: For image delivery
5. **Database**: Full PostgreSQL for structured data (characters, campaigns)

---

## Appendix

### Dependencies

#### Backend (requirements.txt)
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
websockets==12.0
pydantic==2.5.0
python-dotenv==1.0.0
toml==0.10.2
PyPDF2==3.0.1
openai==1.3.5
anthropic==0.7.0
google-generativeai==0.3.0
faiss-cpu==1.7.4
numpy==1.26.2
psycopg2-binary==2.9.9
pgvector==0.2.3
python-jose==3.3.0
```

#### Frontend (package.json)
```json
{
  "name": "rpg-assistant-frontend",
  "version": "1.0.0",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "svelte": "^4.2.0",
    "@sveltejs/kit": "^1.27.0"
  },
  "devDependencies": {
    "@sveltejs/adapter-node": "^1.3.1",
    "vite": "^5.0.0"
  }
}
```

### Configuration Examples

#### Minimal config.toml
```toml
[app]
host = "0.0.0.0"
port = 8000

[llm]
default_provider = "openai"

[providers.openai]
model = "gpt-4o"
context_window = 128000

[vectorstore]
type = "faiss"
chunk_size = 1000
chunk_overlap = 0.2
top_k = 5

[embedding]
model = "text-embedding-3-large"
provider = "openai"
```

#### Development .env
```env
OPENAI_API_KEY=sk-test-key
ANTHROPIC_API_KEY=sk-ant-test-key
POSTGRES_PASSWORD=dev_password
```

---

## Glossary

- **RAG**: Retrieval Augmented Generation - technique for grounding LLM responses in specific documents
- **Embedding**: Vector representation of text for semantic search
- **Chunk**: Segment of a document used for embedding and retrieval
- **Vector Store**: Database optimized for similarity search on embeddings
- **Context Window**: Maximum tokens an LLM can process in one request
- **Streaming**: Sending LLM response incrementally as it's generated
- **WebSocket**: Protocol for real-time bidirectional communication
- **Session**: A conversation thread with persistent history

---

## Version History

### v1.0.0 (Initial Specification)
- Core RAG functionality
- Multi-provider LLM support
- Document management
- Image generation
- Conversation persistence
- Docker deployment

---

## Contact & Support

For questions or issues:
1. Check logs: `make logs`
2. Validate config: `make validate-config`
3. Review this specification
4. Open GitHub issue (if applicable)

---

**End of Specification**

*Last Updated: 2024-12-09*
*Document Version: 1.0.0*
