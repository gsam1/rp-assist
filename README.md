# RPG Assistant

A locally-hosted RAG (Retrieval Augmented Generation) web application for Tabletop RPG content. Upload your rulebooks, campaign notes, and other documents, then chat with an AI assistant that has access to your entire RPG library.

## Features

- **Multi-Provider LLM Support**: Choose between OpenAI GPT-4, Anthropic Claude, Google Gemini, or local models
- **Document Management**: Upload and index PDF and TXT files
- **Note-Taking**: Create and edit notes that are automatically indexed
- **RAG Chat**: Ask questions and get answers grounded in your documents with source citations
- **Image Generation**: Generate RPG artwork using DALL-E 3
- **Conversation Persistence**: Save and load previous conversations
- **Vector Search**: Efficient semantic search using PGVector or FAISS
- **Docker Deployment**: Easy setup with Docker Compose

## Quick Start

### Prerequisites

- Docker and Docker Compose
- API keys for your chosen LLM provider (OpenAI, Anthropic, and/or Google)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rpg-assistant
   ```

2. **Copy configuration files**
   ```bash
   cp config.example.toml config.toml
   cp .env.example .env
   ```

3. **Edit `.env` with your API keys**
   ```bash
   nano .env
   ```

   Add your API keys:
   ```env
   OPENAI_API_KEY=sk-your-key-here
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   GOOGLE_API_KEY=your-google-key-here
   ```

4. **Edit `config.toml` (optional)**

   Adjust settings like:
   - Default LLM provider
   - Model names
   - Chunk sizes
   - Vector store type (pgvector or faiss)

5. **Build and start services**
   ```bash
   make build
   make start
   ```

6. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000

## Configuration

### LLM Providers

Configure providers in `config.toml`:

```toml
[llm]
default_provider = "openai"  # openai, anthropic, google, local

[providers.openai]
model = "gpt-4o"
context_window = 128000

[providers.anthropic]
model = "claude-sonnet-4-20250514"
context_window = 200000

[providers.google]
model = "gemini-2.0-flash-exp"
context_window = 1000000
```

### Vector Store

Choose between PGVector (PostgreSQL) or FAISS (file-based):

```toml
[vectorstore]
type = "pgvector"  # or "faiss"
chunk_size = 1000
chunk_overlap = 0.2
top_k = 5
```

### Embedding

Configure embedding provider:

```toml
[embedding]
model = "text-embedding-3-large"
provider = "openai"
```

## Usage

### Uploading Documents

1. Click the **Documents** tab in the sidebar
2. Click **+ Upload**
3. Select PDF or TXT files
4. Documents are automatically processed, chunked, and indexed

### Creating Notes

1. Click the **Notes** tab in the sidebar
2. Click **+ New**
3. Enter a filename
4. Write your notes in the editor
5. Notes auto-save and are automatically indexed

### Chatting

1. Type your question in the chat input
2. The AI will search your documents and provide answers with source citations
3. Citations show which documents and pages were used

### Generating Images

1. Click the **Images** tab in the sidebar
2. Click **+ Generate**
3. Describe the image you want
4. Generated images can be saved permanently

## Make Commands

Common operations:

```bash
make help           # Show all available commands
make start          # Start all services
make stop           # Stop all services
make restart        # Restart all services
make build          # Build Docker images
make logs           # View logs from all services
make clean          # Remove containers and volumes
make reindex        # Full reindex of all documents
make index-new      # Incremental index of new documents
make shell-backend  # Open shell in backend container
make validate-config # Validate configuration files
```

## Project Structure

```
rpg-assistant/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── services/       # Business logic
│   │   ├── models/         # Data models
│   │   ├── utils/          # Utilities
│   │   ├── db/             # Database schema
│   │   ├── main.py         # FastAPI app
│   │   └── cli.py          # CLI commands
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                # Svelte frontend
│   ├── src/
│   │   ├── lib/
│   │   │   ├── components/ # UI components
│   │   │   ├── stores/     # State management
│   │   │   └── api/        # API client
│   │   └── routes/         # Pages
│   ├── package.json
│   └── Dockerfile
├── data/                    # Application data
│   ├── uploads/            # Uploaded documents
│   ├── notes/              # User notes
│   ├── images/             # Generated images
│   ├── conversations/      # Saved conversations
│   └── metadata/           # Index metadata
├── logs/                    # Application logs
├── config.toml             # Configuration
├── .env                    # Secrets (API keys)
├── docker-compose.yml      # Container orchestration
├── Makefile                # Commands
└── README.md               # This file
```

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API documentation.

### Key Endpoints

- `GET /api/health` - Health check
- `GET /api/config` - Get configuration
- `WS /api/chat` - WebSocket chat endpoint
- `POST /api/documents/upload` - Upload document
- `GET /api/documents` - List documents
- `GET /api/notes` - List notes
- `PUT /api/notes/{filename}` - Create/update note
- `POST /api/images/generate` - Generate image

## Troubleshooting

### Backend won't start

```bash
# Check logs
make logs

# Validate configuration
make validate-config

# Check if PostgreSQL is running
docker compose ps
```

### LLM provider not reachable

1. Check API keys in `.env`
2. Verify network connectivity
3. Run health check: `make validate-config`

### Vector store errors

**For PGVector:**
- Ensure PostgreSQL is running: `docker compose ps`
- Check PostgreSQL logs: `docker compose logs postgres`

**For FAISS:**
- Check disk space in `data/vectordb/`
- Try clearing and reindexing: `make clean && make start && make reindex`

### Frontend can't connect to backend

1. Check backend is running: `docker compose ps`
2. Verify ports are not in use
3. Check browser console for errors

## Development

### Hot Reloading

Both frontend and backend support hot reloading:

- **Backend**: Edit Python files, uvicorn auto-reloads
- **Frontend**: Edit Svelte files, Vite auto-reloads

### Adding a New LLM Provider

1. Implement `LLMProvider` interface in `backend/app/services/llm_service.py`
2. Add provider configuration to `config.example.toml`
3. Update factory function in `create_llm_provider()`

### Custom CLI Commands

Add new commands to `backend/app/cli.py`:

```python
def my_command():
    """Your command logic"""
    pass

# Add to main():
subparsers.add_parser('my-command', help='Description')
```

## Technology Stack

### Backend
- **Framework**: FastAPI
- **Vector DB**: PGVector / FAISS
- **Embedding**: OpenAI Embeddings
- **LLMs**: OpenAI GPT-4, Anthropic Claude, Google Gemini, Local models
- **Document Processing**: PyPDF2
- **WebSocket**: FastAPI WebSocket support

### Frontend
- **Framework**: SvelteKit
- **Build Tool**: Vite
- **State**: Svelte Stores
- **Communication**: WebSocket + REST API

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Database**: PostgreSQL with pgvector extension
- **Orchestration**: Make

## Security Notes

- This application is designed for local use
- No authentication by default
- API keys are stored in `.env` (not committed)
- Keep `.env` and `config.toml` secure

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here]

## Support

For issues and questions:
1. Check logs: `make logs`
2. Validate config: `make validate-config`
3. Review troubleshooting section
4. Open a GitHub issue

## Acknowledgments

- Built with FastAPI, Svelte, and modern AI technologies
- Inspired by NotebookLM for the TTRPG community
