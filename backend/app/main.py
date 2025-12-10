"""Main FastAPI application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

from app.utils.config_loader import get_config
from app.utils.logger import setup_logger, get_logger

# Initialize configuration and logging
config = get_config()
logger = setup_logger(
    log_level=config.get('app.log_level', 'INFO'),
    log_file="logs/app.log"
)

# Create FastAPI app
app = FastAPI(
    title="RPG Assistant",
    description="RAG-powered assistant for Tabletop RPG content",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directories exist
data_dirs = [
    "data/uploads",
    "data/notes",
    "data/images",
    "data/conversations",
    "data/metadata",
    "logs"
]

for directory in data_dirs:
    Path(directory).mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("RPG Assistant application starting...")
    logger.info(f"Configuration loaded from: config.toml")
    logger.info(f"Default LLM provider: {config.get('llm.default_provider')}")
    logger.info(f"Vector store type: {config.get('vectorstore.type')}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("RPG Assistant application shutting down...")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "RPG Assistant",
        "version": "1.0.0",
        "status": "running"
    }


# Import and register API routers
# These will be added as we create the API endpoint files
try:
    from app.api import config as config_api
    app.include_router(config_api.router, prefix="/api", tags=["config"])
except ImportError:
    logger.warning("Config API not yet implemented")

try:
    from app.api import chat
    app.include_router(chat.router, prefix="/api", tags=["chat"])
except ImportError:
    logger.warning("Chat API not yet implemented")

try:
    from app.api import documents
    app.include_router(documents.router, prefix="/api", tags=["documents"])
except ImportError:
    logger.warning("Documents API not yet implemented")

try:
    from app.api import images
    app.include_router(images.router, prefix="/api", tags=["images"])
except ImportError:
    logger.warning("Images API not yet implemented")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.get('app.host', '0.0.0.0'),
        port=config.get('app.port', 8000),
        reload=True
    )
