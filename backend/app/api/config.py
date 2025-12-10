"""Configuration and health check API endpoints"""

from fastapi import APIRouter
from typing import Dict, Any

from app.utils.config_loader import get_config
from app.services.llm_service import create_llm_provider
from app.services.vector_store import create_vector_store
from app.utils.logger import get_logger
from app import __version__

logger = get_logger()
router = APIRouter()


@router.get("/config")
async def get_configuration() -> Dict[str, Any]:
    """Get current configuration (non-sensitive parts)"""
    config = get_config()

    return {
        "llm": {
            "default_provider": config.get('llm.default_provider'),
            "current_model": config.get(f"providers.{config.get('llm.default_provider')}.model"),
            "temperature": config.get('llm.temperature')
        },
        "vectorstore": {
            "type": config.get('vectorstore.type'),
            "chunk_size": config.get('vectorstore.chunk_size'),
            "top_k": config.get('vectorstore.top_k')
        },
        "available_providers": ["openai", "anthropic", "google", "local"]
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    config = get_config()
    llm_reachable = False
    vectorstore_initialized = False

    # Check LLM provider
    try:
        provider = create_llm_provider()
        llm_reachable = await provider.health_check()
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")

    # Check vector store
    try:
        store = create_vector_store()
        vectorstore_initialized = True
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")

    status = "healthy" if (llm_reachable and vectorstore_initialized) else "degraded"

    return {
        "status": status,
        "llm_reachable": llm_reachable,
        "vectorstore_initialized": vectorstore_initialized,
        "version": __version__
    }
