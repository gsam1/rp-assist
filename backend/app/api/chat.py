"""Chat API endpoints with WebSocket support"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import uuid

from app.models.chat import (
    ChatMessage,
    Citation,
    ConversationMetadata,
    LoadConversationRequest,
    LoadConversationResponse
)
from app.services.llm_service import create_llm_provider
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import create_vector_store
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger()
router = APIRouter()


# In-memory conversation storage (in production, use a database)
conversation_history: List[ChatMessage] = []


@router.websocket("/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for chat with streaming responses"""
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            session_id = data.get("session_id", str(uuid.uuid4()))

            logger.info(f"Received message: {message[:50]}...")

            # Create user message
            user_message = ChatMessage(
                role="user",
                content=message
            )
            conversation_history.append(user_message)

            # Send message start event
            await websocket.send_json({
                "event": "message_start",
                "session_id": session_id
            })

            try:
                # Get embedding service and vector store
                embedding_service = get_embedding_service()
                vector_store = create_vector_store()

                # Generate query embedding
                query_embedding = embedding_service.embed_text(message)

                # Search for relevant chunks
                config = get_config()
                top_k = config.get('vectorstore.top_k', 5)
                relevant_chunks = vector_store.search(query_embedding, top_k)

                logger.info(f"Found {len(relevant_chunks)} relevant chunks")

                # Create LLM provider
                llm_provider = create_llm_provider()

                # Stream response
                response_content = ""

                async for chunk in llm_provider.generate_streaming(
                    conversation_history,
                    relevant_chunks
                ):
                    response_content += chunk

                    # Send chunk to client
                    await websocket.send_json({
                        "event": "message_chunk",
                        "content": chunk,
                        "session_id": session_id
                    })

                # Create citations
                citations = []
                for i, chunk in enumerate(relevant_chunks):
                    citation = Citation(
                        source=chunk.metadata.get('filename', 'unknown'),
                        document_type=chunk.metadata.get('type', 'pdf'),
                        page=chunk.page,
                        chunk_index=chunk.chunk_index,
                        relevance_score=1.0 - (i * 0.1)  # Simple relevance scoring
                    )
                    citations.append(citation)

                # Create assistant message
                config = get_config()
                current_model = config.get(f"providers.{config.get('llm.default_provider')}.model")

                assistant_message = ChatMessage(
                    role="assistant",
                    content=response_content,
                    model=current_model,
                    citations=citations
                )
                conversation_history.append(assistant_message)

                # Send completion event
                await websocket.send_json({
                    "event": "message_complete",
                    "session_id": session_id,
                    "citations": [c.dict() for c in citations]
                })

                logger.info("Message processing complete")

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "event": "error",
                    "message": str(e),
                    "session_id": session_id
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


@router.post("/conversation/reset")
async def reset_conversation():
    """Clear the current conversation and start fresh"""
    global conversation_history
    conversation_history = []
    logger.info("Conversation reset")

    return {
        "status": "success",
        "message": "Conversation reset"
    }


@router.get("/conversation/last", response_model=ConversationMetadata)
async def get_last_conversation() -> ConversationMetadata:
    """Get metadata about the last conversation session"""
    config = get_config()
    conv_dir = config.get('conversations.save_directory', 'data/conversations')
    Path(conv_dir).mkdir(parents=True, exist_ok=True)

    # Find most recent conversation file
    conv_files = sorted(Path(conv_dir).glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not conv_files:
        return ConversationMetadata(exists=False)

    latest_file = conv_files[0]
    stat = latest_file.stat()

    # Count messages (very rough estimate based on file size)
    with open(latest_file, 'r') as f:
        content = f.read()
        message_count = content.count("**User**") + content.count("**Assistant**")

    return ConversationMetadata(
        exists=True,
        timestamp=datetime.fromtimestamp(stat.st_mtime),
        message_count=message_count,
        file=str(latest_file)
    )


@router.post("/conversation/load", response_model=LoadConversationResponse)
async def load_conversation(request: LoadConversationRequest) -> LoadConversationResponse:
    """Load a previous conversation from markdown file"""
    global conversation_history

    file_path = Path(request.file)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Conversation file not found")

    # Parse markdown file (basic parsing)
    # In a production system, use a proper markdown parser
    messages = _parse_conversation_markdown(str(file_path))

    # Set as current conversation
    conversation_history = messages

    logger.info(f"Loaded conversation from {request.file}")

    return LoadConversationResponse(
        status="success",
        messages=messages
    )


@router.post("/conversation/save")
async def save_conversation():
    """Save the current conversation to a markdown file"""
    if not conversation_history:
        raise HTTPException(status_code=400, detail="No conversation to save")

    config = get_config()
    conv_dir = config.get('conversations.save_directory', 'data/conversations')
    Path(conv_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{session_id}.md"
    file_path = Path(conv_dir) / filename

    # Generate markdown content
    markdown = _generate_conversation_markdown(conversation_history)

    # Write file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    logger.info(f"Conversation saved to {file_path}")

    return {
        "status": "success",
        "file": str(file_path)
    }


def _generate_conversation_markdown(messages: List[ChatMessage]) -> str:
    """Generate markdown content from conversation history"""
    config = get_config()
    current_model = config.get(f"providers.{config.get('llm.default_provider')}.model")

    lines = [
        "# RPG Assistant Conversation",
        f"**Started**: {messages[0].timestamp.strftime('%Y-%m-%d %H:%M:%S') if messages else 'Unknown'}",
        f"**Model**: {current_model}",
        f"**Session ID**: {str(uuid.uuid4())[:8]}",
        "",
        "---",
        ""
    ]

    for i, msg in enumerate(messages, 1):
        lines.append(f"## Message {i}")

        if msg.role == "user":
            lines.append(f"**User** ({msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}):")
            lines.append(msg.content)
        else:
            lines.append(f"**Assistant** ({msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}) [{msg.model or 'unknown'}]:")
            lines.append(msg.content)

            if msg.citations:
                lines.append("")
                lines.append("**Citations**:")
                for j, citation in enumerate(msg.citations, 1):
                    cite_text = f"- [{j}] {citation.source}"
                    if citation.page:
                        cite_text += f", page {citation.page}"
                    cite_text += f", chunk {citation.chunk_index}"
                    lines.append(cite_text)

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _parse_conversation_markdown(file_path: str) -> List[ChatMessage]:
    """Parse conversation from markdown file (basic implementation)"""
    # This is a simplified parser - in production, use a proper markdown parser
    messages = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Simple regex-based parsing (not robust, but functional)
    import re

    # Find all user messages
    user_pattern = r'\*\*User\*\* \((.+?)\):\s*\n(.+?)(?=\n\n\*\*|---|\Z)'
    user_matches = re.findall(user_pattern, content, re.DOTALL)

    for timestamp_str, msg_content in user_matches:
        messages.append(ChatMessage(
            role="user",
            content=msg_content.strip()
        ))

    # Find all assistant messages
    assistant_pattern = r'\*\*Assistant\*\* \((.+?)\) \[(.+?)\]:\s*\n(.+?)(?=\n\n\*\*Citations|---|\Z)'
    assistant_matches = re.findall(assistant_pattern, content, re.DOTALL)

    for timestamp_str, model, msg_content in assistant_matches:
        messages.append(ChatMessage(
            role="assistant",
            content=msg_content.strip(),
            model=model
        ))

    return messages
