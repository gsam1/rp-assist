"""LLM service with multi-provider support"""

from abc import ABC, abstractmethod
from typing import List, AsyncIterator, Optional
import os

import openai
from anthropic import AsyncAnthropic
import google.generativeai as genai

from app.models.chat import ChatMessage
from app.models.document import DocumentChunk
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger()


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    async def generate_streaming(
        self,
        messages: List[ChatMessage],
        context_chunks: List[DocumentChunk]
    ) -> AsyncIterator[str]:
        """Generate streaming response"""
        pass

    @abstractmethod
    async def generate_image(self, prompt: str) -> bytes:
        """Generate an image from a prompt"""
        pass

    @abstractmethod
    async def summarize_conversation(self, messages: List[ChatMessage]) -> str:
        """Summarize a conversation"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is reachable"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation"""

    def __init__(self, config: dict):
        self.model = config.get('model', 'gpt-4o')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 4096)
        openai.api_key = os.getenv('OPENAI_API_KEY')

    async def generate_streaming(
        self,
        messages: List[ChatMessage],
        context_chunks: List[DocumentChunk]
    ) -> AsyncIterator[str]:
        """Generate streaming response from OpenAI"""

        # Build context from chunks
        context = self._build_context(context_chunks)

        # Convert messages to OpenAI format
        openai_messages = []

        # Add system message with context
        if context:
            openai_messages.append({
                "role": "system",
                "content": f"You are a helpful RPG assistant. Use the following context to answer questions:\n\n{context}"
            })

        # Add conversation history
        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Stream response
        stream = await openai.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_image(self, prompt: str) -> bytes:
        """Generate image using DALL-E"""
        response = await openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1
        )

        # Download image
        import httpx
        async with httpx.AsyncClient() as client:
            img_response = await client.get(response.data[0].url)
            return img_response.content

    async def summarize_conversation(self, messages: List[ChatMessage]) -> str:
        """Summarize conversation using OpenAI"""
        # Build summary prompt
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}" for msg in messages
        ])

        prompt = f"Summarize the following conversation concisely:\n\n{conversation_text}"

        response = await openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )

        return response.choices[0].message.content

    async def health_check(self) -> bool:
        """Check OpenAI API connectivity"""
        try:
            await openai.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False

    def _build_context(self, chunks: List[DocumentChunk]) -> str:
        """Build context string from chunks"""
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks):
            source = f"[Source: {chunk.metadata.get('filename', 'unknown')}"
            if chunk.page:
                source += f", Page {chunk.page}"
            source += "]"

            context_parts.append(f"{source}\n{chunk.content}\n")

        return "\n".join(context_parts)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation"""

    def __init__(self, config: dict):
        self.model = config.get('model', 'claude-sonnet-4-20250514')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 4096)
        self.client = AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    async def generate_streaming(
        self,
        messages: List[ChatMessage],
        context_chunks: List[DocumentChunk]
    ) -> AsyncIterator[str]:
        """Generate streaming response from Claude"""

        # Build context
        context = self._build_context(context_chunks)

        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = "You are a helpful RPG assistant."

        if context:
            system_message += f"\n\nUse the following context to answer questions:\n\n{context}"

        for msg in messages:
            anthropic_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Stream response
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_message,
            messages=anthropic_messages
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def generate_image(self, prompt: str) -> bytes:
        """Anthropic doesn't support image generation"""
        raise NotImplementedError("Anthropic does not support image generation")

    async def summarize_conversation(self, messages: List[ChatMessage]) -> str:
        """Summarize conversation using Claude"""
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}" for msg in messages
        ])

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=500,
            temperature=0.5,
            messages=[{
                "role": "user",
                "content": f"Summarize the following conversation concisely:\n\n{conversation_text}"
            }]
        )

        return response.content[0].text

    async def health_check(self) -> bool:
        """Check Anthropic API connectivity"""
        try:
            await self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False

    def _build_context(self, chunks: List[DocumentChunk]) -> str:
        """Build context string from chunks"""
        if not chunks:
            return ""

        context_parts = []
        for chunk in chunks:
            source = f"[Source: {chunk.metadata.get('filename', 'unknown')}"
            if chunk.page:
                source += f", Page {chunk.page}"
            source += "]"

            context_parts.append(f"{source}\n{chunk.content}\n")

        return "\n".join(context_parts)


class GoogleProvider(LLMProvider):
    """Google Gemini provider implementation"""

    def __init__(self, config: dict):
        self.model_name = config.get('model', 'gemini-2.0-flash-exp')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 4096)
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(self.model_name)

    async def generate_streaming(
        self,
        messages: List[ChatMessage],
        context_chunks: List[DocumentChunk]
    ) -> AsyncIterator[str]:
        """Generate streaming response from Gemini"""

        # Build context
        context = self._build_context(context_chunks)

        # Build prompt
        prompt_parts = []

        if context:
            prompt_parts.append(f"Context:\n{context}\n\n")

        # Add conversation history
        for msg in messages:
            prompt_parts.append(f"{msg.role.capitalize()}: {msg.content}\n")

        prompt = "".join(prompt_parts)

        # Stream response
        response = await self.model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            ),
            stream=True
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def generate_image(self, prompt: str) -> bytes:
        """Google Gemini doesn't support image generation in the same way"""
        raise NotImplementedError("Gemini does not support image generation")

    async def summarize_conversation(self, messages: List[ChatMessage]) -> str:
        """Summarize conversation using Gemini"""
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}" for msg in messages
        ])

        prompt = f"Summarize the following conversation concisely:\n\n{conversation_text}"

        response = await self.model.generate_content_async(prompt)
        return response.text

    async def health_check(self) -> bool:
        """Check Google API connectivity"""
        try:
            await self.model.generate_content_async("test")
            return True
        except Exception as e:
            logger.error(f"Google health check failed: {e}")
            return False

    def _build_context(self, chunks: List[DocumentChunk]) -> str:
        """Build context string from chunks"""
        if not chunks:
            return ""

        context_parts = []
        for chunk in chunks:
            source = f"[Source: {chunk.metadata.get('filename', 'unknown')}"
            if chunk.page:
                source += f", Page {chunk.page}"
            source += "]"

            context_parts.append(f"{source}\n{chunk.content}\n")

        return "\n".join(context_parts)


class LocalProvider(LLMProvider):
    """Local LLM provider (OpenAI-compatible endpoint)"""

    def __init__(self, config: dict):
        self.model = config.get('model', 'llama3.1:8b')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2048)
        self.endpoint = config.get('endpoint', 'http://localhost:11434/v1')

        # Use OpenAI client with custom base URL
        self.client = openai.AsyncOpenAI(
            base_url=self.endpoint,
            api_key="dummy"  # Local models often don't need a key
        )

    async def generate_streaming(
        self,
        messages: List[ChatMessage],
        context_chunks: List[DocumentChunk]
    ) -> AsyncIterator[str]:
        """Generate streaming response from local LLM"""

        # Build context
        context = self._build_context(context_chunks)

        # Convert messages
        openai_messages = []

        if context:
            openai_messages.append({
                "role": "system",
                "content": f"You are a helpful RPG assistant. Use the following context to answer questions:\n\n{context}"
            })

        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Stream response
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_image(self, prompt: str) -> bytes:
        """Local models typically don't support image generation"""
        raise NotImplementedError("Local provider does not support image generation")

    async def summarize_conversation(self, messages: List[ChatMessage]) -> str:
        """Summarize conversation using local LLM"""
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}" for msg in messages
        ])

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"Summarize the following conversation concisely:\n\n{conversation_text}"
            }],
            temperature=0.5,
            max_tokens=500
        )

        return response.choices[0].message.content

    async def health_check(self) -> bool:
        """Check local LLM connectivity"""
        try:
            await self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Local LLM health check failed: {e}")
            return False

    def _build_context(self, chunks: List[DocumentChunk]) -> str:
        """Build context string from chunks"""
        if not chunks:
            return ""

        context_parts = []
        for chunk in chunks:
            source = f"[Source: {chunk.metadata.get('filename', 'unknown')}"
            if chunk.page:
                source += f", Page {chunk.page}"
            source += "]"

            context_parts.append(f"{source}\n{chunk.content}\n")

        return "\n".join(context_parts)


def create_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """Factory function to create an LLM provider"""
    config = get_config()

    if provider_name is None:
        provider_name = config.get('llm.default_provider', 'openai')

    provider_config = config.get_section(f'providers.{provider_name}')

    if provider_name == 'openai':
        logger.info("Initializing OpenAI provider")
        return OpenAIProvider(provider_config)
    elif provider_name == 'anthropic':
        logger.info("Initializing Anthropic provider")
        return AnthropicProvider(provider_config)
    elif provider_name == 'google':
        logger.info("Initializing Google provider")
        return GoogleProvider(provider_config)
    elif provider_name == 'local':
        logger.info("Initializing Local provider")
        return LocalProvider(provider_config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
