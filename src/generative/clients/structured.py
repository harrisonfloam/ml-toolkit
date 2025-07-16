"""Structured LLM Clients

Clients that return structured Pydantic models using Instructor.
"""

from typing import TypeVar

import instructor
import openai
from pydantic import BaseModel, Field

from ..settings import settings
from .base import BaseLLMClient, ChatMessage

T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel):
    """Default response model for structured outputs."""

    response: str = Field(..., description="The generated text from the LLM")


class StructuredOpenAIClient(BaseLLMClient):
    """OpenAI client with structured outputs using Instructor."""

    def __init__(self, log_level: str = settings.log_level, **openai_kwargs):
        super().__init__(log_level=log_level)
        openai_client = openai.OpenAI(**openai_kwargs)
        self.client = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.JSON,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        model: str = settings.model_name,
        response_model: type[T] = LLMResponse,
        temperature: float = settings.temperature,
        max_retries: int = getattr(settings, "instructor_max_retries", 3),
        **kwargs,
    ) -> T:
        """Generate a structured response using OpenAI's chat endpoint."""
        payload = {
            "messages": messages,
            "max_retries": max_retries,
            "model": model,
            "temperature": temperature,
            "response_model": response_model,
            **kwargs,
        }
        response = self.client.chat.completions.create(**payload)
        return response


class AsyncStructuredOpenAIClient(BaseLLMClient):
    """Async OpenAI client with structured outputs using Instructor."""

    def __init__(self, log_level: str = settings.log_level, **openai_kwargs):
        super().__init__(log_level=log_level)
        openai_client = openai.AsyncOpenAI(**openai_kwargs)
        self.client = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.JSON,
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str = settings.model_name,
        response_model: type[T] = LLMResponse,
        temperature: float = settings.temperature,
        max_retries: int = getattr(settings, "instructor_max_retries", 3),
        **kwargs,
    ) -> T:
        """Generate a structured response using OpenAI's async chat endpoint."""
        payload = {
            "messages": messages,
            "max_retries": max_retries,
            "model": model,
            "temperature": temperature,
            "response_model": response_model,
            **kwargs,
        }
        response = await self.client.chat.completions.create(**payload)
        return response


class StructuredOllamaClient(StructuredOpenAIClient):
    """Ollama client with structured outputs, inherits from OpenAI structured client."""

    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(
            log_level=log_level,
            base_url=f"{ollama_url}/v1",
            api_key="ollama",  # required, but unused
        )


class AsyncStructuredOllamaClient(AsyncStructuredOpenAIClient):
    """Async Ollama client with structured outputs, inherits from async OpenAI structured client."""

    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(
            log_level=log_level,
            base_url=f"{ollama_url}/v1",
            api_key="ollama",  # required, but unused
        )
