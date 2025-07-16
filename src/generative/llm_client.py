"""LLM Client Module

Function:
- Adds custom logging and callbacks
- Abstracts the client interface (allows for swapping out openai if needed)
"""

import json
import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Awaitable, Dict, List, Optional, TypeVar, Union

import instructor
import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, Field

from src.generative.settings import settings
from src.utils.callbacks import CallbackMeta, with_callbacks

logger = logging.getLogger(__name__)

ChatMessage = Dict[str, Any]


class BaseLLMClient(ABC, metaclass=CallbackMeta):
    def __init__(self, log_level: str = settings.log_level, *args, **kwargs):
        logger.setLevel(log_level.upper())
        logger.debug(f"{self.__class__.__name__} initialized.")

    @with_callbacks
    @abstractmethod
    def chat(self, *args, **kwargs) -> Union[Any, Awaitable[Any]]:
        pass

    def _pre_chat(self, *args, **kwargs):
        payload = {**kwargs, **dict(enumerate(args))}
        logger.debug(f"Chat request:\n{json.dumps(payload, indent=2, default=str)}")

    def _post_chat(self, result, duration, *args, **kwargs):
        logger.debug(
            f"Chat response in {duration:.4f}s:\n{json.dumps(result.model_dump(), indent=2, default=str)}"
        )
        return result


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, log_level: str = settings.log_level, **openai_kwargs):
        super().__init__(log_level=log_level)
        self.client = openai.OpenAI(**openai_kwargs)

    def chat(
        self,
        messages: List[ChatMessage],
        model: str = settings.model_name,
        temperature: float = settings.temperature,
        **kwargs,
    ) -> ChatCompletion:
        payload = {
            "model": model,
            "messages": messages,  # type: ignore
            "temperature": temperature,
            **kwargs,
        }
        response = self.client.chat.completions.create(stream=False, **payload)
        return response


class AsyncOpenAILLMClient(BaseLLMClient):
    def __init__(self, log_level: str = settings.log_level, **openai_kwargs):
        super().__init__(log_level=log_level)
        self.client = openai.AsyncOpenAI(**openai_kwargs)

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str = settings.model_name,
        temperature: float = settings.temperature,
        **kwargs,
    ) -> ChatCompletion:
        payload = {
            "model": model,
            "messages": messages,  # type: ignore
            "temperature": temperature,
            **kwargs,
        }
        response = await self.client.chat.completions.create(stream=False, **payload)
        return response


class OllamaLLMClient(OpenAILLMClient):
    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(
            log_level=log_level,
            base_url=ollama_url,
            api_key="ollama",  # required, but unused
        )


class AsyncOllamaLLMClient(AsyncOpenAILLMClient):
    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(
            log_level=log_level,
            base_url=ollama_url,
            api_key="ollama",  # required, but unused
        )


class MockLLMClient(BaseLLMClient):
    """Base mock LLM client with configurable response types and seeding."""

    def __init__(
        self,
        response_type: Optional[int] = None,
        seed: Optional[int] = None,
        log_level: str = settings.log_level,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(log_level=log_level, **kwargs)
        self.response_type = response_type
        if seed is not None:
            random.seed(seed)

    def _create_mock_response(self, content: str) -> Union[ChatCompletion, Any]:
        """Create a mock ChatCompletion with the given content."""
        return ChatCompletion(
            id="mock_id",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model="mock-llm",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=content,
                    ),
                    finish_reason="stop",
                )
            ],
        )

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str = settings.model_name,
        temperature: float = settings.temperature,
        **kwargs,
    ) -> Union[ChatCompletion, Any]:
        user_message = messages[-1]["content"]

        mock_responses = [
            user_message,
            "This is a mock response.",
            f"This is a mock response from {model} with temperature {temperature}.",
            f"Let me think about '{user_message}'...",
        ]

        if self.response_type is None:
            content = random.choice(mock_responses)
        else:
            # Ensure response_type is within bounds
            content = mock_responses[abs(self.response_type) % len(mock_responses)]

        return self._create_mock_response(content)


# Structured clients with Instructor - integrated with existing patterns
T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel):
    response: str = Field(..., description="The generated text from the LLM")


class StructuredOllamaClient(BaseLLMClient):
    """
    An LLM Client that relies on the Ollama API, wrapped with Instructor for structured outputs.

    Example:
        >>> client = StructuredOllamaClient()
        >>> result = client.chat([{"role": "user", "content": "What is the capital of France?"}])
        >>> print(result)
        LLMResponse(response='Paris')
    """

    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(log_level=log_level)
        openai_client = openai.OpenAI(
            base_url=f"{ollama_url}/v1",
            api_key="ollama",  # required, but unused
        )
        # Wrap the client with Instructor to enable structured outputs
        self.client = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.JSON,
        )

    def chat(
        self,
        messages: List[ChatMessage],
        model: str = settings.model_name,
        response_model: type[T] = LLMResponse,
        temperature: float = settings.temperature,
        max_retries: int = getattr(settings, "instructor_max_retries", 3),
        **kwargs,
    ) -> T:
        """Generate a structured response from the LLM using the chat endpoint."""
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


class AsyncStructuredOllamaClient(BaseLLMClient):
    """Async version of StructuredOllamaClient."""

    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(log_level=log_level)
        openai_client = openai.AsyncOpenAI(
            base_url=f"{ollama_url}/v1",
            api_key="ollama",  # required, but unused
        )
        self.client = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.JSON,
        )

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str = settings.model_name,
        response_model: type[T] = LLMResponse,
        temperature: float = settings.temperature,
        max_retries: int = getattr(settings, "instructor_max_retries", 3),
        **kwargs,
    ) -> T:
        """Generate a structured response from the LLM using the chat endpoint."""
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


class MockStructuredClient(MockLLMClient):
    """Mock structured LLM client that returns realistic fake data instantly."""

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str = "mock-model",
        temperature: float = 0.0,
        response_model: type[T] = LLMResponse,
        max_retries: int = 1,
        **kwargs,
    ) -> T:
        """Generate a mock structured response instantly."""
        schema = response_model.model_json_schema()
        included_fields = schema.get("properties", {}).keys()

        user_message = messages[-1]["content"] if messages else "Hello"
        mock_responses = [
            f"Mock response for: {user_message[:50]}...",
            "This is a mock structured response.",
            f"Mock {response_model.__name__.lower()} from {model}.",
            f"Generated mock data for: '{user_message[:30]}'...",
        ]

        mock_data = {}
        for field_name in included_fields:
            if field_name == "response":
                if self.response_type is None:
                    mock_data[field_name] = random.choice(mock_responses)
                else:
                    mock_data[field_name] = mock_responses[
                        abs(self.response_type) % len(mock_responses)
                    ]
            elif field_name == "content":
                mock_data[field_name] = (
                    f"Mock {response_model.__name__.lower()} content for: {user_message[:30]}..."
                )
            else:
                mock_data[field_name] = f"Mock {field_name}"

        result = response_model(**mock_data)
        return result
