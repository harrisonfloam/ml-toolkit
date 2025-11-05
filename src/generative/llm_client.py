"""Unified LLM Client

A single-class LLM client that handles OpenAI and Ollama providers
with support for sync/async, streaming, and structured outputs.

Features:
- Supports standard chat completions and structured outputs
- Async and streaming support
- Built-in logging with callbacks
"""

import json
import logging
from typing import Any, AsyncIterator, Optional, TypeVar, Union

import instructor
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field

from ..utils.callbacks import CallbackMeta, with_callbacks
from ..utils.logging import truncate_long_strings
from .settings import settings

logger = logging.getLogger(__name__)

# Type definitions
ChatMessage = dict[str, Any]
T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel):
    """Default response model for structured outputs."""

    response: str = Field(..., description="The generated text from the LLM")


class LLMClient(metaclass=CallbackMeta):
    """Unified LLM client supporting OpenAI and Ollama.

    Args:
        model: Default model name for all requests, can be specified per method call
        temperature: Default temperature for all requests
        is_async: Whether to initialize async client first, defaults to False for sync
        base_url: Base URL for API, use for Ollama, leave None for OpenAI
        api_key: API key, default from environment for OpenAI
        is_mock: Placeholder for mock mode (TODO: implement mock functionality)
        log_level: Logging level, default from settings
        **openai_kwargs: Additional kwargs passed to OpenAI client initialization
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = settings.temperature,
        is_async: bool = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        is_mock: bool = False,  # TODO: implement mock functionality
        log_level: str = settings.log_level,
        **openai_kwargs,
    ):
        # Set up logging
        logger.setLevel(log_level.upper())
        logger.debug(f"{self.__class__.__name__} initialized")

        # Store defaults
        self.model = model
        self.temperature = temperature
        self.is_async = is_async

        self.is_mock = is_mock

        # Store client kwargs for lazy initialization
        self._client_kwargs = {}
        if base_url:
            self._client_kwargs["base_url"] = base_url
        if api_key:
            self._client_kwargs["api_key"] = api_key
        self._client_kwargs.update(openai_kwargs)

        # Lazy initialization - client created on first use
        # TODO: get rid of these attributes
        self._client = None
        self._instructor = None
        self._in_async_context = is_async

    @property
    def client(self):
        """Lazily initialized OpenAI client, switches between sync/async as needed."""
        if self._client is None or self._in_async_context != self.is_async:
            import openai

            if self._in_async_context:
                self._client = openai.AsyncOpenAI(**self._client_kwargs)
            else:
                self._client = openai.OpenAI(**self._client_kwargs)
            self.is_async = self._in_async_context
            self._instructor = None  # Reset instructor when switching
        return self._client

    @property
    def instructor_client(self):
        """Lazily initialized Instructor client, matches current client sync/async state."""
        # Access self.client to ensure it's initialized and in correct mode
        _ = self.client

        if self._instructor is None:
            self._instructor = instructor.from_openai(
                self._client,  # type: ignore
                mode=instructor.Mode.JSON,
            )
        return self._instructor

    def _set_async_mode(self, is_async: bool):
        """Context setter to switch between sync/async mode."""
        self._in_async_context = is_async

    @with_callbacks
    def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[type[T]] = None,
        max_retries: Optional[int] = 3,
        **kwargs,
    ) -> Union[ChatCompletion, T]:
        """Synchronous chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name, overrides default if provided
            temperature: Temperature, overrides default if provided
            response_format: Optional Pydantic model for structured output
            max_retries: Maximum retries for Instructor validation if using response_format
            **kwargs: Additional arguments passed to the API

        Returns:
            ChatCompletion object if response_format is None or instance of response_format
        """
        model = model or self.model or settings.model_name
        temperature = temperature or self.temperature

        # If response_format is provided, use instructor
        if response_format is not None:
            self._set_async_mode(False)
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_model": response_format,
                "max_retries": max_retries,
                **kwargs,
            }

            return self.instructor_client.chat.completions.create(**payload)

        # Otherwise, use normal OpenAI client
        self._set_async_mode(False)

        payload = {
            "model": model,
            "messages": messages,  # type: ignore
            "temperature": temperature,
            **kwargs,
        }

        return self.client.chat.completions.create(stream=False, **payload)  # type: ignore

    @with_callbacks
    async def achat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[type[T]] = None,
        max_retries: Optional[int] = 3,
        **kwargs,
    ) -> Union[ChatCompletion, T]:
        """Asynchronous chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name, overrides default if provided
            temperature: Temperature, overrides default if provided
            response_format: Optional Pydantic model for structured output
            max_retries: Maximum retries for Instructor validation if using response_format
            **kwargs: Additional arguments passed to the API

        Returns:
            ChatCompletion object if response_format is None or instance of response_format
        """
        model = model or self.model or settings.model_name
        temperature = temperature or self.temperature

        # If response_format is provided, use instructor
        if response_format is not None:
            self._set_async_mode(True)
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_model": response_format,
                "max_retries": max_retries,
                **kwargs,
            }

            return await self.instructor_client.chat.completions.create(**payload)

        # Otherwise, use normal OpenAI client
        self._set_async_mode(True)

        payload = {
            "model": model,
            "messages": messages,  # type: ignore
            "temperature": temperature,
            **kwargs,
        }

        return await self.client.chat.completions.create(stream=False, **payload)  # type: ignore

    @with_callbacks
    def generate(
        self,
        messages: list[ChatMessage],
        response_format: Optional[type[T]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = 3,
        **kwargs,
    ) -> Union[ChatCompletion, T]:
        """Synchronous generation with optional structured output.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            response_format: Optional Pydantic model for structured output
            model: Model name, overrides default if provided
            temperature: Temperature, overrides default if provided
            max_retries: Maximum retries for Instructor validation if using response_format
            **kwargs: Additional arguments passed to the API

        Returns:
            Instance of response_format if provided or ChatCompletion object
        """
        model = model or self.model or settings.model_name
        temperature = temperature or self.temperature

        # If response_format is provided, use instructor
        if response_format is not None:
            self._set_async_mode(False)
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_model": response_format,
                "max_retries": max_retries,
                **kwargs,
            }

            return self.instructor_client.chat.completions.create(**payload)

        # Otherwise, use normal OpenAI client for regular completion
        self._set_async_mode(False)

        payload = {
            "model": model,
            "messages": messages,  # type: ignore
            "temperature": temperature,
            **kwargs,
        }

        return self.client.chat.completions.create(stream=False, **payload)  # type: ignore

    @with_callbacks
    async def agenerate(
        self,
        messages: list[ChatMessage],
        response_format: Optional[type[T]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = 3,
        **kwargs,
    ) -> Union[ChatCompletion, T]:
        """Asynchronous generation with optional structured output.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            response_format: Optional Pydantic model for structured output
            model: Model name, overrides default if provided
            temperature: Temperature, overrides default if provided
            max_retries: Maximum retries for Instructor validation if using response_format
            **kwargs: Additional arguments passed to the API

        Returns:
            Instance of response_format if provided or ChatCompletion object
        """
        model = model or self.model or settings.model_name
        temperature = temperature or self.temperature

        # If response_format is provided, use instructor
        if response_format is not None:
            self._set_async_mode(True)
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_model": response_format,
                "max_retries": max_retries,
                **kwargs,
            }

            return await self.instructor_client.chat.completions.create(**payload)

        # Otherwise, use normal OpenAI client for regular completion
        self._set_async_mode(True)

        payload = {
            "model": model,
            "messages": messages,  # type: ignore
            "temperature": temperature,
            **kwargs,
        }

        return await self.client.chat.completions.create(stream=False, **payload)  # type: ignore

    async def stream(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[Union[str, dict]]:
        """Asynchronous streaming chat completion.

        Yields content chunks as strings, then a final dict with the complete response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name, overrides default if provided
            temperature: Temperature, overrides default if provided
            **kwargs: Additional arguments passed to the API

        Yields:
            str: Content chunks during streaming
            dict: Final complete response with choices, model, and usage info
        """
        model = model or self.model or settings.model_name
        temperature = temperature or self.temperature

        self._set_async_mode(True)
        assert model is not None, "model must be provided or set as default model"

        completion_stream = await self.client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,  # type: ignore
            temperature=temperature,
            stream=True,
            **kwargs,
        )

        content_chunks = []
        finish_reason = "stop"

        async for chunk in completion_stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                content_chunks.append(content)
                yield content
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        # Yield complete response object at the end
        full_content = "".join(content_chunks)
        yield {
            "choices": [
                {"message": {"content": full_content}, "finish_reason": finish_reason}
            ],
            "model": model,
            "usage": {"prompt_tokens": 0, "completion_tokens": len(content_chunks)},
        }

    # TODO: Reconsider callback-based logging approach - might be cleaner to implement logging directly
    def _pre_chat(self, *args, **kwargs):
        """Log the chat request before execution."""
        payload = {**kwargs, **dict(enumerate(args))}
        truncated_payload = truncate_long_strings(payload)
        logger.debug(
            f"Chat request:\n{json.dumps(truncated_payload, indent=2, default=str)}"
        )

    def _post_chat(self, result, duration, *args, **kwargs):
        """Log the chat response after execution."""
        try:
            result_dump = (
                result.model_dump() if hasattr(result, "model_dump") else str(result)
            )
        except Exception:
            result_dump = str(result)
        result_dump = truncate_long_strings(result_dump)
        logger.debug(
            f"Chat response in {duration:.4f}s:\n{json.dumps(result_dump, indent=2, default=str)}"
        )
        return result

    _pre_achat = _pre_chat
    _post_achat = _post_chat
    _pre_generate = _pre_chat
    _post_generate = _post_chat
    _pre_agenerate = _pre_chat
    _post_agenerate = _post_chat
