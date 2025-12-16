"""Unified LLM Client

A single-class LLM client that handles OpenAI, Ollama, and Mistral providers
with support for sync/async, streaming, and structured outputs.

Features:
- Supports standard chat completions and structured outputs
- Async and streaming support
- Built-in logging
- Factory methods for different providers (from_openai, from_mistral)
"""

import json
import logging
import time
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)

import instructor
from pydantic import BaseModel

from ..utils.logging import truncate_long_strings

logger = logging.getLogger(__name__)

# Type definitions
ChatMessage = dict[str, Any]
T = TypeVar("T", bound=BaseModel)
LLMProvider = Union[Literal["openai", "ollama", "mistral"], str]


class StreamResponse(TypedDict):
    """Final response structure for streaming completions."""

    choices: list[dict[str, Any]]
    model: str
    usage: dict[str, int]


class LLMClient:
    """Unified LLM client supporting OpenAI, Ollama, and Mistral providers.

    Use factory methods for provider-specific initialization:
        - LLMClient.from_openai() for OpenAI (default)
        - LLMClient.from_openai(provider="ollama") for Ollama
        - LLMClient.from_mistral() for Mistral AI

    The default constructor delegates to from_openai() for backward compatibility.

    Args:
        model: Default model name for all requests
        temperature: Default temperature for all requests
        base_url: Base URL for API, use for Ollama, leave None for OpenAI
        api_key: API key, None uses environment variable
        log_level: Logging level
        **kwargs: Additional kwargs passed to client
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        log_level: str = "INFO",
        **kwargs,
    ):
        """Create an LLMClient for OpenAI or Ollama (default factory).

        For Mistral, use LLMClient.from_mistral() instead.

        Args:
            model: Model name, e.g., "gpt-4", "llama3.2:1b"
            temperature: Default temperature for all requests
            base_url: Base URL, use for Ollama, leave None for OpenAI
            api_key: API key, None uses environment variable
            log_level: Logging level
            **kwargs: Additional kwargs passed to client
        """
        # Delegate to from_openai factory
        instance = self.from_openai(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            log_level=log_level,
            **kwargs,
        )
        # Copy all attributes from the factory-created instance
        self.__dict__.update(instance.__dict__)

    def _init_from_factory(
        self,
        model: str,
        sync_client: Any,  # type: ignore
        async_client: Any,  # type: ignore
        sync_instructor: Any,  # type: ignore
        async_instructor: Any,  # type: ignore
        temperature: float = 0.7,
        base_url: Optional[str] = None,
        log_level: str = "INFO",
        provider: LLMProvider = "openai",
    ):
        """Internal initialization called by factory methods, not for direct use."""
        # Set up logging
        logger.setLevel(log_level.upper())
        logger.debug(f"{self.__class__.__name__} initialized")

        # Store defaults
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.provider = provider

        # Store pre-configured clients
        self.sync_client = sync_client
        self.async_client = async_client
        self.sync_instructor = sync_instructor
        self.async_instructor = async_instructor

    def _log_request(self, method_name: str, **params):
        """Log request parameters."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        truncated = truncate_long_strings(params)
        logger.debug(
            f"{method_name} request:\n{json.dumps(truncated, indent=2, default=str)}"
        )

    def _log_response(self, method_name: str, result: Any, duration: float):
        """Log response and timing."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        try:
            result_dump = (
                result.model_dump() if hasattr(result, "model_dump") else str(result)
            )
        except Exception:
            result_dump = str(result)
        truncated = truncate_long_strings(result_dump)
        logger.debug(
            f"{method_name} response in {duration:.4f}s:\n{json.dumps(truncated, indent=2, default=str)}"
        )

    @classmethod
    def from_openai(
        cls,
        model: str,
        temperature: float = 0.7,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        log_level: str = "INFO",
        provider: LLMProvider = "openai",
        **openai_kwargs,
    ) -> "LLMClient":
        """Create an LLMClient for OpenAI or Ollama.

        Args:
            model: Default model for all requests
            temperature: Default temperature for all requests
            base_url: Base URL, for Ollama /v1 is auto-appended if missing
            api_key: API key, leave None to use environment variable or "ollama" for Ollama
            log_level: Logging level, default "INFO"
            provider: LLMProvider name, "openai" or "ollama", default "openai"
            **openai_kwargs: Additional kwargs passed to OpenAI client

        Returns:
            LLMClient instance configured for OpenAI/Ollama
        """
        import openai

        # Normalize base_url for Ollama (ensure /v1 suffix)
        normalized_base_url = base_url
        if base_url and provider == "ollama" and not base_url.endswith("/v1"):
            logger.debug("Appending /v1 to Ollama base_url.")
            normalized_base_url = base_url.rstrip("/") + "/v1"

        # Build client kwargs
        kwargs = openai_kwargs.copy()
        if normalized_base_url:
            kwargs["base_url"] = normalized_base_url
        if api_key:
            kwargs["api_key"] = api_key
        elif provider == "ollama":
            kwargs["api_key"] = "ollama"  # Ollama accepts any value

        # Create both sync and async clients
        sync_client = openai.OpenAI(**kwargs)
        async_client = openai.AsyncOpenAI(**kwargs)

        # Wrap with instructor for structured outputs
        sync_instructor = instructor.from_openai(sync_client, mode=instructor.Mode.JSON)
        async_instructor = instructor.from_openai(
            async_client, mode=instructor.Mode.JSON
        )

        # Create instance and initialize
        instance = cls.__new__(cls)
        instance._init_from_factory(
            model=model,
            sync_client=sync_client,
            async_client=async_client,
            sync_instructor=sync_instructor,
            async_instructor=async_instructor,
            temperature=temperature,
            base_url=normalized_base_url,
            log_level=log_level,
            provider=provider,
        )
        return instance

    @classmethod
    def from_mistral(
        cls,
        model: str,
        server_url: str,
        api_key: str,
        temperature: float = 0.7,
        log_level: str = "INFO",
        **mistral_kwargs,
    ) -> "LLMClient":
        """Create an LLMClient for Mistral AI.

        Args:
            model: Model name, e.g., "mistral-medium-latest"
            server_url: Mistral API server URL
            api_key: Mistral API key
            temperature: Default temperature for all requests
            log_level: Logging level
            **mistral_kwargs: Additional kwargs passed to Mistral client

        Returns:
            LLMClient instance configured for Mistral
        """
        from mistralai import Mistral  # type: ignore

        # Build client kwargs
        kwargs = mistral_kwargs.copy()
        kwargs["server_url"] = server_url
        kwargs["api_key"] = api_key

        # Create a single Mistral client (handles both sync and async)
        mistral_client = Mistral(**kwargs)

        # Mistral uses the same client for both sync and async
        # sync uses client.chat.complete(), async uses client.chat.complete_async()
        sync_client = async_client = mistral_client

        # Wrap with instructor for structured outputs
        # Use instructor.from_mistral for sync
        sync_instructor = instructor.from_mistral(  # type: ignore
            sync_client, mode=instructor.Mode.MISTRAL_TOOLS, use_async=False
        )
        # Use instructor.from_mistral for async
        async_instructor = instructor.from_mistral(  # type: ignore
            async_client, mode=instructor.Mode.MISTRAL_TOOLS, use_async=True
        )

        # Create instance and initialize
        instance = cls.__new__(cls)
        instance._init_from_factory(
            model=model,
            sync_client=sync_client,
            async_client=async_client,
            sync_instructor=sync_instructor,
            async_instructor=async_instructor,
            temperature=temperature,
            base_url=server_url,  # Map server_url to base_url internally
            log_level=log_level,
            provider="mistral",
        )
        return instance

    def _complete(self, client: Any, **payload) -> Any:  # type: ignore
        """Call appropriate completion method based on client type."""
        if self.provider == "mistral":
            return client.chat.complete(**payload)  # type: ignore
        else:  # OpenAI/Ollama
            return client.chat.completions.create(stream=False, **payload)  # type: ignore

    def _stream(self, client: Any, **payload):  # type: ignore
        """Streaming method based on client type."""
        if self.provider == "mistral":
            return client.chat.stream(**payload)  # type: ignore
        else:  # OpenAI/Ollama
            return client.chat.completions.create(stream=True, **payload)  # type: ignore

    async def _acomplete(self, client: Any, **payload) -> Any:  # type: ignore
        """Async completion method based on client type."""
        if self.provider == "mistral":
            return await client.chat.complete_async(**payload)  # type: ignore
        else:  # OpenAI/Ollama
            return await client.chat.completions.create(stream=False, **payload)  # type: ignore

    async def _astream(self, client: Any, **payload):  # type: ignore
        """Async streaming method based on client type."""
        if self.provider == "mistral":
            return await client.chat.stream_async(**payload)  # type: ignore
        else:  # OpenAI/Ollama
            return await client.chat.completions.create(stream=True, **payload)  # type: ignore

    def _build_and_validate_payload(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> dict:
        """Build request payload with defaults and validate parameters."""
        # Validate stream + response_format combination
        if kwargs.get("stream") and kwargs.get("response_format") is not None:
            raise ValueError("Streaming is not supported with structured output")

        # Extract and apply defaults
        model = kwargs.pop("model", None) or self.model
        temperature = kwargs.pop("temperature", None)
        if temperature is None:
            temperature = self.temperature

        # Remove params that shouldn't go to API
        kwargs.pop("stream", None)
        kwargs.pop("response_format", None)

        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

    def _build_stream_response(
        self, content_chunks: list[str], finish_reason: str, model: str
    ) -> StreamResponse:
        """Build final streaming response dict."""
        full_content = "".join(content_chunks)
        return {
            "choices": [
                {"message": {"content": full_content}, "finish_reason": finish_reason}
            ],
            "model": model,
            "usage": {"prompt_tokens": 0, "completion_tokens": len(content_chunks)},
        }

    def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[type[T]] = None,
        max_retries: int = 3,
        stream: bool = False,
        **kwargs,
    ) -> Union[Any, T, Iterator[Union[str, StreamResponse]]]:  # type: ignore
        """Synchronous chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name, overrides instance default if provided
            temperature: Temperature, overrides instance default if provided
            response_format: Optional Pydantic model for structured output
            max_retries: Maximum retries for Instructor validation if using response_format
            stream: Enable streaming mode, yields content chunks then final response
            **kwargs: Additional arguments passed to the API

        Returns:
            ChatCompletion object, instance of response_format, or Iterator if stream=True
        """
        payload = self._build_and_validate_payload(
            messages,
            model=model,
            temperature=temperature,
            stream=stream,
            response_format=response_format,
            **kwargs,
        )

        # If response_format is provided, use instructor
        if response_format is not None:
            self._log_request(
                "chat",
                messages=messages,
                model=model,
                response_format=response_format.__name__,
            )
            start_time = time.time()
            result = self.sync_instructor.chat.completions.create(  # type: ignore
                response_model=response_format,
                max_retries=max_retries,  # type: ignore
                **payload,
            )
            self._log_response("chat", result, time.time() - start_time)
            return result

        # Handle streaming (no logging for streaming - doesn't make sense)
        if stream:
            return self._chat_stream(payload)

        # Normal completion
        self._log_request(
            "chat", messages=messages, model=model, temperature=temperature
        )
        start_time = time.time()
        result = self._complete(self.sync_client, **payload)
        self._log_response("chat", result, time.time() - start_time)
        return result

    def _chat_stream(self, payload: dict) -> Iterator[Union[str, StreamResponse]]:
        """Internal method for streaming chat completions."""
        completion_stream = self._stream(self.sync_client, **payload)
        content_chunks = []
        finish_reason = "stop"

        for chunk in completion_stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                content_chunks.append(content)
                yield content
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        yield self._build_stream_response(
            content_chunks, finish_reason, payload["model"]
        )

    async def achat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[type[T]] = None,
        max_retries: int = 3,
        stream: bool = False,
        **kwargs,
    ) -> Union[Any, T, AsyncIterator[Union[str, StreamResponse]]]:  # type: ignore
        """Asynchronous chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name, overrides instance default if provided
            temperature: Temperature, overrides instance default if provided
            response_format: Optional Pydantic model for structured output
            max_retries: Maximum retries for Instructor validation if using response_format
            stream: Enable streaming mode, yields content chunks then final response
            **kwargs: Additional arguments passed to the API

        Returns:
            ChatCompletion object, instance of response_format, or AsyncIterator if stream=True
        """
        payload = self._build_and_validate_payload(
            messages,
            model=model,
            temperature=temperature,
            stream=stream,
            response_format=response_format,
            **kwargs,
        )

        # If response_format is provided, use instructor
        if response_format is not None:
            self._log_request(
                "achat",
                messages=messages,
                model=model,
                response_format=response_format.__name__,
            )
            start_time = time.time()
            result = await self.async_instructor.chat.completions.create(  # type: ignore
                response_model=response_format,
                max_retries=max_retries,  # type: ignore
                **payload,
            )
            self._log_response("achat", result, time.time() - start_time)
            return result

        # Handle streaming
        if stream:
            return self._achat_stream(payload)

        # Normal completion
        self._log_request(
            "achat", messages=messages, model=model, temperature=temperature
        )
        start_time = time.time()
        result = await self._acomplete(self.async_client, **payload)
        self._log_response("achat", result, time.time() - start_time)
        return result

    async def _achat_stream(
        self, payload: dict
    ) -> AsyncIterator[Union[str, StreamResponse]]:
        """Internal method for async streaming chat completions."""
        completion_stream = await self._astream(self.async_client, **payload)
        content_chunks = []
        finish_reason = "stop"

        async for chunk in completion_stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                content_chunks.append(content)
                yield content
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        yield self._build_stream_response(
            content_chunks, finish_reason, payload["model"]
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[type[T]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
        stream: bool = False,
        **kwargs,
    ) -> Union[Any, T, Iterator[Union[str, StreamResponse]]]:  # type: ignore
        """Synchronous generation with optional structured output.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system message to prepend
            response_format: Optional Pydantic model for structured output
            model: Model name, overrides instance default if provided
            temperature: Temperature, overrides instance default if provided
            max_retries: Maximum retries for Instructor validation if using response_format
            stream: Enable streaming mode, yields content chunks then final response
            **kwargs: Additional arguments passed to the API

        Returns:
            Instance of response_format, ChatCompletion object, or Iterator if stream=True
        """
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
            max_retries=max_retries,
            stream=stream,
            **kwargs,
        )

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[type[T]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
        stream: bool = False,
        **kwargs,
    ) -> Union[Any, T, AsyncIterator[Union[str, StreamResponse]]]:  # type: ignore
        """Asynchronous generation with optional structured output.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system message to prepend
            response_format: Optional Pydantic model for structured output
            model: Model name, overrides instance default if provided
            temperature: Temperature, overrides instance default if provided
            max_retries: Maximum retries for Instructor validation if using response_format
            stream: Enable streaming mode, yields content chunks then final response
            **kwargs: Additional arguments passed to the API

        Returns:
            Instance of response_format, ChatCompletion object, or AsyncIterator if stream=True
        """
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.achat(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
            max_retries=max_retries,
            stream=stream,
            **kwargs,
        )
