"""Unified LLM Client

A single-class LLM client that handles OpenAI, Ollama, and Mistral providers
with support for sync/async, streaming, and structured outputs.

Features:
- Supports standard chat completions and structured outputs
- Async and streaming support
- Built-in logging with callbacks
- Factory methods for different providers (from_openai, from_mistral)
"""

import json
import logging
from typing import Any, AsyncIterator, Optional, TypeVar, Union

import instructor
from pydantic import BaseModel

from ..utils.callbacks import CallbackMeta, with_callbacks
from ..utils.logging import truncate_long_strings

logger = logging.getLogger(__name__)

# Type definitions
ChatMessage = dict[str, Any]
T = TypeVar("T", bound=BaseModel)


class LLMClient(metaclass=CallbackMeta):
    """Unified LLM client supporting OpenAI, Ollama, and Mistral providers.

    Use factory methods for provider-specific initialization:
        - LLMClient.from_openai() for OpenAI and Ollama (default)
        - LLMClient.from_mistral() for Mistral AI

    The default constructor delegates to from_openai() for backward compatibility.

    Args:
        model: Default model name for all requests
        temperature: Default temperature for all requests
        base_url: Base URL for API (use for Ollama, leave None for OpenAI)
        api_key: API key (None uses environment variable)
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
            model: Model name (e.g., "gpt-4", "llama3.2:1b")
            temperature: Default temperature for all requests
            base_url: Base URL (use for Ollama, leave None for OpenAI)
            api_key: API key (None uses environment variable)
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
    ):
        """Internal initialization called by factory methods, not for direct use."""
        # Set up logging
        logger.setLevel(log_level.upper())
        logger.debug(f"{self.__class__.__name__} initialized")

        # Store defaults
        self.model = model
        self.temperature = temperature
        self.base_url = base_url

        # Store pre-configured clients
        self.sync_client = sync_client
        self.async_client = async_client
        self.sync_instructor = sync_instructor
        self.async_instructor = async_instructor

    @classmethod
    def from_openai(
        cls,
        model: str,
        temperature: float = 0.7,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        log_level: str = "INFO",
        **openai_kwargs,
    ) -> "LLMClient":
        """Create an LLMClient for OpenAI or Ollama.

        Args:
            model: Default model for all requests
            temperature: Default temperature for all requests
            base_url: Base URL
            api_key: API key, leave None to use environment variable
            log_level: Logging level, default "INFO"
            **openai_kwargs: Additional kwargs passed to OpenAI client

        Returns:
            LLMClient instance configured for OpenAI/Ollama
        """
        import openai  # Import only when needed

        # Build client kwargs
        kwargs = openai_kwargs.copy()
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

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
            base_url=base_url,
            log_level=log_level,
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
            model: Model name (e.g., "mistral-medium-latest")
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
        )
        return instance

    def _complete(self, client: Any, **payload) -> Any:  # type: ignore
        """Call appropriate completion method based on client type."""
        if hasattr(client.chat, "complete"):  # Mistral
            return client.chat.complete(**payload)  # type: ignore
        else:  # OpenAI
            return client.chat.completions.create(stream=False, **payload)  # type: ignore

    async def _acomplete(self, client: Any, **payload) -> Any:  # type: ignore
        """Async completion method based on client type."""
        if hasattr(client.chat, "complete_async"):  # Mistral
            return await client.chat.complete_async(**payload)  # type: ignore
        else:  # OpenAI
            return await client.chat.completions.create(stream=False, **payload)  # type: ignore

    async def _astream(self, client: Any, **payload):  # type: ignore
        """Async streaming method based on client type."""
        if hasattr(client.chat, "stream_async"):  # Mistral
            return await client.chat.stream_async(**payload)  # type: ignore
        else:  # OpenAI
            return await client.chat.completions.create(stream=True, **payload)  # type: ignore

    @with_callbacks
    def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[type[T]] = None,
        max_retries: Optional[int] = 3,
        **kwargs,
    ) -> Union[Any, T]:  # type: ignore
        """Synchronous chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name, overrides instance default if provided
            temperature: Temperature, overrides instance default if provided
            response_format: Optional Pydantic model for structured output
            max_retries: Maximum retries for Instructor validation if using response_format
            **kwargs: Additional arguments passed to the API

        Returns:
            ChatCompletion object if response_format is None or instance of response_format
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

        # If response_format is provided, use instructor
        if response_format is not None:
            # Instructor wraps both OpenAI and Mistral with unified API
            return self.sync_instructor.chat.completions.create(  # type: ignore
                response_model=response_format,
                max_retries=max_retries,  # type: ignore
                **payload,
            )

        # Otherwise, use normal client (handles both OpenAI and Mistral)
        return self._complete(self.sync_client, **payload)

    @with_callbacks
    async def achat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[type[T]] = None,
        max_retries: Optional[int] = 3,
        **kwargs,
    ) -> Union[Any, T]:  # type: ignore
        """Asynchronous chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name, overrides instance default if provided
            temperature: Temperature, overrides instance default if provided
            response_format: Optional Pydantic model for structured output
            max_retries: Maximum retries for Instructor validation if using response_format
            **kwargs: Additional arguments passed to the API

        Returns:
            ChatCompletion object if response_format is None or instance of response_format
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

        # If response_format is provided, use instructor
        if response_format is not None:
            # Instructor wraps both OpenAI and Mistral with unified API
            return await self.async_instructor.chat.completions.create(  # type: ignore
                response_model=response_format,
                max_retries=max_retries,  # type: ignore
                **payload,
            )

        # Otherwise, use normal client (handles both OpenAI and Mistral)
        return await self._acomplete(self.async_client, **payload)

    @with_callbacks
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[type[T]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = 3,
        **kwargs,
    ) -> Union[Any, T]:  # type: ignore
        """Synchronous generation with optional structured output.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system message to prepend
            response_format: Optional Pydantic model for structured output
            model: Model name, overrides instance default if provided
            temperature: Temperature, overrides instance default if provided
            max_retries: Maximum retries for Instructor validation if using response_format
            **kwargs: Additional arguments passed to the API

        Returns:
            Instance of response_format if provided or ChatCompletion object
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
            **kwargs,
        )

    @with_callbacks
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[type[T]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = 3,
        **kwargs,
    ) -> Union[Any, T]:  # type: ignore
        """Asynchronous generation with optional structured output.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system message to prepend
            response_format: Optional Pydantic model for structured output
            model: Model name, overrides instance default if provided
            temperature: Temperature, overrides instance default if provided
            max_retries: Maximum retries for Instructor validation if using response_format
            **kwargs: Additional arguments passed to the API

        Returns:
            Instance of response_format if provided or ChatCompletion object
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
            **kwargs,
        )

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
            model: Model name, overrides instance default if provided
            temperature: Temperature, overrides instance default if provided
            **kwargs: Additional arguments passed to the API

        Yields:
            str: Content chunks during streaming
            dict: Final complete response with choices, model, and usage info
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

        # Get streaming response (handles both OpenAI and Mistral)
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
