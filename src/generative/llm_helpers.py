"""Simple helper functions for working with LLMs.

Client Creation:
    create_ollama_client(): OpenAI client for Ollama, sync or async
    create_mistral_client(): OpenAI-compatible client for MistralAI, sync or async

Completion Functions:
    completion(): Simple completion
    async_completion(): Async version of completion()
    chat_completion(): Standard chat completion
    async_chat_completion(): Async version of chat_completion()

Streaming Functions:
    stream_completion(): Streams text chunks, yields final ChatCompletion
    async_stream_completion(): Async version of stream_completion()

Mock Functions:
    mock_chat_completion(): Mock chat response for testing
    mock_stream_completion(): Mock streaming response for testing
"""

import logging
import os
import random
import time
from datetime import datetime
from typing import Any, AsyncIterator, Iterator, Optional, TypeVar, cast, overload

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)
from pydantic import BaseModel
from typing_extensions import Literal

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@overload
def create_ollama_client(
    base_url: str = "http://localhost:11434",
    async_client: Literal[False] = False,
    **kwargs,
) -> OpenAI: ...


@overload
def create_ollama_client(
    base_url: str = "http://localhost:11434",
    async_client: Literal[True] = ...,
    **kwargs,
) -> AsyncOpenAI: ...


def create_ollama_client(
    base_url: str = "http://localhost:11434",
    async_client: bool = False,
    **kwargs,
) -> OpenAI | AsyncOpenAI:
    """Create an OpenAI client configured for Ollama.

    Args:
        base_url: Ollama server URL
        async_client: If True, return AsyncOpenAI client
        **kwargs: Additional OpenAI client arguments

    Returns:
        OpenAI or AsyncOpenAI client configured for Ollama

    Example:
        >>> client = create_ollama_client()
        >>> async_client = create_ollama_client(async_client=True)
    """
    normalized_url = f"{base_url.rstrip('/')}/v1"

    if async_client:
        return AsyncOpenAI(
            api_key="ollama",
            base_url=normalized_url,
            **kwargs,
        )
    else:
        return OpenAI(
            api_key="ollama",
            base_url=normalized_url,
            **kwargs,
        )


@overload
def create_mistral_client(
    api_key: Optional[str] = None,
    server_url: str = "https://api.mistral.ai",
    async_client: Literal[False] = False,
    **kwargs,
) -> OpenAI: ...


@overload
def create_mistral_client(
    api_key: Optional[str] = None,
    server_url: str = "https://api.mistral.ai",
    async_client: Literal[True] = ...,
    **kwargs,
) -> AsyncOpenAI: ...


def create_mistral_client(
    api_key: Optional[str] = None,
    server_url: str = "https://api.mistral.ai",
    async_client: bool = False,
    **kwargs,
) -> OpenAI | AsyncOpenAI:
    """Creates an OpenAI-compatible client for MistralAI.

    Args:
        api_key: Mistral API key, None reads from MISTRAL_API_KEY env var
        server_url: Mistral API server URL
        async_client: If True, return async-compatible client
        **kwargs: Additional Mistral client arguments

    Returns:
        OpenAI or AsyncOpenAI client configured for Mistral

    Example:
        >>> client = create_mistral_client(api_key="your-key")
        >>> async_client = create_mistral_client(api_key="your-key", async_client=True)
    """
    from mistralai import Mistral

    resolved_api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Mistral API key must be provided or set in MISTRAL_API_KEY environment variable"
        )

    mistral_client = Mistral(api_key=resolved_api_key, server_url=server_url, **kwargs)

    # Create adapter class that maps OpenAI API to Mistral API
    class CompletionsAdapter:
        def __init__(self, mistral_client, is_async: bool):
            self._mistral = mistral_client
            self._is_async = is_async

        def create(self, **kwargs):
            # Handle streaming
            stream = kwargs.pop("stream", False)

            if stream:
                return (
                    self._mistral.chat.stream_async(**kwargs)
                    if self._is_async
                    else self._mistral.chat.stream(**kwargs)
                )
            else:
                return (
                    self._mistral.chat.complete_async(**kwargs)
                    if self._is_async
                    else self._mistral.chat.complete(**kwargs)
                )

    # Inject OpenAI-compatible completions interface
    mistral_client.chat.completions = CompletionsAdapter(mistral_client, async_client)  # type: ignore

    # Return as OpenAI client type for type checking
    if async_client:
        return cast(AsyncOpenAI, mistral_client)
    else:
        return cast(OpenAI, mistral_client)


def completion(
    client: OpenAI,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Simple prompt completion.

    Args:
        client: OpenAI client
        model: Model name
        prompt: User prompt
        system_prompt: Optional system message
        temperature: Sampling temperature
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        str if no response_model, otherwise instance of response_model

    Example:
        >>> text = completion(client, "llama3.2:1b", "What is 2+2?")
        >>> answer = completion(client, "llama3.2:1b", "What is 2+2?", response_model=Answer)
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    _log_chat_request(model=model, temperature=temperature, messages=messages)

    msg_params = cast(list[ChatCompletionMessageParam], messages)

    start_time = time.time()
    if response_model:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        result = response_model.model_validate_json(response.choices[0].message.content)
        duration = time.time() - start_time
        _log_chat_response(
            model=model, content_len=len(str(result)), duration_s=duration
        )
        return result
    else:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        duration = time.time() - start_time
        _log_chat_response(model=model, content_len=len(content), duration_s=duration)
        return content


async def async_completion(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Async version of completion.

    Args:
        client: AsyncOpenAI client
        model: Model name
        prompt: User prompt
        system_prompt: Optional system message
        temperature: Sampling temperature
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        str if no response_model, otherwise instance of response_model
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    _log_chat_request(model=model, temperature=temperature, messages=messages)

    msg_params = cast(list[ChatCompletionMessageParam], messages)

    start_time = time.time()
    if response_model:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        result = response_model.model_validate_json(response.choices[0].message.content)
        duration = time.time() - start_time
        _log_chat_response(
            model=model, content_len=len(str(result)), duration_s=duration
        )
        return result
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        duration = time.time() - start_time
        _log_chat_response(model=model, content_len=len(content), duration_s=duration)
        return content


def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Chat completion with message history.

    Args:
        client: OpenAI client
        model: Model name
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        str if no response_model, otherwise instance of response_model

    Example:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> response = chat_completion(client, "llama3.2:1b", messages)
    """
    _log_chat_request(model=model, temperature=temperature, messages=messages)

    msg_params = cast(list[ChatCompletionMessageParam], messages)

    start_time = time.time()
    if response_model:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        result = response_model.model_validate_json(response.choices[0].message.content)
        duration = time.time() - start_time
        _log_chat_response(
            model=model, content_len=len(str(result)), duration_s=duration
        )
        return result
    else:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        duration = time.time() - start_time
        _log_chat_response(model=model, content_len=len(content), duration_s=duration)
        return content


async def async_chat_completion(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Async version of chat_completion.

    Args:
        client: AsyncOpenAI client
        model: Model name
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        str if no response_model, otherwise instance of response_model
    """
    _log_chat_request(model=model, temperature=temperature, messages=messages)

    msg_params = cast(list[ChatCompletionMessageParam], messages)

    start_time = time.time()
    if response_model:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        result = response_model.model_validate_json(response.choices[0].message.content)
        duration = time.time() - start_time
        _log_chat_response(
            model=model, content_len=len(str(result)), duration_s=duration
        )
        return result
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        duration = time.time() - start_time
        _log_chat_response(model=model, content_len=len(content), duration_s=duration)
        return content


def stream_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    **kwargs,
) -> Iterator[str | ChatCompletion]:
    """Stream text chunks from completion, then yield final ChatCompletion.

    Args:
        client: OpenAI client
        model: Model name
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        **kwargs: Additional arguments for chat.completions.create

    Yields:
        Text chunks as they arrive, then a final ChatCompletion object

    Example:
        >>> for chunk in stream_completion(client, "llama3.2:1b", [{"role": "user", "content": "Hi"}]):
        ...     print(chunk, end="", flush=True)
    """
    _log_chat_request(model=model, temperature=temperature, messages=messages)
    start_time = time.time()

    msg_params = cast(list[ChatCompletionMessageParam], messages)
    stream = client.chat.completions.create(
        model=model,
        messages=msg_params,
        temperature=temperature,
        stream=True,
        **kwargs,
    )

    duration = time.time() - start_time
    logger.debug(
        "LLM stream started in %.4f s, model=%s temperature=%s",
        duration,
        model,
        temperature,
    )

    content_chunks: list[str] = []
    finish_reason = "stop"

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            content_chunks.append(piece)
            yield piece
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    full_content = "".join(content_chunks)
    yield ChatCompletion(
        id=f"chatcmpl-{model}-{datetime.now().timestamp()}",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=full_content),
                finish_reason=finish_reason,
            )
        ],
    )


async def async_stream_completion(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    **kwargs,
) -> AsyncIterator[str | ChatCompletion]:
    """Async version of stream_completion.

    Args:
        client: AsyncOpenAI client
        model: Model name
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        **kwargs: Additional arguments for chat.completions.create

    Yields:
        Text chunks as they arrive, then a final ChatCompletion object

    Example:
        >>> async for chunk in async_stream_completion(client, "llama3.2:1b", [{"role": "user", "content": "Hi"}]):
        ...     print(chunk, end="", flush=True)
    """
    _log_chat_request(model=model, temperature=temperature, messages=messages)
    start_time = time.time()

    msg_params = cast(list[ChatCompletionMessageParam], messages)
    stream = await client.chat.completions.create(
        model=model,
        messages=msg_params,
        temperature=temperature,
        stream=True,
        **kwargs,
    )

    duration = time.time() - start_time
    logger.debug(
        "LLM stream started in %.4f s, model=%s temperature=%s",
        duration,
        model,
        temperature,
    )

    content_chunks: list[str] = []
    finish_reason = "stop"

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            content_chunks.append(piece)
            yield piece
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    full_content = "".join(content_chunks)
    yield ChatCompletion(
        id=f"chatcmpl-{model}-{datetime.now().timestamp()}",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=full_content),
                finish_reason=finish_reason,
            )
        ],
    )


async def mock_chat_completion(
    *,
    messages: list[dict[str, Any]],
    model: str = "mock-llm",
    temperature: float = 0.7,
    response_type: Optional[int] = None,
    seed: Optional[int] = None,
) -> ChatCompletion:
    """Mock chat completion.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        temperature: Sampling temperature
        response_type: Optional index to select specific mock response
        seed: Optional random seed for reproducibility

    Returns:
        ChatCompletion object with mock content

    Example:
        >>> response = await mock_chat_completion(messages=[{"role": "user", "content": "Hello"}])
        >>> print(response.choices[0].message.content)
    """
    if seed is not None:
        random.seed(seed)

    user_message = messages[-1].get("content") if messages else "Hello"
    mock_responses = [
        user_message,
        "This is a mock response.",
        f"This is a mock response from {model} with temperature {temperature}.",
        f"Let me think about '{user_message}'...",
    ]

    if response_type is None:
        content = random.choice(mock_responses)
    else:
        content = mock_responses[abs(response_type) % len(mock_responses)]

    return _create_mock_response(content=content, model=model)


async def mock_stream_completion(
    *,
    messages: list[dict[str, Any]],
    model: str = "mock-llm",
    temperature: float = 0.7,
) -> AsyncIterator[str | ChatCompletion]:
    """Mock streaming generator compatible with stream_completion.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        temperature: Sampling temperature

    Yields:
        Text chunks, then a final ChatCompletion object

    Example:
        >>> async for chunk in mock_stream_completion(messages=[{"role": "user", "content": "Hi"}]):
        ...     print(chunk, end="", flush=True)
    """
    user_message = messages[-1].get("content") if messages else "Hello"
    mock_response = (
        f"This is a mock streaming response about '{user_message}' from {model}."
    )

    for word in mock_response.split():
        yield f"{word} "

    yield ChatCompletion(
        id=f"mock-{model}-{datetime.now().timestamp()}",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=mock_response),
                finish_reason="stop",
            )
        ],
    )


def _build_response_format(response_model: type[BaseModel]) -> ResponseFormatJSONSchema:
    """Build OpenAI response_format for structured output."""
    return ResponseFormatJSONSchema(
        type="json_schema",
        json_schema=JSONSchema(
            name=response_model.__name__,
            schema=response_model.model_json_schema(),
            strict=True,
        ),
    )


def _log_chat_request(
    *, model: str, temperature: float, messages: list[dict[str, Any]]
) -> None:
    """Log chat request details."""
    logger.debug(
        "LLM request: model=%s temperature=%s messages=%s",
        model,
        temperature,
        len(messages),
    )


def _log_chat_response(*, model: str, content_len: int, duration_s: float) -> None:
    """Log chat response details."""
    logger.debug(
        "LLM response: model=%s content_len=%s duration_s=%.4f",
        model,
        content_len,
        duration_s,
    )


def _create_mock_response(*, content: str, model: str) -> ChatCompletion:
    """Create a mock ChatCompletion response."""
    return ChatCompletion(
        id="mock_id",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )
