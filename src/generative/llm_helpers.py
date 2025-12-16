"""Simple helper functions for working with LLMs.

No complex wrappers - just practical utilities for common patterns.
"""

from typing import Any, AsyncIterator, Iterator, Optional, TypeVar, cast

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _build_response_format(response_model: type[BaseModel]) -> ResponseFormatJSONSchema:
    """Internal helper to build OpenAI response_format."""
    return ResponseFormatJSONSchema(
        type="json_schema",
        json_schema=JSONSchema(
            name=response_model.__name__,
            schema=response_model.model_json_schema(),
            strict=True,
        ),
    )


def create_ollama_client(
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> OpenAI:
    """Create an OpenAI client configured for Ollama.

    Args:
        model: Optional default model (can override per-request)
        base_url: Ollama server URL (default: http://localhost:11434)
        **kwargs: Additional OpenAI client arguments

    Returns:
        OpenAI client configured for Ollama

    Example:
        >>> client = create_ollama_client()
        >>> response = client.chat.completions.create(
        ...     model="llama3.2:1b",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """
    return OpenAI(
        api_key="ollama",  # Ollama accepts any value
        base_url=f"{base_url.rstrip('/')}/v1",
        **kwargs,
    )


def create_async_ollama_client(
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> AsyncOpenAI:
    """Create an async OpenAI client configured for Ollama.

    Args:
        model: Optional default model (can override per-request)
        base_url: Ollama server URL (default: http://localhost:11434)
        **kwargs: Additional AsyncOpenAI client arguments

    Returns:
        AsyncOpenAI client configured for Ollama

    Example:
        >>> client = create_async_ollama_client()
        >>> response = await client.chat.completions.create(
        ...     model="llama3.2:1b",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """
    return AsyncOpenAI(
        api_key="ollama",
        base_url=f"{base_url.rstrip('/')}/v1",
        **kwargs,
    )


def completion(
    client: OpenAI,
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Simple prompt -> completion (text or structured).

    Args:
        client: OpenAI client
        prompt: User prompt
        model: Model name
        system_prompt: Optional system message
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        Text response (str) or instance of response_model if provided

    Example:
        >>> # Text response
        >>> client = create_ollama_client()
        >>> text = completion(client, "What is 2+2?", model="llama3.2:1b")
        >>>
        >>> # Structured response
        >>> from pydantic import BaseModel
        >>> class Answer(BaseModel):
        ...     result: int
        >>> answer = completion(
        ...     client, "What is 2+2?",
        ...     model="llama3.2:1b",
        ...     response_model=Answer
        ... )
        >>> print(answer.result)
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if response_model:
        response = client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            **kwargs,
        )
        return response.choices[0].message.content  # type: ignore


async def async_completion(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Async version of completion.

    Args:
        client: AsyncOpenAI client
        prompt: User prompt
        model: Model name
        system_prompt: Optional system message
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        Text response (str) or instance of response_model if provided
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if response_model:
        response = await client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            **kwargs,
        )
        return response.choices[0].message.content  # type: ignore


def chat_completion(
    client: OpenAI,
    messages: list[dict[str, Any]],
    model: str,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Full chat completion with message history.

    Args:
        client: OpenAI client
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        Text response (str) or instance of response_model if provided

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Hello"}
        ... ]
        >>> response = chat_completion(client, messages, model="llama3.2:1b")
    """
    if response_model:
        response = client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            **kwargs,
        )
        return response.choices[0].message.content  # type: ignore


async def async_chat_completion(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    model: str,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Async version of chat_completion.

    Args:
        client: AsyncOpenAI client
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        Text response (str) or instance of response_model if provided
    """
    if response_model:
        response = await client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            **kwargs,
        )
        return response.choices[0].message.content  # type: ignore


def stream_completion(
    client: OpenAI,
    messages: list[dict[str, Any]],
    model: str,
    **kwargs,
) -> Iterator[str]:
    """Stream text chunks from an LLM completion.

    Args:
        client: OpenAI client
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        **kwargs: Additional arguments for chat.completions.create

    Yields:
        Text chunks as they arrive

    Example:
        >>> client = create_ollama_client()
        >>> for chunk in stream_completion(
        ...     client,
        ...     [{"role": "user", "content": "Write a story"}],
        ...     model="llama3.2:1b"
        ... ):
        ...     print(chunk, end="", flush=True)
    """
    stream = client.chat.completions.create(
        model=model,
        messages=cast(list[ChatCompletionMessageParam], messages),
        stream=True,
        **kwargs,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def async_stream_completion(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    model: str,
    **kwargs,
) -> AsyncIterator[str]:
    """Async version of stream_completion.

    Args:
        client: AsyncOpenAI client
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        **kwargs: Additional arguments for chat.completions.create

    Yields:
        Text chunks as they arrive

    Example:
        >>> client = create_async_ollama_client()
        >>> async for chunk in async_stream_completion(
        ...     client,
        ...     [{"role": "user", "content": "Write a story"}],
        ...     model="llama3.2:1b"
        ... ):
        ...     print(chunk, end="", flush=True)
    """
    stream = await client.chat.completions.create(
        model=model,
        messages=cast(list[ChatCompletionMessageParam], messages),
        stream=True,
        **kwargs,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
