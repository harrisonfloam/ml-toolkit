"""Simple helper functions for working with LLMs."""

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
        str if no response_model, otherwise instance of response_model

    Example:
        >>> text = completion(client, "What is 2+2?", model="llama3.2:1b")
        >>> # With structured output:
        >>> answer = completion(client, "What is 2+2?", model="llama3.2:1b", response_model=Answer)
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Cast: our dicts match the ChatCompletionMessageParam protocol
    msg_params = cast(list[ChatCompletionMessageParam], messages)

    if response_model:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            **kwargs,
        )
        content = response.choices[0].message.content
        return content or ""


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
        str if no response_model, otherwise instance of response_model
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    msg_params = cast(list[ChatCompletionMessageParam], messages)

    if response_model:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            **kwargs,
        )
        content = response.choices[0].message.content
        return content or ""


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
        str if no response_model, otherwise instance of response_model

    Example:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> response = chat_completion(client, messages, model="llama3.2:1b")
    """
    msg_params = cast(list[ChatCompletionMessageParam], messages)

    if response_model:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            **kwargs,
        )
        content = response.choices[0].message.content
        return content or ""


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
        str if no response_model, otherwise instance of response_model
    """
    msg_params = cast(list[ChatCompletionMessageParam], messages)

    if response_model:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            **kwargs,
        )
        content = response.choices[0].message.content
        return content or ""


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
        >>> for chunk in stream_completion(client, [{"role": "user", "content": "Hi"}], model="llama3.2:1b"):
        ...     print(chunk, end="", flush=True)
    """
    msg_params = cast(list[ChatCompletionMessageParam], messages)
    stream = client.chat.completions.create(
        model=model,
        messages=msg_params,
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
        >>> async for chunk in async_stream_completion(client, [{"role": "user", "content": "Hi"}], model="llama3.2:1b"):
        ...     print(chunk, end="", flush=True)
    """
    msg_params = cast(list[ChatCompletionMessageParam], messages)
    stream = await client.chat.completions.create(
        model=model,
        messages=msg_params,
        stream=True,
        **kwargs,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
