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

Model Listing:
    list_models(): List available models
    async_list_models(): Async version of list_models()

Mock Functions:
    mock_chat_completion(): Mock chat response for testing
    mock_stream_completion(): Mock streaming response for testing
"""

import logging
import os
import random
import time
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Literal,
    Optional,
    TypeVar,
    cast,
    overload,
)

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

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@overload
def create_ollama_client(
    base_url: str = "http://localhost:11434",
    *,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> OpenAI: ...


@overload
def create_ollama_client(
    base_url: str = "http://localhost:11434",
    *,
    async_client: Literal[True],
    **kwargs: Any,
) -> AsyncOpenAI: ...


def create_ollama_client(
    base_url: str = "http://localhost:11434",
    *,
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
        # Tag provider as 'ollama'
        ollama_client = AsyncOpenAI(
            api_key="ollama",
            base_url=normalized_url,
            **kwargs,
        )
        setattr(ollama_client, "_llm_provider", "ollama")
        return ollama_client
    else:
        ollama_client = OpenAI(
            api_key="ollama",
            base_url=normalized_url,
            **kwargs,
        )
        setattr(ollama_client, "_llm_provider", "ollama")
        return ollama_client


@overload
def create_mistral_client(
    api_key: Optional[str] = None,
    server_url: str = "https://api.mistral.ai",
    *,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> OpenAI: ...


@overload
def create_mistral_client(
    api_key: Optional[str] = None,
    server_url: str = "https://api.mistral.ai",
    *,
    async_client: Literal[True],
    **kwargs: Any,
) -> AsyncOpenAI: ...


def create_mistral_client(
    api_key: Optional[str] = None,
    server_url: str = "https://api.mistral.ai",
    *,
    async_client: bool = False,
    **kwargs: Any,
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
    from mistralai import Mistral  # Optional dependency, only required for Mistral API

    resolved_api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Mistral API key must be provided or set in MISTRAL_API_KEY environment variable"
        )

    mistral_client = Mistral(api_key=resolved_api_key, server_url=server_url, **kwargs)

    # Tag provider as 'mistralai'
    setattr(mistral_client, "_llm_provider", "mistralai")

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
    **kwargs: Any,
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
        return _process_structured_response(response, response_model, model, start_time)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            **kwargs,
        )
        return _process_text_response(response, model, start_time)


async def async_completion(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    response_model: Optional[type[T]] = None,
    **kwargs: Any,
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
        return _process_structured_response(response, response_model, model, start_time)
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            **kwargs,
        )
        return _process_text_response(response, model, start_time)


def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    response_model: Optional[type[T]] = None,
    **kwargs: Any,
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
        return _process_structured_response(response, response_model, model, start_time)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            **kwargs,
        )
        return _process_text_response(response, model, start_time)


async def async_chat_completion(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    response_model: Optional[type[T]] = None,
    **kwargs: Any,
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
        return _process_structured_response(response, response_model, model, start_time)
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            temperature=temperature,
            **kwargs,
        )
        return _process_text_response(response, model, start_time)


def stream_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    **kwargs: Any,
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
        f"LLM stream started in {duration:.4f} s, model={model} temperature={temperature}"
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
    **kwargs: Any,
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
        f"LLM stream started in {duration:.4f} s, model={model} temperature={temperature}"
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


def list_models(
    client: OpenAI,
    base_url: str = "http://localhost:11434",
    include_capabilities: bool = False,
) -> list[str] | list[dict[str, Any]]:
    """List available models.

    Args:
        client: OpenAI client
        base_url: Base URL for capability detection (Ollama only)
        include_capabilities: If True, fetch Ollama-specific capabilities

    Returns:
        list[str] of model names, or list[dict] with capabilities if include_capabilities=True

    Raises:
        NotImplementedError: If include_capabilities=True for non-Ollama providers

    Example:
        >>> models = list_models(client)
        ['llama3.2:1b', 'mistral:latest']
        >>> models = list_models(client, include_capabilities=True)
        [{'name': 'llama3.2:1b', 'capabilities': ['completion']}, ...]
    """
    import httpx  # Optional dependency, only required for capability detection

    models = client.models.list()
    model_names = [model.id for model in models.data]

    if not include_capabilities:
        return model_names

    # Ollama capability detection
    import concurrent.futures  # Stdlib, for parallel HTTP requests

    ollama_api_url = f"{base_url.rstrip('/')}/api/show"

    def fetch_model_capabilities(model_name: str) -> dict[str, Any]:
        with httpx.Client(timeout=10.0) as http_client:
            try:
                response = http_client.post(
                    ollama_api_url,
                    json={"name": model_name},
                )
                response.raise_for_status()
                model_info = response.json()
                capabilities = model_info.get("capabilities", [])
                return {"name": model_name, "capabilities": capabilities}
            except (httpx.HTTPError, httpx.ConnectError) as e:
                # Not Ollama or capability detection not supported
                raise NotImplementedError(
                    f"Capability detection not supported for this provider: {e}"
                ) from e

    with concurrent.futures.ThreadPoolExecutor() as executor:
        models_info = list(executor.map(fetch_model_capabilities, model_names))

    return models_info


async def async_list_models(
    client: AsyncOpenAI,
    base_url: str = "http://localhost:11434",
    include_capabilities: bool = False,
) -> list[str] | list[dict[str, Any]]:
    """Async version of list_models.

    Args:
        client: AsyncOpenAI client
        base_url: Base URL for capability detection (Ollama only)
        include_capabilities: If True, fetch Ollama-specific capabilities

    Returns:
        list[str] of model names, or list[dict] with capabilities if include_capabilities=True

    Raises:
        NotImplementedError: If include_capabilities=True for non-Ollama providers

    Example:
        >>> models = await async_list_models(client)
        ['llama3.2:1b', 'mistral:latest']
        >>> models = await async_list_models(client, include_capabilities=True)
        [{'name': 'llama3.2:1b', 'capabilities': ['completion']}, ...]
    """
    import httpx  # Optional dependency, only required for capability detection

    models = await client.models.list()
    model_names = [model.id for model in models.data]

    if not include_capabilities:
        return model_names

    # Ollama capability detection
    import asyncio  # Stdlib, for parallel async requests

    ollama_api_url = f"{base_url.rstrip('/')}/api/show"

    async def fetch_model_capabilities(
        model_name: str, http_client: httpx.AsyncClient
    ) -> dict[str, Any]:
        try:
            response = await http_client.post(
                ollama_api_url,
                json={"name": model_name},
            )
            response.raise_for_status()
            model_info = response.json()
            capabilities = model_info.get("capabilities", [])
            return {"name": model_name, "capabilities": capabilities}
        except (httpx.HTTPError, httpx.ConnectError) as e:
            # Not Ollama or capability detection not supported
            raise NotImplementedError(
                f"Capability detection not supported for this provider: {e}"
            ) from e

    async with httpx.AsyncClient(timeout=10.0) as http_client:
        models_info = await asyncio.gather(
            *[
                fetch_model_capabilities(model_name, http_client)
                for model_name in model_names
            ]
        )

    return list(models_info)


def mock_chat_completion(
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


def _process_structured_response(
    response: ChatCompletion,
    response_model: type[T],
    model: str,
    start_time: float,
) -> T:
    """Process a structured response with a Pydantic model."""
    result = response_model.model_validate_json(response.choices[0].message.content)  # type: ignore
    duration = time.time() - start_time
    _log_chat_response(
        model=model,
        content_len=len(str(result)),
        duration_s=duration,
        usage=response.usage,
    )
    return result


def _process_text_response(
    response: ChatCompletion,
    model: str,
    start_time: float,
) -> str:
    """Process a text response."""
    content = response.choices[0].message.content or ""
    duration = time.time() - start_time
    _log_chat_response(
        model=model,
        content_len=len(content),
        duration_s=duration,
        usage=response.usage,
    )
    return content


def _log_chat_request(
    *, model: str, temperature: float, messages: list[dict[str, Any]]
) -> None:
    """Log chat request details."""
    logger.debug(
        f"LLM request: model={model} temperature={temperature} messages={len(messages)}"
    )


def _log_chat_response(
    *, model: str, content_len: int, duration_s: float, usage=None
) -> None:
    """Log chat response details."""
    usage_str = ""
    if usage:
        usage_str = f" tokens={{prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}}}"
    logger.debug(
        f"LLM response: model={model} content_len={content_len} duration_s={duration_s:.4f}{usage_str}"
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
