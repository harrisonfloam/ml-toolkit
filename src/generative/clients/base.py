"""Base LLM Clients

Core implementations for OpenAI and Ollama clients without structured output dependencies.

Function:
- Adds custom logging and callbacks
- Abstracts the client interface (allows for swapping out openai if needed)
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Dict, List, Union

import openai
from openai.types.chat import ChatCompletion

from ...utils.callbacks import CallbackMeta, with_callbacks
from ..settings import settings

logger = logging.getLogger(__name__)

ChatMessage = Dict[str, Any]


class BaseLLMClient(ABC, metaclass=CallbackMeta):
    """Abstract base class for all LLM clients."""

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
        try:
            result_dump = (
                result.model_dump() if hasattr(result, "model_dump") else str(result)
            )
        except Exception:
            result_dump = str(result)
        logger.debug(
            f"Chat response in {duration:.4f}s:\n{json.dumps(result_dump, indent=2, default=str)}"
        )
        return result


class OpenAILLMClient(BaseLLMClient):
    """OpenAI client for chat completions."""

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
    """Async OpenAI client for chat completions."""

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
    """Ollama client using OpenAI-compatible API."""

    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(
            log_level=log_level,
            base_url=ollama_url,
            api_key="ollama",  # required, but unused
        )


class AsyncOllamaLLMClient(AsyncOpenAILLMClient):
    """Async Ollama client using OpenAI-compatible API."""

    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(
            log_level=log_level,
            base_url=ollama_url,
            api_key="ollama",  # required, but unused
        )
