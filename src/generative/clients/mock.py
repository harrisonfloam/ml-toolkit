"""Mock LLM Clients

Mock implementations for testing and development.
"""

import random
from datetime import datetime
from typing import Any, Optional, TypeVar, Union, cast

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

from ..settings import settings
from .base import BaseLLMClient, ChatMessage

T = TypeVar("T", bound=BaseModel)


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

    def chat(
        self,
        messages: list[ChatMessage],
        model: str = settings.model_name,
        temperature: float = settings.temperature,
        **kwargs,
    ) -> Union[ChatCompletion, Any]:
        """Generate a mock ChatCompletion response."""
        user_message = messages[-1]["content"] if messages else "Hello"

        mock_responses = [
            user_message,  # Echo
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


class MockStructuredClient(MockLLMClient):
    """Mock structured LLM client that returns realistic fake Pydantic models."""

    def chat(
        self,
        messages: list[ChatMessage],
        model: str = "mock-model",
        temperature: float = 0.0,
        response_model: Optional[type[T]] = None,
        max_retries: int = 1,
        **kwargs,
    ) -> T:
        """Generate a mock structured response using the provided response_model."""
        if response_model is None:
            # Import here to avoid circular import
            from .structured import LLMResponse

            response_model = cast(type[T], LLMResponse)

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
