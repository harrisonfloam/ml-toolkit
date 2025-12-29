"""Agent construction.

Setup (example .env):
- OPENAI_API_KEY=...
- OPENAI_CHAT_MODEL_ID=gpt-4o-mini
- (optional, for Ollama) OPENAI_BASE_URL=http://localhost:11434/v1

Run:
- `python -m quiz_generator` (preferred)
- `python -m quiz_generator.agent` (compat)

Try:
- Paste an arXiv URL
- Or: "ingest this text: ..."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI

from quiz_generator import tools as agent_tools
from quiz_generator.llm_helpers import create_ollama_client
from quiz_generator.logging_config import init_logging
from quiz_generator.settings import settings

logger = logging.getLogger(__name__)


INSTRUCTIONS = """
You are a quiz generation agent.

You have these tools:
- ingest_arxiv_url: ingest an arXiv URL into a Paper artifact.
- ingest_text: ingest raw extracted text into a Paper artifact.
- plan_quiz: plan question topics/difficulties from a Paper (returns quiz_id + plans).
- generate_quiz: generate a quiz from a Paper + quiz_id + plans.

Rules:
- Use tools to do the work.
- IDs are orchestrator-owned: never invent or change question IDs yourself.
- After ingesting, plan and generate unless the user asks otherwise.
- When returning tool results, you may return the JSON object(s) directly.
""".strip()


@dataclass(frozen=True)
class AgentConfig:
    provider: Literal["ollama", "openai"] = "ollama"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


def _build_async_client_and_model(config: AgentConfig) -> tuple[AsyncOpenAI, str]:
    if config.provider == "ollama":
        base_url = config.base_url or settings.ollama_base_url
        model = config.model or settings.ollama_chat_model_id
        client = create_ollama_client(base_url=base_url, async_client=True)
        return client, model

    # OpenAI (or any OpenAI-compatible provider) via the official OpenAI endpoint by default.
    model = config.model or settings.openai_chat_model_id
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    return client, model


def build_agent(*, config: AgentConfig | None = None) -> ChatAgent:
    init_logging()
    resolved_config = config or AgentConfig()
    async_openai, model_id = _build_async_client_and_model(resolved_config)

    agent_tools.set_tool_runtime(client=async_openai, model=model_id)

    logger.info("Starting quiz generator agent")
    logger.info("Provider: %s", resolved_config.provider)
    logger.info("Model: %s", model_id)
    if resolved_config.base_url:
        logger.info("Base URL: %s", resolved_config.base_url)

    chat_client = OpenAIChatClient(model_id=model_id, async_client=async_openai)
    agent = ChatAgent(
        name="QuizGeneratorIngestAgent",
        description="Quiz generator agent for papers (ingest -> plan -> generate).",
        instructions=INSTRUCTIONS,
        chat_client=chat_client,
        tools=[
            agent_tools.ingest_arxiv_url,
            agent_tools.ingest_text,
            agent_tools.plan_quiz,
            agent_tools.generate_quiz,
        ],
    )
    return agent


if __name__ == "__main__":
    from .cli import main

    main()
