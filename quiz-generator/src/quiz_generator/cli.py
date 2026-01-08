from __future__ import annotations

import argparse
import asyncio
import logging
from time import perf_counter

from .agent import AgentConfig, build_agent

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> AgentConfig:
    parser = argparse.ArgumentParser(prog="quiz_generator")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM provider (uses QUIZ_GENERATOR_ defaults unless overridden).",
    )
    parser.add_argument(
        "--model", default=None, help="Chat model ID (overrides default)."
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL (e.g. http://localhost:11434 for Ollama).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override (otherwise uses environment/defaults).",
    )

    args = parser.parse_args(argv)
    return AgentConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )


async def run_cli(config: AgentConfig) -> None:
    agent = build_agent(config=config)

    print("Quiz generator ready. Type 'exit' to quit.")

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            return

        logger.debug("User input: %s", user_text)
        start = perf_counter()
        try:
            result = await agent.run(user_text)
        except asyncio.CancelledError:
            logger.info("Cancelled request")
            print("\nCancelled.\n")
            continue
        except Exception:
            logger.exception("agent.run failed")
            print("\nERROR: agent.run failed (see logs above).\n")
            continue
        finally:
            logger.info("agent.run completed in %.2fs", perf_counter() - start)

        logger.debug("Agent output: %s", result)
        print(result)


def main(argv: list[str] | None = None) -> None:
    try:
        config = _parse_args(argv)
        asyncio.run(run_cli(config))
    except KeyboardInterrupt:
        print("\nExiting.")
