from __future__ import annotations

from typing import cast

from openai import AsyncOpenAI

from quiz_generator.llm_helpers import async_chat_completion

from .models import Difficulty, MCQQuestionContent


async def generate_mcq_content(
    *,
    client: AsyncOpenAI,
    model: str,
    context: str,
    topic: str | None = None,
    difficulty: Difficulty = "medium",  # TODO: clean this up
    temperature: float = 0.3,
) -> MCQQuestionContent:
    """Generate a single 4-option, single-correct MCQ grounded in `context`.

    V0 behavior:
    - The function both *invents* the question and generates choices.
    - Evidence is optional; it may be returned as an empty list.

    Args:
            client: OpenAI-compatible client (OpenAI API or Ollama).
            model: Model name.
            context: The only source material the model may use.
            topic: Optional nudge for what to ask about.
            difficulty: easy|medium|hard.
            temperature: Sampling temperature.
    Returns:
            MCQQuestion parsed/validated via strict JSON schema output.

    TODO:
    - this should take 'instructions' so the agent can define constraints in natural language (dont overlap with this other question, etc)
    """
    if not context or not context.strip():
        raise ValueError("context must be a non-empty string")

    system = """You generate a single multiple-choice question (MCQ) strictly grounded in the provided context.

Do not use outside knowledge.

Return ONLY JSON that matches the provided schema."""

    topic_line = f"\ntopic: {topic}" if topic else ""
    user = f"""difficulty: {difficulty}{topic_line}
context:
{context.strip()}"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    result = cast(
        MCQQuestionContent,
        await async_chat_completion(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            response_model=MCQQuestionContent,
        ),
    )

    return result
