from __future__ import annotations

from typing import cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from quiz_generator.ingestor import Paper
from quiz_generator.llm_helpers import async_chat_completion

from .models import Difficulty, QuestionPlanContent


class QuestionPlanContents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plans: list[QuestionPlanContent] = Field(
        min_length=1,
        description="Planned question intents.",
    )


async def plan_question_contents(
    *,
    client: AsyncOpenAI,
    model: str,
    paper: Paper,
    num_questions: int,
    difficulty: Difficulty = "medium",
    temperature: float = 0.2,
) -> list[QuestionPlanContent]:
    """Plan a set of question topics before generation.

    This step aims to reduce major conceptual overlap across questions.

    TODO: context window management for full papers. If the paper does not fit in the model context,
    we likely need a multi-pass approach (chunk -> extract candidate topics -> merge/dedupe/prune
    to `num_questions`) rather than sending the entire paper in one call.

    TODO: add a simple post-pass to de-duplicate/adjust overly-similar topics.
    """
    if not paper.text or not paper.text.strip():
        raise ValueError("paper.text must be a non-empty string")
    if num_questions <= 0:
        raise ValueError("num_questions must be > 0")

    system = """You plan a set of multiple-choice questions (MCQs) for a quiz.

Return ONLY JSON that matches the provided schema.
"""

    title_line = f"\npaper_title: {paper.title}" if paper.title else ""
    user = f"""Plan questions that cover distinct, important ideas from the paper.
Avoid asking the same question twice.

Return {num_questions} plans. Use difficulty='{difficulty}' for every plan.

num_questions: {num_questions}
difficulty: {difficulty}
paper_id: {paper.paper_id}{title_line}

paper_context:
{paper.text.strip()}"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    plans = cast(
        QuestionPlanContents,
        await async_chat_completion(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            response_model=QuestionPlanContents,
        ),
    )

    return plans.plans
