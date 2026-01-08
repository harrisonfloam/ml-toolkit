from __future__ import annotations

import uuid

from openai import AsyncOpenAI

from quiz_generator.ingestor import Paper

from .quiz.generator import generate_mcq_content
from .quiz.models import (
    Difficulty,
    MCQQuestion,
    QuestionPlan,
    Quiz,
)
from .quiz.planner import plan_question_contents


def truncate_text(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    text = (text or "").strip()
    if not text:
        raise ValueError("text must be a non-empty string")
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def truncate_paper(paper: Paper, *, max_chars: int) -> Paper:
    truncated = truncate_text(paper.text, max_chars=max_chars)
    return paper.model_copy(update={"text": truncated})


def make_quiz_id() -> str:
    return f"quiz_{uuid.uuid4().hex}"


def make_question_ids(*, quiz_id: str, num_questions: int) -> list[str]:
    if not quiz_id or not quiz_id.strip():
        raise ValueError("quiz_id must be a non-empty string")
    if num_questions <= 0:
        raise ValueError("num_questions must be > 0")
    return [f"{quiz_id}_q{i + 1}" for i in range(num_questions)]


async def plan_quiz(
    *,
    client: AsyncOpenAI,
    model: str,
    paper: Paper,
    num_questions: int,
    difficulty: Difficulty = "medium",
    temperature: float = 0.2,
    max_paper_chars: int = 12_000,
    quiz_id: str | None = None,
) -> tuple[str, list[QuestionPlan]]:
    """Plan questions for a quiz.

    V0 envelope rules:
    - LLM returns only content (topic/difficulty)
    - Orchestrator assigns quiz/question IDs
    """

    resolved_quiz_id = quiz_id or make_quiz_id()
    truncated_paper = truncate_paper(paper, max_chars=max_paper_chars)

    contents = await plan_question_contents(
        client=client,
        model=model,
        paper=truncated_paper,
        num_questions=num_questions,
        difficulty=difficulty,
        temperature=temperature,
    )

    question_ids = make_question_ids(
        quiz_id=resolved_quiz_id, num_questions=num_questions
    )

    plans: list[QuestionPlan] = []
    for question_id, content in zip(question_ids, contents, strict=True):
        plans.append(
            QuestionPlan(
                question_id=question_id,
                topic=content.topic,
                difficulty=content.difficulty,
            )
        )

    return resolved_quiz_id, plans


async def generate_quiz(
    *,
    client: AsyncOpenAI,
    model: str,
    paper: Paper,
    quiz_id: str,
    plans: list[QuestionPlan],
    temperature: float = 0.3,
    max_paper_chars: int = 12_000,
) -> Quiz:
    """Generate a full quiz from a paper and precomputed plans."""

    truncated_paper = truncate_paper(paper, max_chars=max_paper_chars)
    context = truncated_paper.text

    questions: list[MCQQuestion] = []
    for plan in plans:
        content = await generate_mcq_content(
            client=client,
            model=model,
            context=context,
            topic=plan.topic,
            difficulty=plan.difficulty,
            temperature=temperature,
        )
        questions.append(
            MCQQuestion(
                question_id=plan.question_id,
                prompt=content.prompt,
                choices=content.choices,
                correct_choice_id=content.correct_choice_id,
                rationale=content.rationale,
            )
        )

    return Quiz(
        quiz_id=quiz_id,
        paper_id=paper.paper_id,
        title=paper.title,
        questions=questions,
    )


async def plan_and_generate_quiz(
    *,
    client: AsyncOpenAI,
    model: str,
    paper: Paper,
    num_questions: int,
    difficulty: Difficulty = "medium",
    plan_temperature: float = 0.2,
    generate_temperature: float = 0.3,
    max_paper_chars: int = 12_000,
) -> tuple[list[QuestionPlan], Quiz]:
    quiz_id, plans = await plan_quiz(
        client=client,
        model=model,
        paper=paper,
        num_questions=num_questions,
        difficulty=difficulty,
        temperature=plan_temperature,
        max_paper_chars=max_paper_chars,
    )
    quiz = await generate_quiz(
        client=client,
        model=model,
        paper=paper,
        quiz_id=quiz_id,
        plans=plans,
        temperature=generate_temperature,
        max_paper_chars=max_paper_chars,
    )
    return plans, quiz
