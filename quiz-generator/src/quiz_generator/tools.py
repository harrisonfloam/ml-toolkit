"""Agent Framework tools for this repo.

Design:
- Keep these as thin wrappers around the pure-ish domain functions in `quiz_generator.*`.
- Return JSON-serializable data (dicts) so the agent can emit structured results.
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated, Any

from agent_framework import ai_function
from openai import AsyncOpenAI
from pydantic import Field

from quiz_generator.ingestor import Paper
from quiz_generator.ingestor import ingest_arxiv_url as _ingest_arxiv_url
from quiz_generator.ingestor import ingest_text as _ingest_text

from .orchestrator import generate_quiz as _generate_quiz
from .orchestrator import plan_quiz as _plan_quiz
from .quiz.models import Difficulty, QuestionPlan


@dataclass(frozen=True)
class ToolRuntime:
    client: AsyncOpenAI
    model: str


_runtime: ToolRuntime | None = None


def set_tool_runtime(*, client: AsyncOpenAI, model: str) -> None:
    global _runtime
    _runtime = ToolRuntime(client=client, model=model)


def _require_runtime() -> ToolRuntime:
    if _runtime is None:
        raise RuntimeError(
            "Tool runtime is not configured. Call build_agent() (or set_tool_runtime()) first."
        )
    return _runtime


@ai_function(
    description=(
        "Ingest an arXiv URL by downloading the PDF and extracting text. "
        "Returns a Paper artifact as a JSON object."
    )
)
async def ingest_arxiv_url(
    arxiv_url: Annotated[
        str,
        Field(
            description=(
                "An arXiv URL (abs or pdf), e.g. https://arxiv.org/abs/1706.03762"
            )
        ),
    ],
    title: Annotated[
        str | None,
        Field(description="Optional title override."),
    ] = None,
    keep_pdf: Annotated[
        bool,
        Field(
            description=(
                "If true, keep the downloaded PDF and include its local path in metadata."
            )
        ),
    ] = False,
) -> dict[str, Any]:
    paper = await asyncio.to_thread(
        _ingest_arxiv_url, arxiv_url=arxiv_url, title=title, keep_pdf=keep_pdf
    )
    return paper.model_dump()


@ai_function(
    description=(
        "Ingest raw paper text. Returns a Paper artifact as a JSON object. "
        "Use this when you already have extracted text."
    )
)
async def ingest_text(
    text: Annotated[
        str,
        Field(description="Extracted paper text to ingest."),
    ],
    source_url: Annotated[
        str | None,
        Field(description="Optional source URL (if known)."),
    ] = None,
    title: Annotated[
        str | None,
        Field(description="Optional title."),
    ] = None,
) -> dict[str, Any]:
    paper = await asyncio.to_thread(
        _ingest_text, text=text, source_url=source_url, title=title
    )
    return paper.model_dump()


@ai_function(
    description=(
        "Plan quiz questions from an ingested Paper. "
        "Returns a quiz_id plus a list of planned questions (with orchestrator-owned IDs)."
    )
)
async def plan_quiz(
    paper: Annotated[
        dict[str, Any],
        Field(
            description="A Paper artifact as JSON (from ingest_text/ingest_arxiv_url)."
        ),
    ],
    num_questions: Annotated[
        int,
        Field(description="Number of questions to plan.", gt=0, le=50),
    ] = 5,
    difficulty: Annotated[
        Difficulty,
        Field(description="Difficulty: easy|medium|hard."),
    ] = "medium",
    max_paper_chars: Annotated[
        int,
        Field(
            description="Max characters of paper text to use (truncate).",
            gt=0,
            le=200_000,
        ),
    ] = 12_000,
) -> dict[str, Any]:
    parsed_paper = Paper.model_validate(paper)
    runtime = _require_runtime()
    quiz_id, plans = await _plan_quiz(
        client=runtime.client,
        model=runtime.model,
        paper=parsed_paper,
        num_questions=num_questions,
        difficulty=difficulty,
        max_paper_chars=max_paper_chars,
    )
    return {
        "quiz_id": quiz_id,
        "paper_id": parsed_paper.paper_id,
        "plans": [p.model_dump() for p in plans],
    }


@ai_function(
    description=(
        "Generate a quiz from a Paper and a list of planned questions (from plan_quiz). "
        "Returns a Quiz envelope as JSON."
    )
)
async def generate_quiz(
    paper: Annotated[
        dict[str, Any],
        Field(
            description="A Paper artifact as JSON (from ingest_text/ingest_arxiv_url)."
        ),
    ],
    quiz_id: Annotated[
        str,
        Field(description="quiz_id returned by plan_quiz.", min_length=1),
    ],
    plans: Annotated[
        list[dict[str, Any]],
        Field(description="List of QuestionPlan envelopes returned by plan_quiz."),
    ],
    max_paper_chars: Annotated[
        int,
        Field(
            description="Max characters of paper text to use (truncate).",
            gt=0,
            le=200_000,
        ),
    ] = 12_000,
) -> dict[str, Any]:
    parsed_paper = Paper.model_validate(paper)
    parsed_plans = [QuestionPlan.model_validate(p) for p in plans]
    runtime = _require_runtime()
    quiz = await _generate_quiz(
        client=runtime.client,
        model=runtime.model,
        paper=parsed_paper,
        quiz_id=quiz_id,
        plans=parsed_plans,
        max_paper_chars=max_paper_chars,
    )
    return quiz.model_dump()
