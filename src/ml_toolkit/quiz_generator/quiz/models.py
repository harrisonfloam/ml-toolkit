from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ChoiceId = Literal["A", "B", "C", "D"]
Difficulty = Literal["easy", "medium", "hard"]


class MCQChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: ChoiceId = Field(description="Choice label. Must be one of A, B, C, D.")
    text: str = Field(
        min_length=1,
        description="Choice text shown to the user.",
    )


class MCQQuestion(BaseModel):
    """Single-correct 4-option multiple choice question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(
        min_length=1,
        description="Identifier for this question (caller-generated or LLM-generated).",
    )
    prompt: str = Field(
        min_length=1,
        description="The question text presented to the user.",
    )

    choices: list[MCQChoice] = Field(
        min_length=4,
        max_length=4,
        description="Exactly 4 choices labeled A, B, C, D.",
    )
    correct_choice_id: ChoiceId = Field(
        description="The single correct choice id (A, B, C, or D).",
    )

    rationale: str = Field(
        min_length=1,
        description="Short explanation of why the correct answer is correct.",
    )

    def pretty(self) -> str:
        """Human-friendly rendering for notebooks/logs."""
        lines = [f"{self.question_id}: {self.prompt.strip()}"]
        for choice in self.choices:
            lines.append(f"  {choice.id}. {choice.text.strip()}")
        lines.append(f"Answer: {self.correct_choice_id}")
        if self.rationale.strip():
            lines.append(f"Rationale: {self.rationale.strip()}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty()

    @field_validator("choices")
    @classmethod
    def _choices_are_abcd_once(cls, choices: list[MCQChoice]) -> list[MCQChoice]:
        ids = [c.id for c in choices]
        if len(ids) != 4 or set(ids) != {"A", "B", "C", "D"}:
            raise ValueError("choices must contain exactly A, B, C, D (once each)")
        return choices

    @model_validator(mode="after")
    def _correct_choice_present(self) -> "MCQQuestion":
        if self.correct_choice_id not in {c.id for c in self.choices}:
            raise ValueError("correct_choice_id must match one of the choice ids")
        return self


class MCQQuestionContent(BaseModel):
    """Content-only MCQ (no IDs).

    V0 envelope pattern:
    - LLM returns only this model.
    - Orchestrator assigns stable IDs and wraps into `MCQQuestion`.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        min_length=1,
        description="The question text presented to the user.",
    )

    choices: list[MCQChoice] = Field(
        min_length=4,
        max_length=4,
        description="Exactly 4 choices labeled A, B, C, D.",
    )
    correct_choice_id: ChoiceId = Field(
        description="The single correct choice id (A, B, C, or D).",
    )

    rationale: str = Field(
        min_length=1,
        description="Short explanation of why the correct answer is correct.",
    )

    @field_validator("choices")
    @classmethod
    def _choices_are_abcd_once(cls, choices: list[MCQChoice]) -> list[MCQChoice]:
        ids = [c.id for c in choices]
        if len(ids) != 4 or set(ids) != {"A", "B", "C", "D"}:
            raise ValueError("choices must contain exactly A, B, C, D (once each)")
        return choices

    @model_validator(mode="after")
    def _correct_choice_present(self) -> "MCQQuestionContent":
        if self.correct_choice_id not in {c.id for c in self.choices}:
            raise ValueError("correct_choice_id must match one of the choice ids")
        return self


class QuestionPlanContent(BaseModel):
    """Content-only question plan (no IDs)."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, description="Topic/focus for the question.")
    difficulty: Difficulty = Field(description="Difficulty: easy|medium|hard.")


class QuestionPlan(BaseModel):
    """Planned question envelope with orchestrator-owned ID."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1, description="Identifier for the question.")
    topic: str = Field(min_length=1, description="Topic/focus for the question.")
    difficulty: Difficulty = Field(description="Difficulty: easy|medium|hard.")


class Quiz(BaseModel):
    """Quiz envelope."""

    model_config = ConfigDict(extra="forbid")

    quiz_id: str = Field(min_length=1, description="Identifier for this quiz.")
    paper_id: str = Field(min_length=1, description="Identifier for the source paper.")
    title: str | None = Field(default=None, description="Optional quiz title.")
    questions: list[MCQQuestion] = Field(
        min_length=1,
        description="Generated quiz questions (enveloped with stable IDs).",
    )
