import json

from quiz_generator import tools as agent_tools
from quiz_generator.quiz.models import MCQChoice, MCQQuestion, Quiz


def test_mcq_requires_abcd_choices_once_each() -> None:
    q = MCQQuestion(
        question_id="q1",
        prompt="What is 2+2?",
        choices=[
            MCQChoice(id="A", text="3"),
            MCQChoice(id="B", text="4"),
            MCQChoice(id="C", text="5"),
            MCQChoice(id="D", text="22"),
        ],
        correct_choice_id="B",
        rationale="2+2 equals 4.",
    )
    assert q.correct_choice_id == "B"


def test_domain_models_have_json_schema_and_round_trip() -> None:
    q = MCQQuestion(
        question_id="quiz_123_q1",
        prompt="What is 2+2?",
        choices=[
            MCQChoice(id="A", text="3"),
            MCQChoice(id="B", text="4"),
            MCQChoice(id="C", text="5"),
            MCQChoice(id="D", text="22"),
        ],
        correct_choice_id="B",
        rationale="2+2 equals 4.",
    )
    quiz = Quiz(
        quiz_id="quiz_123",
        paper_id="paper_abc",
        title="Test Quiz",
        questions=[q],
    )

    # Schema generation should not crash.
    schema = Quiz.model_json_schema()
    assert schema.get("type") == "object"

    # JSON serialization should not crash.
    dumped = quiz.model_dump()
    dumped_json = quiz.model_dump_json()
    assert isinstance(dumped, dict)
    assert isinstance(dumped_json, str)

    # Round-trip should preserve key fields.
    quiz2 = Quiz.model_validate(dumped)
    assert quiz2.quiz_id == quiz.quiz_id
    assert quiz2.paper_id == quiz.paper_id
    assert quiz2.questions[0].correct_choice_id == "B"


def test_agent_tools_have_json_schema_specs() -> None:
    # The agent framework relies on tool input models being JSON-schema serializable.
    # This catches regressions where annotations become unresolved (e.g., forward refs).
    tools = [
        agent_tools.ingest_arxiv_url,
        agent_tools.ingest_text,
        agent_tools.plan_quiz,
        agent_tools.generate_quiz,
    ]

    for tool in tools:
        assert hasattr(tool, "to_json_schema_spec")
        spec = tool.to_json_schema_spec()
        assert isinstance(spec, dict)
        json.dumps(spec)  # must be JSON-serializable
