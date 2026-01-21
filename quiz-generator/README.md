# quiz-generator

Generate a multiple choice quiz to encourage deep retention of research paper content.

## Status

Work in progress; the CLI and Python API are usable but still evolving.

## Major dependencies

- [agent-framework](https://github.com/microsoft/agent-framework): Microsoft's newest agentic framework, positioned to become the industry standard after merging [Semantic Kernel](https://github.com/microsoft/semantic-kernel) and [AutoGen](https://github.com/microsoft/autogen).
- [openai](https://github.com/openai/openai-python): OpenAI's Python SDK.
- [MarkItDown](https://pypi.org/project/markitdown/): Simple PDF/docx text extraction.
- [Pydantic](https://docs.pydantic.dev/): Typed models for quiz plans and generated quizzes.
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/): Extremely convenient configuration management.
- [Streamlit](https://streamlit.io/): simple UI for interactive use.

## CLI

Run the CLI and specify initial configuration:
```bash
# Uses Ollama by default
python -m quiz_generator

# Override provider/model
python -m quiz_generator --provider=ollama --model mistral-small3.2:latest

# Use OpenAI (or any OpenAI-compatible provider)
python -m quiz_generator --provider=openai
```

Give the quiz generator agent a task using natural language:

```text
$ python -m quiz_generator
[INFO] quiz_generator.agent: Starting quiz generator agent
[INFO] quiz_generator.agent: Provider: ollama
[INFO] quiz_generator.agent: Model: mistral-small3.2:latest
Quiz generator ready. Type 'exit' to quit.
You> Ingest https://arxiv.org/abs/1706.03762 then generate a 5-question medium-difficulty multiple-choice quiz. Return only the generated quiz JSON.
{...}
You> exit
```

## Python API

### Deterministic workflow

```python
from openai import AsyncOpenAI

from quiz_generator.ingestor import ingest_arxiv_url
from quiz_generator.llm_helpers import create_ollama_client
from quiz_generator.orchestrator import plan_and_generate_quiz

# Ingest a paper
paper = ingest_arxiv_url(arxiv_url="https://arxiv.org/abs/1706.03762")

# Create an LLM client - Ollama or any OpenAI-compatible provider
client: AsyncOpenAI = create_ollama_client(base_url="http://localhost:11434", async_client=True)
# client = AsyncOpenAI()

plans, quiz = await plan_and_generate_quiz(
    client=client,
    model="mistral-small3.2:latest",
    paper=paper,
    num_questions=5,
    difficulty="medium",
)

print(f"Planned: {len(plans)}")
print(f"Generated: {len(quiz.questions)}")
print(quiz.questions[0])

# Example output (`MCQQuestion.__str__` uses `MCQQuestion.pretty()`):
# quiz_0123456789abcdef0123456789abcdef_q1: What is the primary role of self-attention in the Transformer?
#   A. To add convolutional inductive bias
#   B. To enable each token to weigh other tokens' importance
#   C. To reduce the sequence length by pooling
#   D. To replace positional encoding entirely
# Answer: B
# Rationale: Self-attention lets each position attend to other positions, capturing long-range dependencies.
```

### Agentic workflow

Microsoft Agent Framework's `ChatAgent` uses the same tools as in the deterministic workflow to assemble a quiz.

```python
from quiz_generator.agent import build_agent

agent = build_agent()

# NOTE: `await` must be in an async context
result = await agent.run(
    "Check out https://arxiv.org/abs/1706.03762, then generate a 5-question medium-difficulty multiple-choice quiz."
)
```