# ml-toolkit

A collection of reusable AI / ML components and related projects.

## [gen-ai](src/ml_toolkit/gen_ai/README.md)

Standardized GenAI utilities and helpers.

## [quiz-generator](src/ml_toolkit/quiz_generator/README.md)

Generate quizzes from research papers using agentic tool calling.

## [ai-embodiment](src/ml_toolkit/ai_embodiment/README.md)

Agent loop and embodiment primitives.

## [data-pipeline](src/ml_toolkit/data_pipeline/README.md)

Building blocks for data ingestion, transforms, and pipeline execution.

## [memory-layer](src/ml_toolkit/memory_layer/README.md)

Simple recursive text-based memory management.

## symbolic-world-modeling

## Installation and development

Install [uv](https://docs.astral.sh/uv/), then run `uv sync` at the repo root to create/update the environment and install `ml-toolkit`.

Most functionality lives behind extras (so base installs stay lightweight). For a "full" dev environment:

```bash
uv sync --all-extras
```

To install only the dependencies for a specific area, sync the relevant extra(s):

```bash
uv sync --extra gen-ai --extra quiz-generator
```

To add a dependency to a specific extra, use uv from the repo root:

```bash
uv add --optional <extra> <pkg>
uv lock && uv sync
```