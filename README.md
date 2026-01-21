# ml-toolkit

A collection of reusable AI / ML components and related projects.

## [gen-ai](/gen-ai/)

Standardized GenAI utilities and helpers.

## [quiz-generator](/quiz-generator/)

Generate quizzes from research papers using agentic tool calling.

## [ai-embodiment](/ai-embodiment/)

Agent loop and embodiment primitives.

## [data-pipeline](/data-pipeline/)

Building blocks for data ingestion, transforms, and pipeline execution.

## [memory-layer](/memory-layer/)

Simple recursive text-based memory management.

## symbolic-world-modeling

## Installation and development

Install [uv](https://docs.astral.sh/uv/), then run `uv sync` at the repo root to create/update the shared environment and install all workspace members.

To add new packages for a specific component, use uv from that subdirectory, then update the workspace environment:

```bash
cd <component> && uv add <pkg>  # update dependencies in /component/pyproject.toml
uv lock && uv sync  # update /pyproject.toml and install new dependencies in /.venv
```