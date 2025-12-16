from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration for a named dataset."""

    format: str
    path: str


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline configuration loaded from a file or created directly."""

    data_root: Path = Path("data")
    runs_root: Path = Path("runs")
    stages: tuple[str, ...] = ("raw", "cleaned", "transformed")
    datasets: Mapping[str, DatasetSpec] = field(default_factory=dict)
    base_dir: Path | None = None

    def resolve_path(self, maybe_relative: Path) -> Path:
        """Resolve a path relative to the config base dir."""

        if maybe_relative.is_absolute() or self.base_dir is None:
            return maybe_relative
        return (self.base_dir / maybe_relative).resolve()

    @property
    def resolved_data_root(self) -> Path:
        """Absolute data root."""

        return self.resolve_path(self.data_root)

    @property
    def resolved_runs_root(self) -> Path:
        """Absolute runs root."""

        return self.resolve_path(self.runs_root)

    def next_stage(self, src_stage: str) -> str:
        """Return the next stage after src_stage."""

        try:
            index = self.stages.index(src_stage)
        except ValueError as exc:
            raise ValueError(f"Unknown src_stage: {src_stage}") from exc

        if index >= len(self.stages) - 1:
            raise ValueError(f"No next stage after: {src_stage}")
        return self.stages[index + 1]


@dataclass(frozen=True)
class RunContext:
    """Context passed to transforms."""

    dataset: str
    src_stage: str
    dst_stage: str
    run_id: str
    params: Mapping[str, Any]
    input_path: Path
    output_path: Path
    logger: logging.Logger


TransformFn = Callable[[Any, RunContext], Any]

# TODO: Support streaming transforms.
# Target shape (v1-ish): iterable[dict] -> iterable[dict] (or iterator/generator),
# with IO able to stream formats like JSONL.


def transform_name(transform: TransformFn) -> str:
    """Best-effort stable name for a transform callable."""

    module = getattr(transform, "__module__", None)
    qualname = getattr(transform, "__qualname__", None)
    if module and qualname:
        return f"{module}:{qualname}"
    return repr(transform)
