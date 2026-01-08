from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from data_pipeline.types import DatasetSpec, PipelineConfig


def load_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Config must be a mapping")

    data_root = Path(raw.get("data_root", "data"))
    runs_root = Path(raw.get("runs_root", "runs"))
    stages_value = raw.get("stages", ["raw", "cleaned", "transformed"])
    if not isinstance(stages_value, list) or not all(
        isinstance(s, str) for s in stages_value
    ):
        raise ValueError("stages must be a list of strings")
    stages = tuple(stages_value)

    datasets_raw: Any = raw.get("datasets", {})
    if datasets_raw is None:
        datasets_raw = {}
    if not isinstance(datasets_raw, Mapping):
        raise ValueError("datasets must be a mapping")

    datasets: dict[str, DatasetSpec] = {}
    for name, spec_raw in datasets_raw.items():
        if not isinstance(name, str):
            raise ValueError("dataset names must be strings")
        if not isinstance(spec_raw, Mapping):
            raise ValueError(f"dataset spec for {name} must be a mapping")
        fmt = spec_raw.get("format")
        rel_path = spec_raw.get("path")
        if not isinstance(fmt, str) or not isinstance(rel_path, str):
            raise ValueError(
                f"dataset spec for {name} must have string format and path"
            )
        datasets[name] = DatasetSpec(format=fmt, path=rel_path)

    cfg = PipelineConfig(
        data_root=data_root,
        runs_root=runs_root,
        stages=stages,
        datasets=datasets,
        base_dir=config_path.parent.resolve(),
    )
    return replace(cfg)
