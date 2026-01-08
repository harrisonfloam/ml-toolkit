from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_pipeline.types import DatasetSpec, PipelineConfig


@dataclass(frozen=True)
class DatasetRegistry:
    """Registry for datasets defined in config. Registries centralize dataset/path rules for easier use downstream."""

    config: PipelineConfig

    def get(self, dataset: str) -> DatasetSpec:
        """Return dataset spec by name."""

        try:
            return self.config.datasets[dataset]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset: {dataset}") from exc

    def resolve_path(self, dataset: str, stage: str) -> Path:
        """Resolve dataset path for a stage."""

        # TODO: Decide whether to allow absolute/external dataset paths.
        spec = self.get(dataset)
        return (self.config.resolved_data_root / stage / spec.path).resolve()
