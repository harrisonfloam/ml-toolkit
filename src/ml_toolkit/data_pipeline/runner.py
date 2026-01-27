from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from . import io
from .config import load_config
from .registry import DatasetRegistry
from .types import PipelineConfig, RunContext, TransformFn, transform_name


class Pipeline:
    """Run transforms between pipeline stages."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.registry = DatasetRegistry(config)
        self.logger = logging.getLogger("pipeline")

    @classmethod
    def from_file(cls, path: str | Path) -> "Pipeline":
        """Create a Pipeline from a YAML config file."""

        return cls(load_config(path))

    def run(
        self,
        *,
        dataset: str,
        src_stage: str,
        dst_stage: str | None = None,
        transform: TransformFn,
        params: Mapping[str, Any] | None = None,
        overwrite: bool = True,
        overwrite_manifest: bool = False,
        run_id: str | None = None,
    ) -> RunContext:
        """Run a transform and write output + a manifest."""

        # TODO: Consider better editor/type-checker hints for dataset/stage names.
        # In general we can't statically type YAML-defined strings without codegen
        # (e.g., generating a stub module with Literals) or config-as-code.

        params = params or {}
        if dst_stage is None:
            dst_stage = self.config.next_stage(src_stage)

        if src_stage not in self.config.stages:
            raise ValueError(f"Unknown src_stage: {src_stage}")
        if dst_stage not in self.config.stages:
            raise ValueError(f"Unknown dst_stage: {dst_stage}")

        src_index = self.config.stages.index(src_stage)
        dst_index = self.config.stages.index(dst_stage)
        if dst_index < src_index:
            self.logger.warning(
                "Running backwards in stages: %s -> %s", src_stage, dst_stage
            )
        if abs(dst_index - src_index) > 1:
            self.logger.warning(
                "Skipping intermediate stages: %s -> %s", src_stage, dst_stage
            )

        input_path = self.registry.resolve_path(dataset, src_stage)
        output_path = self.registry.resolve_path(dataset, dst_stage)
        if src_stage == dst_stage:
            if not overwrite:
                raise ValueError(
                    "src_stage == dst_stage requires overwrite=True (in-place overwrite)"
                )
            output_path = input_path

        dataset_spec = self.registry.get(dataset)

        self.logger.info("Running dataset=%s %s -> %s", dataset, src_stage, dst_stage)
        self.logger.info("Input: %s", input_path)
        self.logger.info("Output: %s", output_path)

        input_obj = io.load(dataset_spec, input_path)

        # TODO: Optional schema checks / stricter validation.
        run_id = run_id or _default_run_id()
        ctx = RunContext(
            dataset=dataset,
            src_stage=src_stage,
            dst_stage=dst_stage,
            run_id=run_id,
            params=params,
            input_path=input_path,
            output_path=output_path,
            logger=self.logger,
        )

        output_obj = transform(input_obj, ctx)

        io.save(dataset_spec, output_path, output_obj, overwrite=overwrite, atomic=True)

        self._write_manifest(
            ctx,
            transform=transform,
            overwrite_manifest=overwrite_manifest,
        )

        return ctx

    def _write_manifest(
        self,
        ctx: RunContext,
        *,
        transform: TransformFn,
        overwrite_manifest: bool,
    ) -> None:
        """Write a minimal run manifest."""

        runs_root = self.config.resolved_runs_root
        run_dir = runs_root / ctx.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists() and not overwrite_manifest:
            raise FileExistsError(
                f"Manifest already exists: {manifest_path} (set overwrite_manifest=True)"
            )

        payload = _build_manifest_payload(ctx, transform=transform)

        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _build_manifest_payload(
    ctx: RunContext, *, transform: TransformFn
) -> dict[str, Any]:
    """Build the run manifest payload."""

    # TODO: If/when we add a richer metadata model, this can become a dataclass.
    return {
        "run_id": ctx.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": ctx.dataset,
        "src_stage": ctx.src_stage,
        "dst_stage": ctx.dst_stage,
        "input_path": str(ctx.input_path),
        "output_path": str(ctx.output_path),
        "transform": transform_name(transform),
        "params": dict(ctx.params),
        "python_version": platform.python_version(),
    }


def _default_run_id() -> str:
    """Create a readable run id."""

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{now}-{suffix}"
