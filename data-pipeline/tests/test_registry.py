from pathlib import Path

from data_pipeline.registry import DatasetRegistry
from data_pipeline.types import DatasetSpec, PipelineConfig


def test_resolve_path(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        data_root=tmp_path / "data",
        runs_root=tmp_path / "runs",
        stages=("raw", "cleaned"),
        datasets={"users": DatasetSpec(format="csv", path="users/users.csv")},
    )
    reg = DatasetRegistry(cfg)
    p = reg.resolve_path("users", "raw")
    assert p == (tmp_path / "data" / "raw" / "users" / "users.csv").resolve()
