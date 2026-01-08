from pathlib import Path

import pandas as pd
from data_pipeline.runner import Pipeline
from data_pipeline.types import DatasetSpec, PipelineConfig, RunContext


def test_run_writes_output_and_manifest(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        data_root=tmp_path / "data",
        runs_root=tmp_path / "runs",
        stages=("raw", "cleaned"),
        datasets={"users": DatasetSpec(format="csv", path="users/users.csv")},
    )
    p = Pipeline(cfg)

    raw_path = tmp_path / "data" / "raw" / "users" / "users.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [1, 2]}).to_csv(raw_path, index=False)

    def t(df: pd.DataFrame, ctx: RunContext) -> pd.DataFrame:
        return df.assign(x=df["x"] + 1)

    ctx = p.run(dataset="users", src_stage="raw", transform=t)

    out_path = tmp_path / "data" / "cleaned" / "users" / "users.csv"
    assert out_path.exists()
    out_df = pd.read_csv(out_path)
    assert out_df["x"].tolist() == [2, 3]

    manifest_path = tmp_path / "runs" / ctx.run_id / "manifest.json"
    assert manifest_path.exists()
