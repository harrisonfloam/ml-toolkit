# data-pipeline

Minimal staged data pipeline framework for running transforms between named stages (e.g. 'raw', 'cleaned'), with simple dataset config, IO helpers, and per-run manifests.

## Status

First draft of the Python API complete; future work includes CLI functionality, specialized pipelines for applying LLM transforms.

## Quickstart

```python
from pathlib import Path

import pandas as pd

from ml_toolkit.data_pipeline.runner import Pipeline
from ml_toolkit.data_pipeline.types import DatasetSpec, PipelineConfig, RunContext


cfg = PipelineConfig(
	data_root=Path("./data"),
	runs_root=Path("./runs"),
	stages=("raw", "cleaned"),
	datasets={"users": DatasetSpec(format="csv", path="users/users.csv")},
)

p = Pipeline(cfg)


def transform(df: pd.DataFrame, ctx: RunContext) -> pd.DataFrame:
	return df.assign(x=df["x"] + 1)


ctx = p.run(dataset="users", src_stage="raw", transform=transform)
print(ctx.output_path)
```
