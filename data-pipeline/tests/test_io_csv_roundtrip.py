from pathlib import Path

import pandas as pd

from pipeline.io import load, save
from pipeline.types import DatasetSpec


def test_csv_round_trip(tmp_path: Path) -> None:
    spec = DatasetSpec(format="csv", path="x.csv")
    out_path = tmp_path / "x.csv"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    save(spec, out_path, df, overwrite=True)
    df2 = load(spec, out_path)
    pd.testing.assert_frame_equal(df, df2)
