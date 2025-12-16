from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.types import DatasetSpec


def load(spec: DatasetSpec, path: Path) -> Any:
    """Load an object from disk based on dataset format."""

    fmt = spec.format.lower()
    match fmt:
        case "csv":
            return pd.read_csv(path)
        case "parquet":
            return pd.read_parquet(path)

    # TODO: Support jsonl/txt/dir formats
    # TODO: handle in memory datasets
    raise ValueError(f"Unsupported dataset format: {spec.format}")


def save(
    spec: DatasetSpec,
    path: Path,
    obj: Any,
    *,
    overwrite: bool = True,
    atomic: bool = True,
) -> None:
    """Save an object to disk based on dataset format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    fmt = spec.format.lower()
    match fmt:
        case "csv":
            _write_atomic(path, lambda p: obj.to_csv(p, index=False), atomic=atomic)
            return
        case "parquet":
            _write_atomic(path, lambda p: obj.to_parquet(p, index=False), atomic=atomic)
            return

    # TODO: Support jsonl/txt/dir formats.
    raise ValueError(f"Unsupported dataset format: {spec.format}")


def _write_atomic(path: Path, writer, *, atomic: bool) -> None:
    """Write to a temp file and replace the destination."""

    if not atomic:
        writer(path)
        return

    # TODO: Revisit atomic write guarantees across filesystems/platforms.
    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".", dir=str(path.parent)
        )
        tmp_path = Path(tmp_name)
        writer(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_fd is not None:
            try:
                import os

                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None and tmp_path.exists() and tmp_path != path:
            try:
                tmp_path.unlink()
            except OSError:
                pass
