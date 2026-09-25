"""Read a CSV table into a :class:`~teotensor.data.dataset.Dataset`.

The first row is column names. One named column may be the target. Every
other column has to be numeric.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from teotensor.data.dataset import Dataset


def read_csv(source: str | Path, *, target: str | None = None) -> Dataset:
    """Load ``source`` as text or as a path to a text file.

    Parameters
    ----------
    source : str or Path
        CSV text, or a path to a file that exists.
    target : str or None, optional
        Column kept as ``targets``. ``None`` or ``""`` leaves every column
        in ``features``.
    """
    path = Path(source)
    text = path.read_text(encoding="utf-8") if path.is_file() else str(source)
    rows = [
        row
        for row in csv.reader(io.StringIO(text))
        if any(cell.strip() for cell in row)
    ]
    if len(rows) < 2:
        msg = "CSV needs a header and at least one data row."
        raise ValueError(msg)
    header = [cell.strip() for cell in rows[0]]
    body = rows[1:]
    columns: dict[str, list[str]] = {name: [] for name in header}
    for row in body:
        if len(row) != len(header):
            msg = "Every CSV row must have the same number of columns."
            raise ValueError(msg)
        for name, cell in zip(header, row, strict=True):
            columns[name].append(cell.strip())
    target_name = (target or "").strip() or None
    targets: NDArray[np.generic] | None = None
    if target_name is not None:
        if target_name not in columns:
            msg = f"Target column {target_name!r} is not in the header."
            raise ValueError(msg)
        targets = _as_target(columns.pop(target_name))
    if not columns:
        msg = "The table has no feature columns."
        raise ValueError(msg)
    names = tuple(columns)
    matrix = np.column_stack(
        [_as_float(values, name) for name, values in columns.items()]
    )
    return Dataset(
        features=np.asarray(matrix, dtype=np.float64),
        targets=targets,
        feature_names=names,
        target_name=target_name,
    )


def _as_float(values: list[str], name: str) -> NDArray[np.float64]:
    """Parse one feature column."""
    try:
        return np.asarray([float(value) for value in values], dtype=np.float64)
    except ValueError as exc:
        msg = f"Column {name!r} must be numeric."
        raise ValueError(msg) from exc


def _as_target(values: list[str]) -> NDArray[np.generic]:
    """Parse the answer column. Numbers stay numbers; other text stays text."""
    try:
        return np.asarray([float(value) for value in values], dtype=np.float64)
    except ValueError:
        return np.asarray(values)
