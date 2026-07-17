"""Helpers for turning nested artifact values into JSON-friendly data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert NumPy scalars/arrays and nested structures into JSON data.

    Parameters
    ----------
    value : Any
        Arbitrary nested value.

    Returns
    -------
    Any
        A structure composed of ``dict``, ``list``, and JSON scalars.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())
    msg = f"Object of type {type(value).__name__} is not JSON-serializable."
    raise TypeError(msg)
