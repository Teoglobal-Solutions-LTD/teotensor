"""``.ttw`` files: a zip of a JSON manifest plus a NumPy ``arrays.npz``.

This is the inference file. It holds numbers and the settings needed to rebuild
the model. It does not hold a Python object, so opening it does not run code.
The whole-model pickle (``save`` / ``load``) is a separate file for continuing
work in this process.

Only classes inside ``teotensor`` may be named in the manifest.
"""

from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from teotensor.core.mixins import _is_estimator


def encode_params(value: Any) -> Any:
    """Turn constructor settings into JSON-friendly data.

    Components become ``{"__component__": "module.Class", "params": {...}}``.
    Tuples and lists keep their type. Numbers, strings, and ``None`` pass
    through.
    """
    if _is_estimator(value):
        qualname = f"{type(value).__module__}.{type(value).__qualname__}"
        params = {
            key: encode_params(item)
            for key, item in value.get_params(deep=False).items()
        }
        return {"__component__": qualname, "params": params}
    if isinstance(value, tuple):
        return {"__tuple__": [encode_params(item) for item in value]}
    if isinstance(value, list):
        return {"__list__": [encode_params(item) for item in value]}
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): encode_params(item) for key, item in value.items()}
    msg = f"Cannot store parameter of type {type(value).__name__} in a .ttw file."
    raise TypeError(msg)


def decode_params(value: Any) -> Any:
    """Reverse :func:`encode_params`."""
    if isinstance(value, dict) and "__component__" in value:
        cls = _load_teotensor_class(str(value["__component__"]))
        params = {key: decode_params(item) for key, item in value["params"].items()}
        return cls(**params)
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(decode_params(item) for item in value["__tuple__"])
    if isinstance(value, dict) and "__list__" in value:
        return [decode_params(item) for item in value["__list__"]]
    if isinstance(value, dict):
        return {str(key): decode_params(item) for key, item in value.items()}
    return value


def write_ttw(
    path: str | Path,
    *,
    model_module: str,
    model_name: str,
    params: dict[str, Any],
    meta: dict[str, Any],
    arrays: dict[str, NDArray[Any]],
) -> Path:
    """Write a ``.ttw`` zip.

    Parameters
    ----------
    path : path
        Destination. Parent folders are created.
    model_module, model_name : str
        Import path of the model class. Must start with ``teotensor.``.
    params : dict
        Encoded constructor settings.
    meta : dict
        Fitted facts that are not arrays (sizes, task name, convergence).
    arrays : dict of ndarray
        Named numeric payloads.

    Returns
    -------
    Path
        Absolute path written.
    """
    if not model_module.startswith("teotensor."):
        msg = f"Refusing to write a model outside teotensor: {model_module}."
        raise ValueError(msg)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "teotensor-weights",
        "version": 1,
        "model_module": model_module,
        "model_name": model_name,
        "params": params,
        "meta": meta,
        "arrays": {
            name: {
                "shape": list(np.asarray(array).shape),
                "dtype": str(np.asarray(array).dtype),
            }
            for name, array in arrays.items()
        },
    }
    arrays_path_name = "arrays.npz"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        # np.savez wants a real file or a BytesIO.
        import io

        buffer = io.BytesIO()
        # NumPy 2.5 stubs type savez(**kwargs) as bool. The values are arrays.
        packed = {name: np.asarray(array) for name, array in arrays.items()}
        np.savez(buffer, **cast(Any, packed))
        archive.writestr(arrays_path_name, buffer.getvalue())
    return target.resolve()


def read_ttw(path: str | Path) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Read a ``.ttw`` zip into ``(manifest, arrays)``.

    Raises
    ------
    ValueError
        If the format marker is missing or the model lives outside teotensor.
    """
    target = Path(path)
    with zipfile.ZipFile(target, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != "teotensor-weights":
            msg = f"{target} is not a TeoTensor weights file."
            raise ValueError(msg)
        module_name = str(manifest.get("model_module", ""))
        if not module_name.startswith("teotensor."):
            msg = f"Refusing to load a model outside teotensor: {module_name}."
            raise ValueError(msg)
        with archive.open("arrays.npz") as handle:
            loaded = np.load(handle)
            arrays = {name: loaded[name] for name in loaded.files}
    return manifest, arrays


def load_model(path: str | Path) -> Any:
    """Rebuild whatever model a ``.ttw`` file names, via its ``load_weights``."""
    manifest, _arrays = read_ttw(path)
    qualname = f"{manifest['model_module']}.{manifest['model_name']}"
    cls = _load_teotensor_class(qualname)
    return cls.load_weights(path)


def _load_teotensor_class(qualname: str) -> type[Any]:
    """Import ``package.module.Class`` if and only if it is inside teotensor."""
    module_name, separator, class_name = qualname.rpartition(".")
    if not separator or not module_name.startswith("teotensor."):
        msg = f"Refusing to import {qualname!r}."
        raise ValueError(msg)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type):
        msg = f"{qualname} is not a class."
        raise TypeError(msg)
    return cls
