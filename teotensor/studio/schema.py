"""Turn ``get_params`` into form fields and apply the form back onto a model.

The form is not a second list of settings. It reads the constructor. A blank
component means "use the documented default at fit time".
"""

from __future__ import annotations

from typing import Any

from teotensor.core.mixins import _is_estimator
from teotensor.nn.activations import ACTIVATIONS, ReLU
from teotensor.nn.init import (
    AutoInit,
    KaimingNormal,
    KaimingUniform,
    LecunNormal,
    Orthogonal,
    XavierNormal,
    XavierUniform,
    Zeros,
)
from teotensor.nn.losses import LOSSES
from teotensor.nn.optim.adam import Adam
from teotensor.nn.optim.rmsprop import RMSprop
from teotensor.nn.optim.schedule import SCHEDULES, Constant
from teotensor.nn.optim.sgd import SGD
from teotensor.nn.penalty import L2, PENALTIES

# Dropdowns. The blank option means the constructor value stays None.
COMPONENT_OPTIONS: dict[str, dict[str, type[Any]]] = {
    "activation": ACTIVATIONS,
    "solver": {"Adam": Adam, "SGD": SGD, "RMSprop": RMSprop},
    "schedule": SCHEDULES,
    "penalty": PENALTIES,
    "weight_init": {
        "AutoInit": AutoInit,
        "XavierUniform": XavierUniform,
        "XavierNormal": XavierNormal,
        "KaimingUniform": KaimingUniform,
        "KaimingNormal": KaimingNormal,
        "LecunNormal": LecunNormal,
        "Orthogonal": Orthogonal,
        "Zeros": Zeros,
    },
    "bias_init": {"Zeros": Zeros},
    "loss": LOSSES,
}

_PLACEHOLDERS: dict[str, str] = {
    "activation": "ReLU",
    "solver": "Adam",
    "schedule": "Constant",
    "penalty": "L2",
    "weight_init": "AutoInit",
    "bias_init": "Zeros",
}


_LAYER_KEYS = {"hidden_layer_sizes", "activation", "hidden_activations"}


def form_fields(estimator: Any) -> list[dict[str, Any]]:
    """Describe editable constructor fields of ``estimator``."""
    fields: list[dict[str, Any]] = []
    for key, value in estimator.get_params(deep=False).items():
        # The studio always trains with a bias. ``bias=False`` stays a
        # constructor choice, not a form switch.
        if key == "bias":
            continue
        if key in _LAYER_KEYS:
            if key == "hidden_layer_sizes":
                fields.append(_layers_field(estimator))
            continue
        if _is_estimator(value):
            fields.append(_component_field(key, type(value).__name__))
            fields.extend(_leaves(key, value))
        elif value is None and key in COMPONENT_OPTIONS:
            fields.append(_component_field(key, ""))
        elif isinstance(value, tuple) and all(isinstance(item, int) for item in value):
            fields.append(
                {"key": key, "kind": "ints", "value": list(value), "label": key}
            )
        elif isinstance(value, bool):
            fields.append({"key": key, "kind": "bool", "value": value, "label": key})
        elif isinstance(value, int):
            fields.append({"key": key, "kind": "int", "value": value, "label": key})
        elif isinstance(value, float):
            fields.append({"key": key, "kind": "float", "value": value, "label": key})
        elif isinstance(value, str):
            fields.append({"key": key, "kind": "str", "value": value, "label": key})
        elif value is None:
            fields.append({"key": key, "kind": "float", "value": "", "label": key})
    return fields


def apply_fields(estimator: Any, fields: list[dict[str, Any]]) -> Any:
    """Write form fields onto ``estimator`` and return it.

    Component choices are applied before their nested numbers, so
    ``solver=Adam`` exists before ``solver__lr`` is set.
    """
    updates: dict[str, Any] = {}
    for field in fields:
        key = str(field["key"])
        kind = str(field["kind"])
        raw = field.get("value")
        if kind == "component":
            choice = str(raw or "")
            if choice == "":
                updates[key] = None
            else:
                options = COMPONENT_OPTIONS.get(key, {})
                if choice not in options:
                    msg = f"Unknown choice {choice!r} for {key}."
                    raise ValueError(msg)
                updates[key] = options[choice]()
        elif kind == "layers":
            _apply_layers(updates, raw)
        elif kind == "ints":
            if isinstance(raw, str):
                parts = [part.strip() for part in raw.split(",") if part.strip()]
                updates[key] = tuple(int(part) for part in parts)
            elif isinstance(raw, list):
                updates[key] = tuple(int(part) for part in raw)
            else:
                msg = f"Field {key} needs a list of integers."
                raise TypeError(msg)
        elif kind == "bool":
            updates[key] = bool(raw)
        elif kind == "int":
            if raw is None or raw == "":
                msg = f"Field {key} needs an integer."
                raise ValueError(msg)
            updates[key] = int(raw)
        elif kind == "float":
            if raw == "" or raw is None:
                updates[key] = None
            else:
                updates[key] = float(raw)
        elif kind == "str":
            updates[key] = str(raw)
    estimator.set_params(**updates)
    return estimator


def _layers_field(estimator: Any) -> dict[str, Any]:
    """One row per hidden layer: neuron count and activation name."""
    sizes = tuple(int(size) for size in estimator.hidden_layer_sizes)
    chosen = estimator.hidden_activations
    if chosen is None:
        name = (
            "ReLU"
            if estimator.activation is None
            else type(estimator.activation).__name__
        )
        names = [name] * len(sizes)
    else:
        names = [type(item).__name__ for item in chosen]
        if len(names) != len(sizes):
            names = (names + ["ReLU"] * len(sizes))[: len(sizes)]
    rows = [
        {"neurons": size, "activation": act}
        for size, act in zip(sizes, names, strict=True)
    ]
    return {
        "key": "layers",
        "kind": "layers",
        "label": "Hidden layers",
        "options": list(ACTIVATIONS),
        "value": rows,
    }


def _apply_layers(updates: dict[str, Any], raw: Any) -> None:
    """Write hidden width and per-layer activations from the row editor."""
    if not isinstance(raw, list):
        msg = "Hidden layers need a list of rows."
        raise TypeError(msg)
    sizes: list[int] = []
    names: list[str] = []
    for row in raw:
        count = int(row["neurons"])
        if count < 1:
            msg = "Each hidden layer needs at least 1 neuron."
            raise ValueError(msg)
        name = str(row["activation"])
        if name not in ACTIVATIONS:
            msg = f"Unknown activation {name!r}."
            raise ValueError(msg)
        sizes.append(count)
        names.append(name)
    updates["hidden_layer_sizes"] = tuple(sizes)
    if names and len(set(names)) == 1:
        updates["activation"] = ACTIVATIONS[names[0]]()
        updates["hidden_activations"] = None
    elif names:
        updates["activation"] = ACTIVATIONS[names[0]]()
        updates["hidden_activations"] = tuple(ACTIVATIONS[name]() for name in names)
    else:
        updates["activation"] = None
        updates["hidden_activations"] = None


def _component_field(key: str, value: str) -> dict[str, Any]:
    """One dropdown field."""
    options = list(COMPONENT_OPTIONS.get(key, {}))
    return {
        "key": key,
        "kind": "component",
        "value": value,
        "options": options,
        "placeholder": _PLACEHOLDERS.get(key, ""),
        "label": key,
    }


def _leaves(prefix: str, component: Any) -> list[dict[str, Any]]:
    """Scalar fields stored on a component."""
    found: list[dict[str, Any]] = []
    for key, value in component.get_params(deep=False).items():
        full = f"{prefix}__{key}"
        if isinstance(value, bool):
            found.append({"key": full, "kind": "bool", "value": value, "label": full})
        elif isinstance(value, int):
            found.append({"key": full, "kind": "int", "value": value, "label": full})
        elif isinstance(value, float):
            found.append({"key": full, "kind": "float", "value": value, "label": full})
    return found


# Imported for the placeholder map's documented defaults. Keeping the names
# referenced avoids a "unused import" if a checker only sees the dict values.
_DOCUMENTED = (ReLU, Constant, L2, AutoInit, Zeros, XavierUniform, KaimingNormal)
