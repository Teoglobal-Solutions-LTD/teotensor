"""Which models the studio knows how to build.

Adding K-Means later is one new :class:`ModelSpec`. The studio shell does not
learn that algorithm's math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from teotensor.classical.decomposition.pca import PCA
from teotensor.nn.estimators.mlp import MLPClassifier, MLPRegressor


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the studio's model list.

    Parameters
    ----------
    name : str
        Stable id used by the page.
    title : str
        Label shown to a person.
    estimator_cls : type
        Class constructed from the form. Not an instance.
    requires_y : bool
        True when the model needs a target column.
    iterative : bool
        True when "one more pass" is meaningful. PCA just fits once.
    description : str
        One sentence under the name.
    """

    name: str
    title: str
    estimator_cls: type[Any]
    requires_y: bool
    iterative: bool
    description: str


def model_registry() -> tuple[ModelSpec, ...]:
    """Return the models the studio can run today."""
    return (
        ModelSpec(
            name="pca",
            title="PCA",
            estimator_cls=PCA,
            requires_y=False,
            iterative=False,
            description="Find the axes that keep the most spread in a table.",
        ),
        ModelSpec(
            name="mlp_classifier",
            title="MLP classifier",
            estimator_cls=MLPClassifier,
            requires_y=True,
            iterative=True,
            description="A stack of neurons that names a class for each row.",
        ),
        ModelSpec(
            name="mlp_regressor",
            title="MLP regressor",
            estimator_cls=MLPRegressor,
            requires_y=True,
            iterative=True,
            description="A stack of neurons that predicts numbers.",
        ),
    )


def spec_by_name(name: str) -> ModelSpec:
    """Return the spec called ``name``.

    Raises
    ------
    KeyError
        If the studio does not know that model.
    """
    for spec in model_registry():
        if spec.name == name:
            return spec
    msg = f"Unknown model {name!r}."
    raise KeyError(msg)
