"""Train a small XOR network and write its HTML report.

Run from the repository root:

    python examples/mlp_dashboard.py

To open the studio, call ``open_studio`` from ``teotensor.studio``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from teotensor.artifacts import export_html
from teotensor.nn import MLPClassifier
from teotensor.nn.activations import Tanh
from teotensor.nn.optim.adam import Adam
from teotensor.nn.penalty import L2


def main() -> None:
    """Fit XOR and write ``mlp_report.html`` next to this file."""
    features = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 1, 1, 0])
    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation=Tanh(),
        solver=Adam(lr=0.08),
        penalty=L2(alpha=0.0),
        max_iter=400,
        batch_size=4,
        validation_fraction=0.0,
        random_state=0,
    )
    model.fit(features, labels)
    destination = Path(__file__).resolve().parent / "mlp_report.html"
    export_html(model.observe(), destination)
    print(f"accuracy={model.score(features, labels):.3f}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
