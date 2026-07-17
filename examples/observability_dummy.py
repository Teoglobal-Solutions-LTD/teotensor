"""Demonstrate TeoTensor observability with an HTML dashboard.

Run from the repository root after ``pip install -e .``::

    python examples/observability_dummy.py

The script opens a self-contained HTML dashboard in your default browser.
"""

from __future__ import annotations

import numpy as np
from teotensor.artifacts import (
    Diagnostic,
    FigureSpec,
    Report,
    TableSpec,
    Trace,
    export_html,
)
from teotensor.core import (
    BaseEstimator,
    ClusterMixin,
    check_array,
    check_is_fitted,
)


class LossToy(BaseEstimator):
    """Minimal iterative estimator with a loss trace."""

    def __init__(self, n_iter: int = 5) -> None:
        self.n_iter = n_iter

    def fit(self, X, y=None):
        del y
        X = check_array(X, dtype=np.float64)
        loss = float(np.mean(X**2))
        curve = []
        for step in range(1, self.n_iter + 1):
            loss *= 0.8
            curve.append({"step": step, "loss": loss})
        self.loss_curve_ = curve
        self.final_loss_ = curve[-1]["loss"]
        return self

    def report(self):
        check_is_fitted(self, "final_loss_")
        return Report(
            title="LossToy report",
            summary="Toy iterative fit with a synthetic loss curve.",
            metrics={"final_loss": self.final_loss_, "n_iter": self.n_iter},
            figures=tuple(self.visualize()),
        )

    def diagnose(self):
        check_is_fitted(self, "final_loss_")
        return [
            Diagnostic(
                code="ok",
                severity="info",
                message="Training finished.",
                context={"final_loss": self.final_loss_},
            )
        ]

    def trace(self):
        check_is_fitted(self, "loss_curve_")
        return [Trace(name="loss", records=tuple(self.loss_curve_))]

    def visualize(self):
        check_is_fitted(self, "loss_curve_")
        return [
            FigureSpec(
                kind="line",
                title="Loss curve",
                data={
                    "x": [r["step"] for r in self.loss_curve_],
                    "y": [r["loss"] for r in self.loss_curve_],
                    "xlabel": "step",
                    "ylabel": "loss",
                },
            )
        ]


class ClusterToy(BaseEstimator, ClusterMixin):
    """Minimal clusterer with a noise diagnostic."""

    def fit(self, X, y=None):
        del y
        X = check_array(X)
        labels = np.zeros(X.shape[0], dtype=int)
        if X.shape[0] >= 2:
            labels[-1] = -1
        self.labels_ = labels
        return self

    def report(self):
        check_is_fitted(self, "labels_")
        unique, counts = np.unique(self.labels_, return_counts=True)
        return Report(
            title="ClusterToy report",
            summary="Toy clustering with one synthetic noise point.",
            metrics={"n_noise": int(np.sum(self.labels_ == -1))},
            tables=(
                TableSpec(
                    title="Sizes",
                    columns=("label", "count"),
                    rows=tuple(
                        (int(a), int(b)) for a, b in zip(unique, counts, strict=True)
                    ),
                ),
            ),
            figures=tuple(self.visualize()),
        )

    def diagnose(self):
        check_is_fitted(self, "labels_")
        n_noise = int(np.sum(self.labels_ == -1))
        return [
            Diagnostic(
                code="noise_points_present" if n_noise else "no_noise",
                severity="warning" if n_noise else "info",
                message=f"noise count = {n_noise}",
            )
        ]

    def visualize(self):
        check_is_fitted(self, "labels_")
        unique, counts = np.unique(self.labels_, return_counts=True)
        return [
            FigureSpec(
                kind="bar",
                title="Cluster sizes",
                data={
                    "labels": [str(int(v)) for v in unique],
                    "y": [int(c) for c in counts],
                    "xlabel": "label",
                    "ylabel": "count",
                },
            )
        ]


def main() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    iterative = LossToy(n_iter=4).fit(X)
    observation = iterative.observe()
    export_html(observation, open_browser=True)
    print("Opened LossToy observation dashboard in your browser.")

    cluster = ClusterToy().fit(X)
    export_html(cluster.observe(), open_browser=True)
    print("Opened ClusterToy observation dashboard in your browser.")


if __name__ == "__main__":
    main()
