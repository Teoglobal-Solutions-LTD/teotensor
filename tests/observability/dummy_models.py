"""Dummy estimators exercising the observability contract."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike
from teotensor.artifacts import (
    Diagnostic,
    FigureSpec,
    Report,
    TableSpec,
    Trace,
)
from teotensor.core import (
    BaseEstimator,
    ClusterMixin,
    check_array,
    check_is_fitted,
)


class DummyIterativeEstimator(BaseEstimator):
    """Toy iterative model that records a synthetic loss curve."""

    def __init__(self, n_iter: int = 5, learning_rate: float = 0.1) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        del y
        X_checked = check_array(X, dtype=np.float64)
        self.n_features_in_ = int(X_checked.shape[1])
        if self.n_iter < 1:
            msg = "n_iter must be >= 1."
            raise ValueError(msg)

        loss = float(np.mean(X_checked**2))
        records: list[dict[str, Any]] = []
        for step in range(1, self.n_iter + 1):
            loss = loss * (1.0 - self.learning_rate)
            records.append({"step": step, "loss": float(loss)})
        self.loss_curve_ = records
        self.final_loss_ = float(records[-1]["loss"])
        return self

    def report(self) -> Report:
        check_is_fitted(self, ("final_loss_", "loss_curve_"))
        return Report(
            title="DummyIterativeEstimator report",
            summary="Synthetic iterative fit completed.",
            metrics={
                "n_iter": self.n_iter,
                "final_loss": self.final_loss_,
                "n_features": self.n_features_in_,
            },
            figures=tuple(self.visualize()),
        )

    def diagnose(self) -> list[Diagnostic]:
        check_is_fitted(self, "final_loss_")
        findings: list[Diagnostic] = []
        if self.final_loss_ > 1.0:
            findings.append(
                Diagnostic(
                    code="high_final_loss",
                    severity="warning",
                    message="Final loss is above 1.0.",
                    context={"final_loss": self.final_loss_},
                )
            )
        else:
            findings.append(
                Diagnostic(
                    code="loss_ok",
                    severity="info",
                    message="Final loss looks reasonable for the toy model.",
                    context={"final_loss": self.final_loss_},
                )
            )
        return findings

    def trace(self) -> list[Trace]:
        check_is_fitted(self, "loss_curve_")
        return [
            Trace(
                name="loss",
                records=tuple(self.loss_curve_),
                metadata={"metric": "loss"},
            )
        ]

    def visualize(self) -> list[FigureSpec]:
        check_is_fitted(self, "loss_curve_")
        steps = [row["step"] for row in self.loss_curve_]
        losses = [row["loss"] for row in self.loss_curve_]
        return [
            FigureSpec(
                kind="line",
                title="Training loss",
                data={
                    "x": steps,
                    "y": losses,
                    "xlabel": "step",
                    "ylabel": "loss",
                },
            )
        ]


class DummyClusterObserver(BaseEstimator, ClusterMixin):
    """Toy clusterer with fake noise diagnostics."""

    def __init__(self, noise_label: int = -1) -> None:
        self.noise_label = noise_label

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        del y
        X_checked = check_array(X, dtype=np.float64)
        n_samples = int(X_checked.shape[0])
        self.n_features_in_ = int(X_checked.shape[1])
        labels = np.zeros(n_samples, dtype=np.int64)
        # Mark the last sample as noise when possible.
        if n_samples >= 2:
            labels[-1] = self.noise_label
        self.labels_ = labels
        return self

    def report(self) -> Report:
        check_is_fitted(self, "labels_")
        unique, counts = np.unique(self.labels_, return_counts=True)
        rows = tuple(
            (int(label), int(count))
            for label, count in zip(unique, counts, strict=True)
        )
        table = TableSpec(
            title="Cluster sizes",
            columns=("label", "count"),
            rows=rows,
        )
        n_noise = int(np.sum(self.labels_ == self.noise_label))
        return Report(
            title="DummyClusterObserver report",
            summary="Toy clustering finished.",
            metrics={
                "n_samples": int(self.labels_.shape[0]),
                "n_noise": n_noise,
                "n_clusters": int(np.sum(unique != self.noise_label)),
            },
            tables=(table,),
            figures=tuple(self.visualize()),
        )

    def diagnose(self) -> list[Diagnostic]:
        check_is_fitted(self, "labels_")
        n_samples = int(self.labels_.shape[0])
        n_noise = int(np.sum(self.labels_ == self.noise_label))
        ratio = float(n_noise / n_samples) if n_samples else 0.0
        if ratio > 0.0:
            return [
                Diagnostic(
                    code="noise_points_present",
                    severity="warning",
                    message="Some points were labeled as noise.",
                    context={
                        "n_noise": n_noise,
                        "noise_ratio": ratio,
                        "noise_label": self.noise_label,
                    },
                )
            ]
        return [
            Diagnostic(
                code="no_noise",
                severity="info",
                message="No noise points detected.",
                context={"noise_label": self.noise_label},
            )
        ]

    def visualize(self) -> list[FigureSpec]:
        check_is_fitted(self, "labels_")
        unique, counts = np.unique(self.labels_, return_counts=True)
        return [
            FigureSpec(
                kind="bar",
                title="Cluster size histogram",
                data={
                    "labels": [str(int(v)) for v in unique],
                    "y": [int(c) for c in counts],
                    "xlabel": "label",
                    "ylabel": "count",
                },
            )
        ]
