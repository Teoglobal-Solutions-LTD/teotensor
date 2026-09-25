"""Multilayer perceptron classifier and regressor.

Settings that have their own fields (activation, solver, penalty, schedule,
initializer, loss) are components. ``None`` means the documented default and
is resolved inside ``fit``, on a copy, so the object passed by the caller does
not collect training memory.

The learned weights live on ``module_``. ``coefs_`` and ``intercepts_`` read
those weights; they are not a second copy kept in sync by hand.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from teotensor.artifacts.playback import Playback
from teotensor.artifacts.types import Diagnostic, FigureSpec, Report, TableSpec, Trace
from teotensor.artifacts.weights import (
    decode_params,
    encode_params,
    read_ttw,
    write_ttw,
)
from teotensor.core.base import BaseEstimator
from teotensor.core.mixins import (
    ClassifierMixin,
    RegressorMixin,
    SerializableMixin,
    clone,
)
from teotensor.core.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)
from teotensor.data.dataset import Dataset
from teotensor.nn.activations import Activation, Identity, ReLU
from teotensor.nn.estimators.playback import build_playback
from teotensor.nn.init import AutoInit, Initializer, Zeros, initializer_name
from teotensor.nn.layers.activations import ActivationLayer
from teotensor.nn.layers.dropout import Dropout
from teotensor.nn.layers.linear import Linear
from teotensor.nn.layers.sequential import Sequential
from teotensor.nn.losses import MAE, MSE, CrossEntropy, Huber, Loss
from teotensor.nn.optim.adam import Adam
from teotensor.nn.optim.schedule import Constant, Schedule
from teotensor.nn.optim.sgd import Optimizer
from teotensor.nn.penalty import L2, Penalty
from teotensor.nn.tensor.core import Tensor
from teotensor.nn.training.loop import EpochStat, TrainResult, fit_module

FloatArray = NDArray[np.float64]
Task = Literal["classifier", "regressor"]

# Diagnostic thresholds. They are not hyperparameters.
_EXPLODING_GRAD = 100.0
_VANISHING_GRAD = 1e-6
_DEAD_FRACTION = 0.5
_SATURATED_FRACTION = 0.5
_OVERFIT_RATIO = 0.5
_LARGE_WEIGHT = 10.0
_IMBALANCE_RATIO = 0.2


class _MLPBase(SerializableMixin, BaseEstimator):
    """Shared training, inspection, and weight export for both MLP tasks."""

    # Filled by ``fit`` / ``_prepare_targets``. Declared so the base methods
    # that read them type-check before a subclass assigns them.
    _encoded_: NDArray[np.generic]
    _sample_weight_: FloatArray | None
    class_counts_: list[int]
    playbacks_: dict[int, Playback]
    _epoch_rows_: list[EpochStat]

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        activation: Activation | None = None,
        hidden_activations: tuple[Activation, ...] | None = None,
        weight_init: Initializer | None = None,
        bias_init: Initializer | None = None,
        *,
        bias: bool = True,
        penalty: Penalty | None = None,
        dropout: float = 0.0,
        solver: Optimizer | None = None,
        schedule: Schedule | None = None,
        max_grad_norm: float | None = None,
        loss: Loss | None = None,
        class_weight: Literal["balanced"] | dict[Any, float] | None = None,
        max_iter: int = 200,
        batch_size: int | Literal["auto"] = "auto",
        shuffle: bool = True,
        tol: float = 1e-4,
        n_iter_no_change: int = 10,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        warm_start: bool = False,
        random_state: int | np.random.RandomState | np.random.Generator | None = None,
        inspect_every: int = 1,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.hidden_activations = hidden_activations
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.bias = bias
        self.penalty = penalty
        self.dropout = dropout
        self.solver = solver
        self.schedule = schedule
        self.max_grad_norm = max_grad_norm
        self.loss = loss
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.warm_start = warm_start
        self.random_state = random_state
        self.inspect_every = inspect_every

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        **fit_params: Any,
    ) -> _MLPBase:
        """Learn weights. With ``warm_start=False`` this starts over.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training examples.
        y : array-like
            Targets. Class labels for a classifier, numbers for a regressor.
        **fit_params : Any
            Accepted for API compatibility and ignored.

        Returns
        -------
        self
            Fitted model.
        """
        del fit_params
        return self._fit(
            X, y, classes=None, epochs=self.max_iter, continuing=self.warm_start
        )

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        classes: ArrayLike | None = None,
        *,
        epochs: int | None = None,
    ) -> _MLPBase:
        """Keep the current weights and train for more passes.

        Parameters
        ----------
        X, y : array-like
            Another batch of examples. On the first call this also builds the
            network.
        classes : array-like or None, optional
            Every class the classifier will ever see. Required on the first
            call when this batch does not contain all of them. Ignored by the
            regressor.
        epochs : int or None, optional
            Passes to run. ``None`` uses ``max_iter``.

        Returns
        -------
        self
            Fitted model.
        """
        n_epochs = self.max_iter if epochs is None else epochs
        return self._fit(X, y, classes=classes, epochs=n_epochs, continuing=True)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Answer for new examples. Dropout is off."""
        logits = self._logits(X)
        return self._decode_prediction(logits)

    def playback(
        self,
        x: ArrayLike,
        y: ArrayLike | None = None,
        *,
        sample_index: int = 0,
        with_backward: bool = True,
    ) -> Playback:
        """Film one example. Backward needs a target.

        Parameters
        ----------
        x : array-like of shape (n_features,) or (1, n_features)
            One example.
        y : array-like or None, optional
            Target for that example. Required when ``with_backward`` is True
            unless the model is a classifier, in which case the predicted
            label is used and the caption still says the error is measured
            against that label.
        sample_index : int, default=0
            Number stored on the film.
        with_backward : bool, default=True
            Record the error walking back as well as the signal going forward.
        """
        check_is_fitted(self, "module_")
        row = check_array(
            np.asarray(x, dtype=np.float64).reshape(1, -1), dtype=np.float64
        )
        if row.shape[1] != self.n_features_in_:
            msg = f"x has {row.shape[1]} features; the model saw {self.n_features_in_}."
            raise ValueError(msg)
        target = self._playback_target(row, y, with_backward)
        assert isinstance(self.module_, Sequential)
        return build_playback(
            self.module_,
            row.reshape(-1),
            sample_index=sample_index,
            with_backward=with_backward,
            loss=self.loss_ if with_backward else None,
            target=target,
        )

    def report(self) -> Report:
        """Summarize architecture, last loss, and whether training settled."""
        check_is_fitted(self, "module_")
        n_params = sum(int(np.prod(coef.shape)) for coef in self.coefs_)
        n_params += sum(int(intercept.size) for intercept in self.intercepts_)
        rows = []
        for index, coef in enumerate(self.coefs_):
            rows.append(
                (
                    index,
                    "Linear",
                    int(coef.shape[0]),
                    int(coef.shape[1]),
                    self.activation_names_[index],
                    int(coef.size + self.intercepts_[index].size),
                    self.init_names_[index],
                )
            )
        summary = (
            f"{self._task()} network, {self.n_iter_} passes, "
            f"last training loss {self.loss_curve_[-1]:.4g}, "
            f"{'settled' if self.converged_ else 'stopped at max passes'}."
        )
        return Report(
            title=f"{type(self).__name__} report",
            summary=summary,
            metrics={
                "n_iter": int(self.n_iter_),
                "train_loss": float(self.loss_curve_[-1]),
                "best_loss": float(self.best_loss_),
                "train_score": float(self.score_curve_[-1]),
                "n_params": n_params,
                "converged": bool(self.converged_),
            },
            tables=(
                TableSpec(
                    title="Layers",
                    columns=(
                        "layer",
                        "type",
                        "in",
                        "out",
                        "activation",
                        "n_params",
                        "init",
                    ),
                    rows=tuple(rows),
                ),
            ),
            figures=tuple(self.visualize()),
        )

    def diagnose(self) -> list[Diagnostic]:
        """Read the same traces the charts use and flag trouble."""
        check_is_fitted(self, "traces_")
        findings: list[Diagnostic] = []
        losses = [float(row["train_loss"]) for row in self.traces_[0].records]
        if any(not np.isfinite(value) for value in losses):
            findings.append(
                Diagnostic(
                    code="nan_loss",
                    severity="error",
                    message="The loss became infinite or not-a-number.",
                    context={"last": losses[-1]},
                )
            )
        if not self.converged_:
            findings.append(
                Diagnostic(
                    code="not_converged",
                    severity="warning",
                    message=(
                        f"Training used all {self.n_iter_} passes without "
                        "the loss settling within tol."
                    ),
                    context={"n_iter": int(self.n_iter_), "tol": float(self.tol)},
                )
            )
        grad_norms = [
            float(row["grad_norm"]) for row in self._trace("grad_norm").records
        ]
        if (
            grad_norms
            and float(np.mean(grad_norms[-min(5, len(grad_norms)) :])) > _EXPLODING_GRAD
        ):
            findings.append(
                Diagnostic(
                    code="exploding_grad",
                    severity="warning",
                    message="The error signal is very large. Use a smaller step.",
                    context={"mean_grad_norm": float(np.mean(grad_norms[-5:]))},
                )
            )
        if len(grad_norms) >= 3 and all(
            value < _VANISHING_GRAD for value in grad_norms[-3:]
        ):
            findings.append(
                Diagnostic(
                    code="vanishing_grad",
                    severity="warning",
                    message="The error signal almost vanished for several passes.",
                    context={"grad_norm": grad_norms[-1]},
                )
            )
        dead = [
            row
            for row in self._trace("layers").records
            if int(row["epoch"]) == int(self.n_iter_)
            and float(row["dead_fraction"]) > _DEAD_FRACTION
        ]
        if dead:
            findings.append(
                Diagnostic(
                    code="dead_relu",
                    severity="warning",
                    message="More than half of a ReLU layer stays at zero.",
                    context={"layers": [int(row["layer"]) for row in dead]},
                )
            )
        saturated = [
            row
            for row in self._trace("layers").records
            if int(row["epoch"]) == int(self.n_iter_)
            and float(row["saturated_fraction"]) > _SATURATED_FRACTION
        ]
        if saturated:
            findings.append(
                Diagnostic(
                    code="saturated_activation",
                    severity="warning",
                    message="A tanh or sigmoid layer is pushed against its limits.",
                    context={"layers": [int(row["layer"]) for row in saturated]},
                )
            )
        if self._overfit():
            findings.append(
                Diagnostic(
                    code="overfit_gap",
                    severity="warning",
                    message="Validation loss stays well above training loss.",
                    context={},
                )
            )
        max_weight = max(float(np.max(np.abs(coef))) for coef in self.coefs_)
        if max_weight > _LARGE_WEIGHT:
            findings.append(
                Diagnostic(
                    code="large_weights",
                    severity="warning",
                    message="A weight grew past the usual range.",
                    context={"max_abs_weight": max_weight},
                )
            )
        if self._imbalanced():
            findings.append(
                Diagnostic(
                    code="class_imbalance",
                    severity="info",
                    message="One class is much rarer than another.",
                    context={"counts": list(self.class_counts_)},
                )
            )
        if not findings:
            findings.append(
                Diagnostic(
                    code="ok",
                    severity="info",
                    message="No training problems were flagged.",
                )
            )
        return findings

    def trace(self) -> list[Trace]:
        """Return the curves written during training."""
        check_is_fitted(self, "traces_")
        return list(self.traces_)

    def visualize(self) -> list[FigureSpec]:
        """Describe charts. This method does not draw them."""
        check_is_fitted(self, "traces_")
        figures: list[FigureSpec] = []
        loss_records = self._trace("loss").records
        figures.append(
            FigureSpec(
                kind="line",
                title="Loss",
                data={
                    "series": _series(loss_records, ("train_loss", "val_loss")),
                    "xlabel": "pass",
                    "ylabel": "loss",
                },
            )
        )
        figures.append(
            FigureSpec(
                kind="line",
                title="Score",
                data={
                    "series": _series(
                        self._trace("score").records, ("train_score", "val_score")
                    ),
                    "xlabel": "pass",
                    "ylabel": "score",
                },
            )
        )
        figures.append(
            FigureSpec(
                kind="line",
                title="Step length",
                data={
                    "x": [
                        int(row["epoch"])
                        for row in self._trace("learning_rate").records
                    ],
                    "y": [
                        float(row["learning_rate"])
                        for row in self._trace("learning_rate").records
                    ],
                    "xlabel": "pass",
                    "ylabel": "learning rate",
                },
            )
        )
        figures.append(
            FigureSpec(
                kind="line",
                title="Error-signal length",
                data={
                    "x": [
                        int(row["epoch"]) for row in self._trace("grad_norm").records
                    ],
                    "y": [
                        float(row["grad_norm"])
                        for row in self._trace("grad_norm").records
                    ],
                    "xlabel": "pass",
                    "ylabel": "gradient norm",
                },
            )
        )
        last_layers = [
            row
            for row in self._trace("layers").records
            if int(row["epoch"]) == int(self.n_iter_)
        ]
        if last_layers:
            figures.append(
                FigureSpec(
                    kind="bar",
                    title="Dead neurons by layer",
                    data={
                        "labels": [str(int(row["layer"])) for row in last_layers],
                        "y": [float(row["dead_fraction"]) for row in last_layers],
                        "xlabel": "layer",
                        "ylabel": "fraction near zero",
                    },
                )
            )
        weights = np.concatenate([coef.reshape(-1) for coef in self.coefs_])
        counts, edges = np.histogram(weights, bins=10)
        labels = [f"{edges[i]:.2g}" for i in range(len(counts))]
        figures.append(
            FigureSpec(
                kind="bar",
                title="Weight histogram",
                data={
                    "labels": labels,
                    "y": [int(count) for count in counts],
                    "xlabel": "weight bin start",
                    "ylabel": "count",
                },
            )
        )
        figures.append(_grad_heatmap(self._trace("layers").records))
        extra = self._extra_figures()
        figures.extend(extra)
        return figures

    def export_weights(self, path: str) -> str:
        """Write a ``.ttw`` file of numbers for later ``load_weights``."""
        check_is_fitted(self, "module_")
        arrays: dict[str, NDArray[Any]] = {}
        for index, (coef, intercept) in enumerate(
            zip(self.coefs_, self.intercepts_, strict=True)
        ):
            arrays[f"coef_{index}"] = coef
            arrays[f"intercept_{index}"] = intercept
        meta: dict[str, Any] = {
            "task": self._task(),
            "n_features_in": int(self.n_features_in_),
            "n_outputs": int(self.n_outputs_),
            "layer_sizes": list(self.layer_sizes_),
            "activation_names": list(self.activation_names_),
            "init_names": list(self.init_names_),
            "n_iter": int(self.n_iter_),
            "best_loss": float(self.best_loss_),
            "converged": bool(self.converged_),
            "loss_curve": [float(value) for value in self.loss_curve_],
            "score_curve": [float(value) for value in self.score_curve_],
        }
        meta.update(self._export_meta())
        written = write_ttw(
            path,
            model_module=type(self).__module__,
            model_name=type(self).__name__,
            params=encode_params(self.get_params(deep=False)),
            meta=meta,
            arrays=arrays,
        )
        return str(written)

    @classmethod
    def load_weights(cls, path: str) -> _MLPBase:
        """Rebuild a model from a ``.ttw`` file and leave it ready to answer."""
        manifest, arrays = read_ttw(path)
        params = decode_params(manifest["params"])
        model = cls(**params)
        model._restore_weights(manifest["meta"], arrays)
        return model

    @property
    def coefs_(self) -> list[FloatArray]:
        """Weight matrices, one per linear layer, copied from ``module_``."""
        check_is_fitted(self, "module_")
        return [
            np.array(layer.weight.data, dtype=np.float64, copy=True)
            for layer in _linears(self.module_)
        ]

    @property
    def intercepts_(self) -> list[FloatArray]:
        """Bias vectors. A layer without bias contributes zeros."""
        check_is_fitted(self, "module_")
        found: list[FloatArray] = []
        for layer in _linears(self.module_):
            if layer.bias is None:
                found.append(np.zeros(layer.out_features, dtype=np.float64))
            else:
                found.append(np.array(layer.bias.data, dtype=np.float64, copy=True))
        return found

    def _fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        classes: ArrayLike | None,
        epochs: int,
        continuing: bool,
    ) -> _MLPBase:
        """Shared body of ``fit`` and ``partial_fit``."""
        x_checked, y_checked = self._check_training_data(X, y)
        n_samples, n_features = x_checked.shape
        if not continuing and hasattr(self, "classes_"):
            del self.classes_
        self._prepare_targets(y_checked, classes)
        hidden = tuple(int(size) for size in self.hidden_layer_sizes)
        if any(size < 1 for size in hidden):
            msg = f"hidden_layer_sizes must be positive; got {hidden}."
            raise ValueError(msg)
        layout_ok = continuing and self._layout_matches(n_features, hidden)
        if not layout_ok:
            self._build(n_features, hidden)
            self.playbacks_ = {}
            self._epoch_rows_ = []
        else:
            self._sync_runtime()
            if not hasattr(self, "_epoch_rows_"):
                self._epoch_rows_ = []
            if not hasattr(self, "playbacks_"):
                self.playbacks_ = {}
        x_train, y_train, x_val, y_val, weights = self._split(x_checked)
        batch = n_samples if self.batch_size == "auto" else int(self.batch_size)
        if self.batch_size == "auto":
            batch = min(200, x_train.shape[0])
        if batch < 1:
            msg = "batch_size must be positive."
            raise ValueError(msg)
        assert self.module_ is not None
        assert self.loss_ is not None
        assert self.solver_ is not None
        assert self.schedule_ is not None

        def _snapshot(stat: EpochStat) -> None:
            if self.inspect_every <= 0 or stat.epoch % int(self.inspect_every) != 0:
                return
            target = y_train[0:1]
            film = build_playback(
                self.module_,
                x_train[0],
                sample_index=0,
                with_backward=True,
                loss=self.loss_,
                target=target,
            )
            self.playbacks_[int(stat.epoch)] = film

        result = fit_module(
            self.module_,
            loss=self.loss_,
            optimizer=self.solver_,
            schedule=self.schedule_,
            penalty=self.penalty_,
            x_train=x_train,
            y_train=y_train,
            sample_weight=weights,
            x_val=x_val,
            y_val=y_val,
            epochs=int(epochs),
            batch_size=batch,
            shuffle=bool(self.shuffle),
            tol=float(self.tol),
            n_iter_no_change=int(self.n_iter_no_change),
            early_stopping=bool(self.early_stopping),
            max_grad_norm=self.max_grad_norm,
            rng=self.random_state_,
            score_fn=self._score_arrays,
            on_epoch=_snapshot,
            start_epoch=len(self._epoch_rows_) + 1,
        )
        self._accept_result(result)
        self._remember_confusion(x_train, y_train)
        return self

    def _build(self, n_features: int, hidden: tuple[int, ...]) -> None:
        """Create the stack, the clones, and the random stream."""
        self.random_state_ = check_random_state(self.random_state)
        activation = clone(self.activation) if self.activation is not None else ReLU()
        if self.hidden_activations is None:
            activations = tuple(clone(activation) for _ in hidden)
        else:
            if len(self.hidden_activations) != len(hidden):
                msg = (
                    "hidden_activations must have one entry per hidden layer; "
                    f"got {len(self.hidden_activations)} for {len(hidden)} layers."
                )
                raise ValueError(msg)
            activations = tuple(clone(item) for item in self.hidden_activations)
        self.activation_ = activation
        self.hidden_activations_ = activations
        self.weight_init_ = (
            clone(self.weight_init) if self.weight_init is not None else AutoInit()
        )
        self.bias_init_ = (
            clone(self.bias_init) if self.bias_init is not None else Zeros()
        )
        self.solver_ = clone(self.solver) if self.solver is not None else Adam()
        self.schedule_ = (
            clone(self.schedule) if self.schedule is not None else Constant()
        )
        self.penalty_ = clone(self.penalty) if self.penalty is not None else L2()
        self.loss_ = clone(self.loss) if self.loss is not None else self._default_loss()
        if not 0.0 <= float(self.dropout) < 1.0:
            msg = f"dropout must satisfy 0 <= p < 1; got {self.dropout}."
            raise ValueError(msg)
        sizes = (n_features, *hidden, int(self.n_outputs_))
        layers: list[Any] = []
        init_names: list[str] = []
        activation_names: list[str] = []
        rng = self.random_state_
        for index in range(len(sizes) - 1):
            linear = Linear(sizes[index], sizes[index + 1], bias=bool(self.bias))
            act: Activation | None = activations[index] if index < len(hidden) else None
            linear.weight.data = self.weight_init_.sample(
                (sizes[index], sizes[index + 1]),
                rng,
                fan_in=sizes[index],
                fan_out=sizes[index + 1],
                activation=act,
            )
            if linear.bias is not None:
                linear.bias.data = self.bias_init_.sample(
                    (sizes[index + 1],),
                    rng,
                    fan_in=sizes[index],
                    fan_out=sizes[index + 1],
                    activation=act,
                )
            layers.append(linear)
            init_names.append(initializer_name(self.weight_init_, act))
            if act is not None:
                activation_names.append(type(act).__name__)
                layers.append(ActivationLayer(act))
                if float(self.dropout) > 0.0:
                    layers.append(Dropout(float(self.dropout), rng))
            else:
                activation_names.append("logits")
        self.module_ = Sequential(layers)
        self.n_features_in_ = n_features
        self.layer_sizes_ = sizes
        self.init_names_ = tuple(init_names)
        self.activation_names_ = tuple(activation_names)
        self._hidden_key_ = hidden
        self._dropout_used_ = float(self.dropout)
        self._bias_used_ = bool(self.bias)

    def _sync_runtime(self) -> None:
        """Copy constructor settings onto the running clones without wiping moments."""
        _sync_component(self.solver, self.solver_)
        _sync_component(self.schedule, self.schedule_)
        _sync_component(self.penalty, self.penalty_)
        _sync_component(self.loss, self.loss_)

    def _layout_matches(self, n_features: int, hidden: tuple[int, ...]) -> bool:
        """Return whether this stack can keep learning on this table."""
        if not hasattr(self, "module_"):
            return False
        return (
            int(self.n_features_in_) == n_features
            and tuple(self._hidden_key_) == hidden
            and int(self.n_outputs_) == int(self.layer_sizes_[-1])
            and bool(self._bias_used_) == bool(self.bias)
            and float(self._dropout_used_) == float(self.dropout)
        )

    def _split(
        self,
        x_checked: FloatArray,
    ) -> tuple[
        FloatArray,
        NDArray[np.generic],
        FloatArray | None,
        NDArray[np.generic] | None,
        FloatArray | None,
    ]:
        """Hold out the tail of the table when ``validation_fraction`` is set."""
        encoded = self._encoded_
        weights = self._sample_weight_
        names = tuple(f"x{index}" for index in range(x_checked.shape[1]))
        table = Dataset(
            features=x_checked,
            targets=encoded,
            sample_weight=weights,
            feature_names=names,
        )
        table.split_tail(float(self.validation_fraction))
        if self.early_stopping and int(table.val_idx.size) < 1:
            msg = (
                "early_stopping=True requires validation_fraction "
                "large enough to hold out a row."
            )
            raise ValueError(msg)
        x_train, y_train, w_train = table.take(table.train_idx)
        if y_train is None:
            msg = "Training split is missing targets."
            raise RuntimeError(msg)
        if int(table.val_idx.size) == 0:
            return x_train, y_train, None, None, w_train
        x_val, y_val, _weights = table.take(table.val_idx)
        if y_val is None:
            msg = "Validation split is missing targets."
            raise RuntimeError(msg)
        return x_train, y_train, x_val, y_val, w_train

    def _accept_result(self, result: TrainResult) -> None:
        """Store curves from a training call, appending when we continued."""
        for row in result.epochs:
            self._epoch_rows_.append(row)
        self.n_iter_ = len(self._epoch_rows_)
        self.converged_ = bool(result.converged)
        # Best watched loss across every pass, including earlier partial_fit calls.
        # A single fit_module call only remembers its own window.
        watched = [
            float(row.val_loss)
            if self.early_stopping and row.val_loss is not None
            else float(row.train_loss)
            for row in self._epoch_rows_
        ]
        self.best_loss_ = float(min(watched)) if watched else float(result.best_loss)
        self.loss_curve_ = tuple(row.train_loss for row in self._epoch_rows_)
        self.score_curve_ = tuple(row.train_score for row in self._epoch_rows_)
        self.traces_ = _pack_traces(self._epoch_rows_)
        self.module_.eval()

    def _remember_confusion(
        self,
        x_train: FloatArray,
        y_train: NDArray[np.generic],
    ) -> None:
        """Store a confusion matrix for classifiers and a small scatter sample."""
        self.confusion_matrix_ = None
        self._scatter_xy_ = None
        self._scatter_pred_ = None
        if self._task() == "classifier":
            logits = self.module_(Tensor(x_train)).data
            predicted = np.argmax(logits, axis=1)
            labels = np.asarray(y_train, dtype=np.int64).reshape(-1)
            n_classes = int(self.n_outputs_)
            matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
            for truth, guess in zip(labels, predicted, strict=True):
                matrix[int(truth), int(guess)] += 1
            self.confusion_matrix_ = matrix
        if x_train.shape[1] == 2:
            take = min(400, x_train.shape[0])
            self._scatter_xy_ = np.array(x_train[:take], copy=True)
            self._scatter_pred_ = np.asarray(self.predict(x_train[:take]))

    def _logits(self, X: ArrayLike) -> FloatArray:
        """Raw scores for ``X`` in exam mode."""
        check_is_fitted(self, "module_")
        x_checked = check_array(X, dtype=np.float64)
        if x_checked.shape[1] != self.n_features_in_:
            msg = (
                f"X has {x_checked.shape[1]} features; "
                f"the model saw {self.n_features_in_}."
            )
            raise ValueError(msg)
        self.module_.eval()
        return self.module_(Tensor(x_checked)).data

    def _trace(self, name: str) -> Trace:
        """Return one named trace."""
        for item in self.traces_:
            if item.name == name:
                return item
        msg = f"Missing trace {name!r}."
        raise KeyError(msg)

    def _overfit(self) -> bool:
        """Return whether validation loss sits well above training loss."""
        records = self._trace("loss").records
        gaps: list[float] = []
        for row in records[-5:]:
            if "val_loss" not in row:
                return False
            train = float(row["train_loss"])
            val = float(row["val_loss"])
            gaps.append((val - train) / max(abs(train), 1e-8))
        return len(gaps) >= 3 and all(gap > _OVERFIT_RATIO for gap in gaps)

    def _imbalanced(self) -> bool:
        """Return whether the rarest class is under a fifth of the largest."""
        counts = self.class_counts_
        if not counts:
            return False
        smallest = min(counts)
        largest = max(counts)
        if largest == 0:
            return False
        return (smallest / largest) < _IMBALANCE_RATIO

    def _check_training_data(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[FloatArray, NDArray[np.generic]]:
        """Validate ``X`` and ``y``. Subclasses narrow ``y``."""
        raise NotImplementedError

    def _prepare_targets(
        self, y: NDArray[np.generic], classes: ArrayLike | None
    ) -> None:
        """Set ``n_outputs_``, encoded targets, and optional sample weights."""
        raise NotImplementedError

    def _default_loss(self) -> Loss:
        """Loss used when the constructor loss is ``None``."""
        raise NotImplementedError

    def _task(self) -> str:
        """``classifier`` or ``regressor``."""
        raise NotImplementedError

    def _decode_prediction(self, logits: FloatArray) -> NDArray[Any]:
        """Turn raw scores into labels or numeric predictions."""
        raise NotImplementedError

    def _score_arrays(
        self, prediction: FloatArray, target: NDArray[np.generic]
    ) -> float:
        """Score one NumPy prediction against encoded targets."""
        raise NotImplementedError

    def _playback_target(
        self,
        row: FloatArray,
        y: ArrayLike | None,
        with_backward: bool,
    ) -> NDArray[np.generic] | None:
        """Encode the target used to film the backward pass."""
        raise NotImplementedError

    def _extra_figures(self) -> list[FigureSpec]:
        """Task-specific charts (confusion, scatter)."""
        return []

    def _export_meta(self) -> dict[str, Any]:
        """Extra JSON fields stored next to the weight arrays."""
        return {}

    def _restore_weights(
        self, meta: dict[str, Any], arrays: dict[str, NDArray[Any]]
    ) -> None:
        """Load numbers into a freshly constructed model."""
        self.n_outputs_ = int(meta["n_outputs"])
        hidden = tuple(self.hidden_layer_sizes)
        self._build(int(meta["n_features_in"]), hidden)
        linears = _linears(self.module_)
        for index, linear in enumerate(linears):
            linear.weight.data = np.asarray(arrays[f"coef_{index}"], dtype=np.float64)
            if linear.bias is not None:
                linear.bias.data = np.asarray(
                    arrays[f"intercept_{index}"], dtype=np.float64
                )
        self.activation_names_ = tuple(meta["activation_names"])
        self.init_names_ = tuple(meta["init_names"])
        self.n_iter_ = int(meta["n_iter"])
        self.best_loss_ = float(meta["best_loss"])
        self.converged_ = bool(meta["converged"])
        self.loss_curve_ = tuple(meta["loss_curve"])
        self.score_curve_ = tuple(meta["score_curve"])
        self._epoch_rows_ = []
        self.traces_ = (
            Trace(
                name="loss",
                records=tuple(
                    {"epoch": index + 1, "train_loss": float(value)}
                    for index, value in enumerate(self.loss_curve_)
                ),
            ),
            Trace(
                name="score",
                records=tuple(
                    {"epoch": index + 1, "train_score": float(value)}
                    for index, value in enumerate(self.score_curve_)
                ),
            ),
            Trace(name="learning_rate", records=()),
            Trace(name="grad_norm", records=()),
            Trace(name="layers", records=()),
        )
        self.playbacks_ = {}
        self.module_.eval()
        self._restore_task(meta)

    def _restore_task(self, meta: dict[str, Any]) -> None:
        """Restore labels or output shape after weights are loaded."""
        raise NotImplementedError


class MLPClassifier(ClassifierMixin, _MLPBase):
    """Classifier: one raw score per class, trained with cross-entropy.

    Two classes still use two scores. The loss turns scores into probabilities.
    There is no softmax layer on the network itself.
    """

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Class probabilities. Columns follow ``classes_``."""
        return _softmax(self._logits(X))

    def _check_training_data(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[FloatArray, NDArray[np.generic]]:
        """Check a classification table. Labels may be strings."""
        x_checked, y_checked = check_X_y(
            X,
            y,
            dtype=np.float64,
            y_numeric=False,
        )
        return x_checked, np.asarray(y_checked)

    def _prepare_targets(
        self, y: NDArray[np.generic], classes: ArrayLike | None
    ) -> None:
        """Map labels to ``0 .. n_classes-1`` and build per-row weights."""
        incoming = np.unique(y) if classes is None else np.asarray(classes)
        if not hasattr(self, "classes_"):
            self.classes_ = np.asarray(incoming)
        elif classes is not None and not np.array_equal(
            np.asarray(classes), self.classes_
        ):
            msg = "classes do not match the classes stored on the first partial_fit."
            raise ValueError(msg)
        known = np.asarray(self.classes_)
        flat = np.asarray(y).reshape(-1)
        if not np.isin(flat, known).all():
            msg = "y contains a label that is not in classes_."
            raise ValueError(msg)
        encoded = np.empty(flat.shape[0], dtype=np.int64)
        for index, label in enumerate(known):
            encoded[flat == label] = index
        self._encoded_ = encoded
        self.n_outputs_ = int(known.shape[0])
        if self.n_outputs_ < 2:
            msg = "MLPClassifier needs at least two classes."
            raise ValueError(msg)
        counts = np.bincount(encoded, minlength=self.n_outputs_)
        self.class_counts_ = [int(count) for count in counts]
        self._sample_weight_ = _class_sample_weight(
            encoded, known, counts, self.class_weight
        )

    def _default_loss(self) -> Loss:
        """Cross-entropy on raw scores."""
        return CrossEntropy()

    def _task(self) -> str:
        """Task name stored in the weights file."""
        return "classifier"

    def _decode_prediction(self, logits: FloatArray) -> NDArray[Any]:
        """Map the winning score back to the original label."""
        winners = np.argmax(logits, axis=1)
        decoded: NDArray[Any] = np.asarray(np.asarray(self.classes_)[winners])
        return decoded

    def _score_arrays(
        self, prediction: FloatArray, target: NDArray[np.generic]
    ) -> float:
        """Fraction of rows whose largest score matches the encoded label."""
        winners = np.argmax(prediction, axis=1)
        labels = np.asarray(target, dtype=np.int64).reshape(-1)
        return float(np.mean(winners == labels))

    def _playback_target(
        self,
        row: FloatArray,
        y: ArrayLike | None,
        with_backward: bool,
    ) -> NDArray[np.generic] | None:
        """Encode one label. Without ``y``, use the model's own prediction."""
        if not with_backward:
            return None
        label = self.predict(row)[0] if y is None else np.asarray(y).reshape(-1)[0]
        known = np.asarray(self.classes_)
        matches = np.flatnonzero(known == label)
        if matches.size != 1:
            msg = f"Playback target {label!r} is not one of classes_."
            raise ValueError(msg)
        return np.asarray([int(matches[0])], dtype=np.int64)

    def _extra_figures(self) -> list[FigureSpec]:
        """Confusion heatmap, plus a scatter when the input has two columns."""
        figures: list[FigureSpec] = []
        if self.confusion_matrix_ is not None:
            labels = [str(label) for label in self.classes_]
            figures.append(
                FigureSpec(
                    kind="heatmap",
                    title="Confusion (training rows used for learning)",
                    data={
                        "matrix": self.confusion_matrix_.tolist(),
                        "xlabels": labels,
                        "ylabels": labels,
                        "xlabel": "predicted",
                        "ylabel": "actual",
                    },
                )
            )
        if self._scatter_xy_ is not None and self._scatter_pred_ is not None:
            figures.append(
                FigureSpec(
                    kind="scatter",
                    title="Training examples in the two input columns",
                    data={
                        "x": self._scatter_xy_[:, 0].tolist(),
                        "y": self._scatter_xy_[:, 1].tolist(),
                        "xlabel": "feature 0",
                        "ylabel": "feature 1",
                    },
                )
            )
        return figures

    def _export_meta(self) -> dict[str, Any]:
        """Labels and class counts travel with the weights."""
        return {
            "classes": [_jsonable_label(label) for label in self.classes_],
            "class_counts": list(self.class_counts_),
        }

    def _restore_task(self, meta: dict[str, Any]) -> None:
        """Put labels back so ``predict`` returns them."""
        self.classes_ = np.asarray(meta["classes"])
        self.class_counts_ = [int(count) for count in meta.get("class_counts", [])]
        self.confusion_matrix_ = None
        self._scatter_xy_ = None
        self._scatter_pred_ = None
        self._encoded_ = np.zeros(0, dtype=np.int64)
        self._sample_weight_ = None


class MLPRegressor(RegressorMixin, _MLPBase):
    """Regressor. ``y`` may be one column or several. Default loss is MSE."""

    def _check_training_data(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[FloatArray, NDArray[np.generic]]:
        """Check a numeric regression table."""
        x_checked = check_array(X, dtype=np.float64)
        y_checked = check_array(y, dtype=np.float64, ensure_2d=False)
        if y_checked.shape[0] != x_checked.shape[0]:
            msg = (
                "X and y have different numbers of rows: "
                f"{x_checked.shape[0]} vs {y_checked.shape[0]}."
            )
            raise ValueError(msg)
        return x_checked, y_checked

    def _prepare_targets(
        self, y: NDArray[np.generic], classes: ArrayLike | None
    ) -> None:
        """Reshape ``y`` to ``(n_samples, n_outputs)``."""
        del classes
        values = np.asarray(y, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
            self._y_was_1d_ = True
        else:
            self._y_was_1d_ = False
        self._encoded_ = values
        self.n_outputs_ = int(values.shape[1])
        self._sample_weight_ = None
        self.class_counts_ = []

    def _default_loss(self) -> Loss:
        """Mean squared error."""
        if isinstance(self.loss, MAE | Huber | MSE):
            copied = clone(self.loss)
            if not isinstance(copied, Loss):
                msg = "Cloned loss is not a Loss."
                raise TypeError(msg)
            return copied
        return MSE()

    def _task(self) -> str:
        """Task name stored in the weights file."""
        return "regressor"

    def _decode_prediction(self, logits: FloatArray) -> NDArray[Any]:
        """Return one column as a vector when the target was one-dimensional."""
        if getattr(self, "_y_was_1d_", logits.shape[1] == 1) and logits.shape[1] == 1:
            return logits.reshape(-1)
        return logits

    def _score_arrays(
        self, prediction: FloatArray, target: NDArray[np.generic]
    ) -> float:
        """R² of this batch. A constant target scores 0 unless the error is 0."""
        pred = np.asarray(prediction, dtype=np.float64)
        truth = np.asarray(target, dtype=np.float64)
        if truth.ndim == 1:
            truth = truth.reshape(-1, 1)
        residual = truth - pred
        ss_res = float(np.sum(np.square(residual)))
        ss_tot = float(np.sum(np.square(truth - np.mean(truth))))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    def _playback_target(
        self,
        row: FloatArray,
        y: ArrayLike | None,
        with_backward: bool,
    ) -> NDArray[np.generic] | None:
        """Numeric target shaped ``(1, n_outputs)``."""
        del row
        if not with_backward:
            return None
        if y is None:
            msg = "Regressor playback with backward needs y."
            raise ValueError(msg)
        values = np.asarray(y, dtype=np.float64).reshape(1, -1)
        if values.shape[1] != self.n_outputs_:
            msg = (
                f"Playback y has {values.shape[1]} outputs; expected {self.n_outputs_}."
            )
            raise ValueError(msg)
        return values

    def _extra_figures(self) -> list[FigureSpec]:
        """Scatter of the two input columns when that is the whole input."""
        if self._scatter_xy_ is None:
            return []
        return [
            FigureSpec(
                kind="scatter",
                title="Training examples in the two input columns",
                data={
                    "x": self._scatter_xy_[:, 0].tolist(),
                    "y": self._scatter_xy_[:, 1].tolist(),
                    "xlabel": "feature 0",
                    "ylabel": "feature 1",
                },
            )
        ]

    def _export_meta(self) -> dict[str, Any]:
        """Remember whether predictions should be flattened."""
        return {"y_was_1d": bool(self._y_was_1d_)}

    def _restore_task(self, meta: dict[str, Any]) -> None:
        """Restore the output layout flag."""
        self._y_was_1d_ = bool(meta.get("y_was_1d", self.n_outputs_ == 1))
        self.class_counts_ = []
        self.confusion_matrix_ = None
        self._scatter_xy_ = None
        self._scatter_pred_ = None
        self._encoded_ = np.zeros((0, self.n_outputs_), dtype=np.float64)
        self._sample_weight_ = None


def _linears(module: Any) -> list[Linear]:
    """Depth-first list of linear layers."""
    found: list[Linear] = []
    if isinstance(module, Linear):
        found.append(module)
    if isinstance(module, Sequential):
        for layer in module.layers:
            found.extend(_linears(layer))
    return found


def _sync_component(template: Any, runtime: Any) -> None:
    """Copy constructor fields from ``template`` onto ``runtime`` if types match."""
    if template is None or runtime is None:
        return
    if type(template) is not type(runtime):
        return
    runtime.set_params(**template.get_params(deep=False))


def _pack_traces(rows: list[EpochStat]) -> tuple[Trace, ...]:
    """Turn epoch rows into the five traces the charts read."""
    loss_rows = []
    score_rows = []
    lr_rows = []
    grad_rows = []
    layer_rows = []
    for row in rows:
        loss: dict[str, Any] = {"epoch": row.epoch, "train_loss": row.train_loss}
        score: dict[str, Any] = {"epoch": row.epoch, "train_score": row.train_score}
        if row.val_loss is not None:
            loss["val_loss"] = row.val_loss
        if row.val_score is not None:
            score["val_score"] = row.val_score
        loss_rows.append(loss)
        score_rows.append(score)
        lr_rows.append({"epoch": row.epoch, "learning_rate": row.learning_rate})
        grad_rows.append({"epoch": row.epoch, "grad_norm": row.grad_norm})
        for layer in row.layers:
            layer_rows.append(
                {
                    "epoch": row.epoch,
                    "layer": layer.layer,
                    "grad_norm": layer.grad_norm,
                    "weight_std": layer.weight_std,
                    "dead_fraction": layer.dead_fraction,
                    "saturated_fraction": layer.saturated_fraction,
                    "activation_mean": layer.activation_mean,
                    "activation_std": layer.activation_std,
                    "activation_name": layer.activation_name,
                }
            )
    return (
        Trace(name="loss", records=tuple(loss_rows)),
        Trace(name="score", records=tuple(score_rows)),
        Trace(name="learning_rate", records=tuple(lr_rows)),
        Trace(name="grad_norm", records=tuple(grad_rows)),
        Trace(name="layers", records=tuple(layer_rows)),
    )


def _series(
    records: tuple[dict[str, Any], ...],
    names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Build line-chart series, skipping a name the records do not have."""
    series: list[dict[str, Any]] = []
    for name in names:
        if not records or name not in records[0]:
            continue
        series.append(
            {
                "name": name,
                "x": [int(row["epoch"]) for row in records],
                "y": [float(row[name]) for row in records],
            }
        )
    return series


def _grad_heatmap(records: tuple[dict[str, Any], ...]) -> FigureSpec:
    """Layer by pass heatmap of gradient length."""
    if not records:
        return FigureSpec(
            kind="heatmap",
            title="Gradient norm by layer and pass",
            data={"matrix": [[0.0]], "xlabels": ["0"], "ylabels": ["0"]},
        )
    epochs = sorted({int(row["epoch"]) for row in records})
    layers = sorted({int(row["layer"]) for row in records})
    lookup = {
        (int(row["epoch"]), int(row["layer"])): float(row["grad_norm"])
        for row in records
    }
    matrix = [[lookup.get((epoch, layer), 0.0) for epoch in epochs] for layer in layers]
    return FigureSpec(
        kind="heatmap",
        title="Gradient norm by layer and pass",
        data={
            "matrix": matrix,
            "xlabels": [str(epoch) for epoch in epochs],
            "ylabels": [str(layer) for layer in layers],
            "xlabel": "pass",
            "ylabel": "layer",
        },
    )


def _softmax(logits: FloatArray) -> FloatArray:
    """Stable softmax over the class axis."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs: FloatArray = np.asarray(
        exp / np.sum(exp, axis=1, keepdims=True), dtype=np.float64
    )
    return probs


def _class_sample_weight(
    encoded: NDArray[np.int64],
    classes: NDArray[Any],
    counts: NDArray[np.int64],
    class_weight: Literal["balanced"] | dict[Any, float] | None,
) -> FloatArray | None:
    """Per-row weights. ``balanced`` is ``n / (n_classes * count)``."""
    if class_weight is None:
        return None
    n_classes = int(classes.shape[0])
    per_class = np.ones(n_classes, dtype=np.float64)
    if class_weight == "balanced":
        total = float(encoded.shape[0])
        for index, count in enumerate(counts):
            if count > 0:
                per_class[index] = total / (n_classes * float(count))
    elif isinstance(class_weight, dict):
        for index, label in enumerate(classes):
            key = label.item() if isinstance(label, np.generic) else label
            if key not in class_weight:
                msg = f"class_weight is missing label {key!r}."
                raise ValueError(msg)
            per_class[index] = float(class_weight[key])
    else:
        msg = "class_weight must be None, 'balanced', or a dict."
        raise TypeError(msg)
    return per_class[encoded]


def _jsonable_label(label: Any) -> Any:
    """Turn a NumPy label into a JSON value."""
    if isinstance(label, np.generic):
        value = label.item()
        return value
    return label


# Identity is part of the activation set users can pass explicitly.
__all__ = ["Identity", "MLPClassifier", "MLPRegressor"]
