"""The one training loop used by hand-built stacks and by MLP models.

Each epoch cuts the table into batches. On a batch the loop runs the network
forward, adds the loss and the penalty, walks the error backward, optionally
clips that error signal, then asks the optimizer to step. At the end of the
epoch it writes one row of scalars. There is no callback object.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import Generator, RandomState
from numpy.typing import NDArray

from teotensor.data.dataset import batches
from teotensor.nn.layers.activations import ActivationLayer
from teotensor.nn.layers.linear import Linear
from teotensor.nn.layers.module import Module
from teotensor.nn.losses import Loss
from teotensor.nn.optim.schedule import Schedule
from teotensor.nn.optim.sgd import Optimizer
from teotensor.nn.penalty import Penalty
from teotensor.nn.tensor.core import Parameter, Tensor

FloatArray = NDArray[np.float64]
RNG = RandomState | Generator

# The studio binds a dict for the epoch counter. Training math ignores it.
_PROGRESS: ContextVar[dict[str, Any] | None] = ContextVar(
    "teotensor_fit_progress", default=None
)
ScoreFn = Callable[[FloatArray, NDArray[np.generic]], float]


@dataclass
class LayerStat:
    """Scalars describing one linear layer at the end of an epoch."""

    layer: int
    grad_norm: float
    weight_std: float
    dead_fraction: float
    saturated_fraction: float
    activation_mean: float
    activation_std: float
    activation_name: str


@dataclass
class EpochStat:
    """Scalars describing one pass over the training table."""

    epoch: int
    train_loss: float
    val_loss: float | None
    train_score: float
    val_score: float | None
    learning_rate: float
    grad_norm: float
    weight_norm: float
    layers: list[LayerStat] = field(default_factory=list)


@dataclass
class TrainResult:
    """What the loop returns to the model."""

    epochs: list[EpochStat]
    converged: bool
    best_loss: float
    n_iter: int


def fit_module(
    module: Module,
    *,
    loss: Loss,
    optimizer: Optimizer,
    schedule: Schedule,
    penalty: Penalty | None,
    x_train: FloatArray,
    y_train: NDArray[np.generic],
    sample_weight: FloatArray | None,
    x_val: FloatArray | None,
    y_val: NDArray[np.generic] | None,
    epochs: int,
    batch_size: int,
    shuffle: bool,
    tol: float,
    n_iter_no_change: int,
    early_stopping: bool,
    max_grad_norm: float | None,
    rng: RNG,
    score_fn: ScoreFn,
    on_epoch: Callable[[EpochStat], None] | None = None,
    start_epoch: int = 1,
) -> TrainResult:
    """Train ``module`` in place and return per-epoch scalars.

    Parameters
    ----------
    module : Module
        Network. Learning mode and exam mode are switched by this function.
    loss, optimizer, schedule, penalty
        Fitted clones. ``penalty`` may be None.
    x_train, y_train : ndarray
        Training table and targets. Targets are already encoded.
    sample_weight : ndarray or None
        Per-row weights for the loss, or None.
    x_val, y_val : ndarray or None
        Held-out tail. Both are None when there is no validation split.
    epochs : int
        Maximum passes over ``x_train``.
    batch_size : int
        Rows per step. The last batch may be smaller.
    shuffle : bool
        If True, shuffle row order at the start of each epoch.
    tol : float
        A pass "improves" when the watched loss drops by more than this.
    n_iter_no_change : int
        Stop after this many passes without an improvement.
    early_stopping : bool
        If True, watch validation loss (or training loss when there is no
        validation split) and restore the weights from the best pass.
    max_grad_norm : float or None
        If set, shrink the whole gradient so its length is at most this.
    rng : RandomState or Generator
        Stream used only for shuffling here. Dropout already holds the same
        stream and draws from it during the forward pass.
    score_fn : callable
        ``score_fn(prediction, target) -> float`` on NumPy arrays.
    on_epoch : callable or None, optional
        Called with the epoch row after it is recorded, while the weights are
        still the ones that just finished that pass. The MLP uses this to
        snapshot a playback film. It is not a general plugin system.

    Returns
    -------
    TrainResult
        Epoch rows, whether patience stopped the run, and the best watched loss.
    """
    if epochs < 1:
        msg = f"epochs must be >= 1; got {epochs}."
        raise ValueError(msg)
    if batch_size < 1:
        msg = f"batch_size must be >= 1; got {batch_size}."
        raise ValueError(msg)

    parameters = module.parameters()
    weights = [param for param in parameters if param.data.ndim == 2]
    module.probe_batch = x_train[: min(256, x_train.shape[0])]  # type: ignore[attr-defined]
    best_loss = math.inf
    best_state: dict[str, FloatArray] | None = None
    quiet = 0
    previous_improved = True
    rows: list[EpochStat] = []
    converged = False

    for epoch in range(start_epoch, start_epoch + epochs):
        if not wait_if_paused():
            break
        _mark("feedforward", epoch)
        multiplier = schedule.scale(epoch, improved=previous_improved)
        learning_rate = float(optimizer.lr) * multiplier
        grad_norm, layer_grads = _run_epoch(
            module,
            parameters=parameters,
            weights=weights,
            loss=loss,
            optimizer=optimizer,
            penalty=penalty,
            x_train=x_train,
            y_train=y_train,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=shuffle,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            rng=rng,
            epoch=epoch,
        )
        if _stopped():
            break
        _mark("evaluate", epoch)
        train_loss, train_score = _evaluate(module, loss, x_train, y_train, score_fn)
        val_loss: float | None = None
        val_score: float | None = None
        if x_val is not None and y_val is not None:
            val_loss, val_score = _evaluate(module, loss, x_val, y_val, score_fn)
        watched = val_loss if early_stopping and val_loss is not None else train_loss
        if not early_stopping:
            watched = train_loss
        previous_improved = (best_loss - watched) > tol
        if previous_improved or best_state is None:
            best_loss = watched
            best_state = module.state_dict()
            quiet = 0
        else:
            quiet += 1
        rows.append(
            EpochStat(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_score=train_score,
                val_score=val_score,
                learning_rate=learning_rate,
                grad_norm=grad_norm,
                weight_norm=_weight_norm(weights),
                layers=_layer_stats(module, layer_grads),
            )
        )
        if on_epoch is not None:
            on_epoch(rows[-1])
        if quiet >= n_iter_no_change:
            converged = True
            break

    if early_stopping and best_state is not None:
        module.load_state_dict(best_state)
    module.eval()
    return TrainResult(
        epochs=rows,
        converged=converged,
        best_loss=float(best_loss),
        n_iter=len(rows),
    )


def _run_epoch(
    module: Module,
    *,
    parameters: list[Parameter],
    weights: list[Parameter],
    loss: Loss,
    optimizer: Optimizer,
    penalty: Penalty | None,
    x_train: FloatArray,
    y_train: NDArray[np.generic],
    sample_weight: FloatArray | None,
    batch_size: int,
    shuffle: bool,
    learning_rate: float,
    max_grad_norm: float | None,
    rng: RNG,
    epoch: int,
) -> tuple[float, dict[int, float]]:
    """One pass. Returns the mean pre-clip gradient length and per-layer lengths."""
    module.train()
    n_rows = x_train.shape[0]
    grad_lengths: list[float] = []
    layer_grads: dict[int, float] = {}
    linears = _linears(module)
    index_rows = np.arange(n_rows, dtype=np.int64)
    for batch in batches(index_rows, batch_size, shuffle=shuffle, rng=rng):
        if not wait_if_paused():
            break
        module.zero_grad()
        _mark("feedforward", epoch, step=True)
        prediction = module(Tensor(x_train[batch]))
        weight = None if sample_weight is None else sample_weight[batch]
        data_loss = loss.forward(prediction, y_train[batch], weight)
        total = data_loss
        if penalty is not None:
            total = total + penalty.loss_term(
                weights, decoupled_l2=optimizer.decoupled_l2
            )
        _mark("backpropagation", epoch, step=True)
        total.backward()
        length = _global_norm(parameters)
        grad_lengths.append(length)
        layer_grads = {
            index: _param_norm(linear) for index, linear in enumerate(linears)
        }
        if max_grad_norm is not None and length > max_grad_norm > 0.0:
            scale = max_grad_norm / (length + 1e-12)
            for param in parameters:
                if param.grad is not None:
                    param.grad = param.grad * scale
        optimizer.step(parameters, lr=learning_rate, penalty=penalty)
    mean_length = float(np.mean(grad_lengths)) if grad_lengths else 0.0
    return mean_length, layer_grads


def bind_progress(box: dict[str, Any]) -> Token[dict[str, Any] | None]:
    """Publish epoch and phase into ``box`` until :func:`release_progress`."""
    return _PROGRESS.set(box)


def release_progress(token: Token[dict[str, Any] | None]) -> None:
    """Stop publishing into the box from :func:`bind_progress`."""
    _PROGRESS.reset(token)


def _mark(phase: str, epoch: int, *, step: bool = False) -> None:
    """Publish ``phase`` for this epoch.

    ``step`` counts one real half-pass, the forward or the backward of one
    batch. Scoring rows is ``evaluate`` and does not take a step: those
    passes are forward, and the studio must not keep painting backward.
    """
    box = _PROGRESS.get()
    if box is None:
        return
    box["phase"] = phase
    box["epoch"] = epoch
    if step:
        box["tick"] = int(box.get("tick", 0)) + 1


def wait_if_paused() -> bool:
    """Block while the studio has paused this fit.

    Returns False when the run should stop, so Ctrl+C can leave a paused
    thread. A missing gate means nothing is steering the fit.
    """
    box = _PROGRESS.get()
    if box is None:
        return True
    gate = box.get("gate")
    if gate is None:
        return not bool(box.get("stop"))
    if not gate.is_set():
        box["phase"] = "paused"
    while not gate.is_set():
        if box.get("stop"):
            gate.set()
            return False
        gate.wait(0.25)
    return not bool(box.get("stop"))


def _stopped() -> bool:
    box = _PROGRESS.get()
    return box is not None and bool(box.get("stop"))


def _evaluate(
    module: Module,
    loss: Loss,
    features: FloatArray,
    target: NDArray[np.generic],
    score_fn: ScoreFn,
) -> tuple[float, float]:
    """Loss and score in exam mode, without building a training step."""
    module.eval()
    prediction = module(Tensor(features))
    value = float(loss.forward(prediction, target, None).data)
    score = float(score_fn(prediction.data, target))
    return value, score


def _layer_stats(module: Module, layer_grads: dict[int, float]) -> list[LayerStat]:
    """Measure activations on the current weights, in exam mode."""
    module.eval()
    linears = _linears(module)
    activations = _activation_layers(module)
    # Pair each hidden activation with the linear layer that feeds it.
    # The output linear has no activation layer after it.
    stats: list[LayerStat] = []
    capture_targets = list(activations)
    for layer in capture_targets:
        layer.capture = True
    # A fresh forward fills last_data. Use whatever batch the caller left
    # available via the module attribute set by the estimator. When it is
    # missing, activation stats stay at zero and only weight stats are real.
    probe = getattr(module, "probe_batch", None)
    if probe is not None and capture_targets:
        module(Tensor(probe))
    for index, linear in enumerate(linears):
        activation = activations[index] if index < len(activations) else None
        dead = 0.0
        saturated = 0.0
        mean = 0.0
        std = 0.0
        name = "logits"
        if activation is not None and activation.last_data is not None:
            data = activation.last_data
            name = type(activation.activation).__name__
            mean = float(np.mean(data))
            std = float(np.std(data))
            if name in {"ReLU", "LeakyReLU"}:
                dead = float(np.mean(data <= 0.0))
            if name == "Tanh":
                saturated = float(np.mean(np.abs(data) > 0.95))
            if name == "Sigmoid":
                saturated = float(np.mean((data < 0.05) | (data > 0.95)))
        weight = linear.weight.data
        stats.append(
            LayerStat(
                layer=index,
                grad_norm=float(layer_grads.get(index, 0.0)),
                weight_std=float(np.std(weight)),
                dead_fraction=dead,
                saturated_fraction=saturated,
                activation_mean=mean,
                activation_std=std,
                activation_name=name,
            )
        )
    for layer in capture_targets:
        layer.capture = False
    return stats


def _linears(module: Module) -> list[Linear]:
    """Every Linear layer, depth-first."""
    found: list[Linear] = []
    if isinstance(module, Linear):
        found.append(module)
    for child in module.children():
        found.extend(_linears(child))
    return found


def _activation_layers(module: Module) -> list[ActivationLayer]:
    """Every activation wrapper, depth-first."""
    found: list[ActivationLayer] = []
    if isinstance(module, ActivationLayer):
        found.append(module)
    for child in module.children():
        found.extend(_activation_layers(child))
    return found


def _global_norm(parameters: list[Parameter]) -> float:
    """Euclidean length of all gradients stacked together."""
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(np.sum(np.square(param.grad)))
    return math.sqrt(total)


def _param_norm(linear: Linear) -> float:
    """Euclidean length of one layer's weight and bias gradients."""
    total = 0.0
    if linear.weight.grad is not None:
        total += float(np.sum(np.square(linear.weight.grad)))
    if linear.bias is not None and linear.bias.grad is not None:
        total += float(np.sum(np.square(linear.bias.grad)))
    return math.sqrt(total)


def _weight_norm(weights: list[Parameter]) -> float:
    """Euclidean length of all weight matrices, biases excluded."""
    total = 0.0
    for weight in weights:
        total += float(np.sum(np.square(weight.data)))
    return math.sqrt(total)
