"""MLP learns XOR, fits a line, exports weights, and films one example."""

from __future__ import annotations

import numpy as np
from teotensor.nn.activations import Tanh
from teotensor.nn.estimators.mlp import MLPClassifier, MLPRegressor
from teotensor.nn.optim.adam import Adam
from teotensor.nn.penalty import L2


def test_pause_blocks_until_resume() -> None:
    import threading

    from teotensor.nn.training.loop import (
        bind_progress,
        release_progress,
        wait_if_paused,
    )

    gate = threading.Event()
    gate.clear()
    box: dict[str, object] = {"phase": "feedforward", "epoch": 2, "gate": gate}
    finished = threading.Event()

    def _wait() -> None:
        token = bind_progress(box)
        try:
            assert wait_if_paused()
        finally:
            release_progress(token)
        finished.set()

    worker = threading.Thread(target=_wait)
    worker.start()
    try:
        for _ in range(20):
            if box["phase"] == "paused":
                break
            threading.Event().wait(0.02)
        assert box["phase"] == "paused"
        assert not finished.is_set()
        gate.set()
        assert finished.wait(1.0)
    finally:
        gate.set()
        worker.join(timeout=1.0)


def test_fit_publishes_the_epoch_and_phase() -> None:
    from teotensor.nn.training.loop import bind_progress, release_progress

    features = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 1, 1, 0])
    box: dict[str, object] = {"epoch": 0, "phase": ""}
    token = bind_progress(box)
    try:
        MLPClassifier(
            hidden_layer_sizes=(4,),
            max_iter=2,
            batch_size=4,
            validation_fraction=0.0,
            random_state=0,
            inspect_every=0,
        ).fit(features, labels)
    finally:
        release_progress(token)
    assert int(box["epoch"]) >= 1
    assert box["phase"] == "evaluate"
    assert int(box["tick"]) == 4


def test_classifier_learns_xor_and_replays_the_same_weights() -> None:
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
        inspect_every=0,
    )
    model.fit(features, labels)
    assert model.score(features, labels) == 1.0
    again = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation=Tanh(),
        solver=Adam(lr=0.08),
        penalty=L2(alpha=0.0),
        max_iter=400,
        batch_size=4,
        validation_fraction=0.0,
        random_state=0,
        inspect_every=0,
    )
    again.fit(features, labels)
    assert np.allclose(model.coefs_[0], again.coefs_[0])
    film = model.playback(features[1], labels[1], with_backward=True)
    phases = [frame.phase for frame in film.frames]
    assert phases[0] == "forward"
    assert "backward" in phases
    assert film.frames[0].kind == "neurons"
    assert len(film.frames[0].values) == film.layer_sizes[0]


def test_regressor_fits_a_line() -> None:
    rng = np.random.RandomState(0)
    features = rng.uniform(-1.0, 1.0, size=(80, 1))
    target = 2.0 * features[:, 0] + 1.0
    model = MLPRegressor(
        hidden_layer_sizes=(),
        solver=Adam(lr=0.1),
        penalty=L2(alpha=0.0),
        max_iter=250,
        validation_fraction=0.0,
        random_state=0,
        inspect_every=0,
    )
    model.fit(features, target)
    assert model.score(features, target) > 0.95


def test_report_and_diagnose_do_not_draw() -> None:
    features = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 1, 1, 0])
    model = MLPClassifier(
        hidden_layer_sizes=(4,),
        activation=Tanh(),
        solver=Adam(lr=0.1),
        penalty=L2(alpha=0.0),
        max_iter=30,
        batch_size=4,
        validation_fraction=0.0,
        random_state=0,
        inspect_every=0,
    )
    model.fit(features, labels)
    report = model.report()
    assert report.metrics["n_iter"] == 30 or model.converged_
    assert model.diagnose()
    assert any(figure.kind == "heatmap" for figure in model.visualize())
