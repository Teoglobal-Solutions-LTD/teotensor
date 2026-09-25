"""``.ttw`` round-trips for PCA and MLP."""

from __future__ import annotations

import numpy as np
from teotensor.classical import PCA
from teotensor.nn.estimators.mlp import MLPRegressor
from teotensor.nn.optim.adam import Adam
from teotensor.nn.penalty import L2


def test_pca_weights_restore_transform(tmp_path) -> None:
    rng = np.random.RandomState(0)
    features = rng.normal(size=(40, 4))
    model = PCA(n_components=2).fit(features)
    path = tmp_path / "pca.ttw"
    model.export_weights(str(path))
    restored = PCA.load_weights(str(path))
    assert np.allclose(model.transform(features), restored.transform(features))


def test_mlp_weights_restore_predict(tmp_path) -> None:
    rng = np.random.RandomState(0)
    features = rng.normal(size=(30, 2))
    target = features[:, 0] - features[:, 1]
    model = MLPRegressor(
        hidden_layer_sizes=(4,),
        solver=Adam(lr=0.05),
        penalty=L2(alpha=0.0),
        max_iter=40,
        validation_fraction=0.0,
        random_state=0,
        inspect_every=0,
    )
    model.fit(features, target)
    path = tmp_path / "mlp.ttw"
    model.export_weights(str(path))
    restored = MLPRegressor.load_weights(str(path))
    assert np.allclose(model.predict(features), restored.predict(features))
