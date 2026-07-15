# Core API

TeoTensor estimators follow a single, sklearn-style contract living in
`teotensor.core`. Every later model (PCA, clustering, trees, …) builds on this
layer.

## Design rules

1. **Hyperparameters in `__init__` only.** Store them as-is. No training,
   no data validation inside the constructor.
2. **Fitted attributes end with `_`.** Examples: `mean_`, `labels_`, `coef_`.
3. **Shared validation.** Call `check_array` / `check_X_y` /
   `check_random_state` inside `fit` / `transform` / `predict`.
4. **Compose mixins.** Do not copy `fit_transform` or `score` into every class.

## Building blocks

| Symbol | Role |
|---|---|
| `BaseEstimator` | Root type; provides `get_params` / `set_params` / `__repr__` |
| `ParamMixin` | Parameter introspection used by `BaseEstimator` |
| `TransformerMixin` | Adds `fit_transform` |
| `ClassifierMixin` | Adds `score` (accuracy) |
| `RegressorMixin` | Adds `score` (`R²`) |
| `ClusterMixin` | Adds `fit_predict` |
| `SerializableMixin` | Adds `save` / `load` (pickle) |
| `check_array` / `check_X_y` / `check_random_state` / `check_is_fitted` | Input guards |

## Minimal transformer

```python
from teotensor.core import BaseEstimator, TransformerMixin, check_array, check_is_fitted
import numpy as np

class MeanCenter(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean: bool = True) -> None:
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        return self

    def transform(self, X):
        check_is_fitted(self, "mean_")
        return check_array(X, dtype=float) - self.mean_
```

`get_params()` / `set_params()` work automatically from the `__init__`
signature. Nested estimators use the `component__param` naming scheme.

## Try it

```powershell
python examples/core_dummy_estimators.py
```

## API reference

::: teotensor.core.BaseEstimator

::: teotensor.core.ParamMixin

::: teotensor.core.validation.check_array

::: teotensor.core.validation.check_X_y

::: teotensor.core.validation.check_random_state
