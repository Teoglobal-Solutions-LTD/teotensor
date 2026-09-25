# PCA

Principal Component Analysis is TeoTensor's first classical estimator: a linear
dimensionality-reduction transform that is **inspectable by default**.

> Models describe data — they do not draw plots. PCA emits `FigureSpec` /
> `Report` / `Diagnostic` artifacts; `export_html` is the primary UI.

## Algorithm

1. Center \(X\) by subtracting the feature-wise mean (`mean_`).
2. Compute an economy SVD of the centered matrix: \(X_c = U\Sigma V^\top\).
3. Keep the first \(k\) right singular vectors as `components_`.
4. Explained variances use the unbiased divisor \(n-1\) (when \(n > 1\)).

`transform` projects onto the principal subspace; `inverse_transform` maps
scores back to feature space (lossy when \(k < \mathrm{rank}\)).

## Quickstart

```python
import numpy as np
from teotensor.classical import PCA
from teotensor.artifacts import export_html

rng = np.random.RandomState(0)
X = rng.normal(size=(200, 6))

pca = PCA(n_components=0.9).fit(X)
Z = pca.transform(X)
X_hat = pca.inverse_transform(Z)

export_html(pca.observe(), open_browser=True)
```

### `n_components`

| Value | Meaning |
|---|---|
| `None` | Keep `min(n_samples, n_features)` |
| `int` | Keep that many components |
| `float` in `(0, 1]` | Smallest \(k\) that retains at least that variance fraction |

## Observability

After `fit`, PCA exposes:

| Method | Content |
|---|---|
| `report()` | Retained variance, reconstruction MSE, per-component table |
| `diagnose()` | Near-zero features, weak retention, ill-conditioned spectrum |
| `visualize()` | Scree bar chart + cumulative explained-variance line |
| `observe()` → `export_html` | Styled interactive dashboard |

## API surface

Fitted attributes (sklearn-compatible names):

- `components_`, `explained_variance_`, `explained_variance_ratio_`
- `singular_values_`, `mean_`, `n_components_`
- `noise_variance_`, `reconstruction_error_`

See also: [Observability](observability.md), example `examples/pca_dashboard.py`.
