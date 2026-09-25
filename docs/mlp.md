# Multilayer perceptron

A multilayer perceptron is a stack of neurons. Each neuron computes a weighted
sum of the previous layer and then an activation. TeoTensor trains that stack
with NumPy only. The arithmetic functions in `teotensor.nn.tensor.ops` are the
door a faster backend can replace later; the model, the trainer, and the
studio do not change when that happens.

## Quickstart

```python
import numpy as np
from teotensor.artifacts import export_html
from teotensor.nn import MLPClassifier
from teotensor.nn.activations import Tanh
from teotensor.nn.optim.adam import Adam
from teotensor.nn.penalty import L2

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([0, 1, 1, 0])

model = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation=Tanh(),
    solver=Adam(lr=0.08),
    penalty=L2(alpha=0.0),
    max_iter=400,
    validation_fraction=0.0,
    random_state=0,
).fit(X, y)

print(model.predict(X))
export_html(model.observe(), "mlp_report.html", open_browser=True)
```

`MLPRegressor` is the same stack for numeric targets. Pass `loss=MAE()` or
`loss=Huber(delta=1.0)` to change the regression loss. The default is mean
squared error.

## What the constructor stores

`None` means "use the documented default when `fit` runs", on a copy:

| Setting | Default |
|---|---|
| `activation` | ReLU |
| `solver` | Adam with step length `0.001` |
| `schedule` | constant step length |
| `penalty` | L2 with strength `1e-4` |
| `weight_init` | chosen from the activation |
| `bias_init` | zeros |
| `loss` | cross-entropy (classifier) or mean squared error (regressor) |

The step length lives on the solver (`solver=Adam(lr=0.001)`), not as a
second field on the model. The penalty is the only knob for large weights.
Biases are not penalized. For Adam the squared part of the penalty is applied
as decay outside Adam's moving averages; for SGD and RMSprop it is added to
the loss.

`hidden_activations=(ReLU(), Tanh())` sets a different activation on each
hidden layer. The output layer has no activation. The loss reads raw scores.
A classifier with two classes still uses two scores.

## Dropout

While learning, each neuron on each training row draws a coin. With
probability `dropout` the output becomes 0 and so does its gradient, so that
neuron does not learn from that row on that step. Survivors are multiplied by
`1 / (1 - dropout)` so the average size of the layer matches exam mode.
`predict`, reports, and playback run in exam mode: nobody is silenced and
there is no scale.

## After training

- `predict` answers from the learned weights. The backward tape is not used.
- `save` / `load` store the whole Python object so you can keep training.
- `export_weights` / `load_weights` store a `.ttw` zip (a JSON manifest plus
  NumPy arrays). That file is the one to keep for inference. Opening it does
  not run code. GGUF is not written: that format is for language-model
  runners, which do not know this network.

`coefs_` and `intercepts_` read the weight matrices and biases off the live
network. They are not a second stored copy.

## What you can inspect

`report`, `diagnose`, `trace`, and `visualize` describe the run. `playback(x, y)`
returns frames of one example: neurons, then wires, layer by layer, then the
error walking back. The studio paints forward frames in teal and blue and
backward frames in amber and crimson.

```python
from teotensor.studio import open_studio
open_studio()  # stays running until Ctrl+C
```

The studio opens on a session list. A train session walks dataset, model,
then fit. An infer session loads a `.ttw` file and scores rows. PCA and both
MLPs share that page.
K-Means and the other algorithms join that list when their classes exist.
