# Data

A file is not the thing the model trains on. CSV text, a MNIST zip, and arrays
already in memory all become one `Dataset`. Every row has one index. Features,
the target, and the sample weight are looked up by that index.

## Splits are lists of indices

```python
import numpy as np
from teotensor.data import batches, from_arrays

features = np.arange(20, dtype=np.float64).reshape(10, 2)
targets = np.array([0, 1] * 5)
table = from_arrays(features, targets, target_name="y")
table.split_tail(0.2)
```

`split_tail(0.2)` moves the last two training indices into `val_idx`. The
matrix is not copied. `test_idx` stays empty until a reader puts rows there.
`MLPClassifier.fit` uses this same tail rule for `validation_fraction`, so a
caller that only has `X` and `y` still gets the old holdout.

`hold_out(fraction, rng, stratify=True)` draws the validation indices at
random from the training list. Each class gives up the same fraction of its
rows. The test list is not touched.

## Two kinds of steps

`scale(table, 255.0)` divides every row by the same number. It does not look
at other rows, so it may run before the split. MNIST stays `uint8` until this
step; the result is `float64`.

`Standardize` is the other kind. `fit(table, table.train_idx)` records the
mean and standard deviation of those rows only. `transform` then applies that
rule to every row, including validation and test. Fitting on the whole table
would let the test set into the training rule.

## Minibatches

```python
for batch in batches(table.train_idx, batch_size=32, shuffle=True, rng=rng):
    x, y, weight = table.take(batch)
```

`batches` shuffles a copy of the index list once per epoch. The last batch may
be shorter. `take` returns `float64` features for those indices, plus the
target and the sample weight cut with the same numbers. The trainer uses this
iterator. It does not decide the split.

## Readers

`read_csv(text_or_path, target="y")` uses the first row as column names. The
named column becomes `targets` and leaves the feature matrix. A numeric target
stays numeric. Other text stays text. Class indices are still computed inside
`MLPClassifier`, not here.

`read_mnist(zip_bytes_or_path)` reads the official IDX gzip members
`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, and the `t10k`
pair. Training rows stay in `train_idx`. Test rows stay in `test_idx`.
Validation is cut from training later. The archive is not downloaded.

`image_shape` is `(28, 28)` for a real MNIST file. The network still sees 784
numbers. A viewer uses the shape to draw the frame.

## Studio

The studio is a sequence of sessions. A train session fits a model. An infer
session loads a `.ttw` file and scores every row. Each session walks three
screens: the dataset, the model, then the run.

Dataset steps are explicit. A divisor field divides every stored number by
whatever you type (8-bit pixels are often 255, other tables are not). Leaving
it empty keeps the file's values. Holding out a fraction moves the tail of
`train_idx` to validation. Centering fits mean and scale on the training rows
only, after that split. Training uses `train_idx`. Inference scores the whole
table, so held-out rows stay available.
