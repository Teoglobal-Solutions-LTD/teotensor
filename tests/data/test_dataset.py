"""Dataset indices, steps, and the two readers."""

from __future__ import annotations

import gzip
import io
import struct
import zipfile
from pathlib import Path

import numpy as np
from teotensor.data import (
    Standardize,
    batches,
    from_arrays,
    read_csv,
    read_mnist,
    scale,
)


def test_split_tail_holds_out_the_last_rows() -> None:
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    table = from_arrays(features)
    table.split_tail(0.2)
    assert list(table.train_idx) == list(range(8))
    assert list(table.val_idx) == [8, 9]
    assert table.test_idx.size == 0


def test_stratified_hold_out_keeps_each_class() -> None:
    targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    table = from_arrays(np.zeros((8, 1)), targets)
    table.hold_out(0.5, np.random.RandomState(0), stratify=True)
    train = targets[table.train_idx]
    val = targets[table.val_idx]
    assert sorted(train.tolist()) == [0, 0, 1, 1]
    assert sorted(val.tolist()) == [0, 0, 1, 1]


def test_batches_leave_a_short_tail() -> None:
    chunks = list(batches(np.arange(5), 2, shuffle=False, rng=np.random.RandomState(0)))
    assert [int(chunk.size) for chunk in chunks] == [2, 2, 1]
    assert list(chunks[0]) == [0, 1]
    assert list(chunks[-1]) == [4]


def test_scale_promotes_uint8_without_touching_the_source() -> None:
    table = from_arrays(np.array([[255, 0]], dtype=np.uint8))
    scaled = scale(table)
    assert scaled.features.dtype == np.float64
    assert scaled.features[0, 0] == 1.0
    assert table.features.dtype == np.uint8


def test_standardize_fits_only_the_given_rows() -> None:
    features = np.array([[0.0], [2.0], [100.0]])
    table = from_arrays(features)
    table.train_idx = np.array([0, 1], dtype=np.int64)
    table.val_idx = np.array([2], dtype=np.int64)
    transformed = Standardize().fit(table, table.train_idx).transform(table)
    assert abs(float(transformed.features[table.train_idx].mean())) < 1e-8
    assert abs(float(transformed.features[2, 0]) - 99.0) < 1e-6


def test_read_csv_reads_a_file_and_a_long_paste(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    loaded = read_csv(path)
    assert loaded.features.shape == (2, 2)
    loaded = read_csv(str(path))
    assert list(loaded.features[0]) == [1.0, 2.0]
    lines = ["a,b,y"]
    for index in range(800):
        bit = index % 2
        lines.append(f"{bit},{1 - bit},{bit}")
    pasted = read_csv("\n".join(lines) + "\n", target="y")
    assert pasted.features.shape == (800, 2)


def test_read_csv_keeps_a_text_target() -> None:
    table = read_csv("a,b,label\n1,2,cat\n3,4,dog\n", target="label")
    assert table.n_features == 2
    assert table.feature_names == ("a", "b")
    assert list(table.targets) == ["cat", "dog"]


def test_read_mnist_zip_keeps_official_roles() -> None:
    train = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    test = np.array([[[9, 0], [1, 2]]], dtype=np.uint8)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("train-images-idx3-ubyte.gz", _images(train))
        archive.writestr("train-labels-idx1-ubyte.gz", _labels(np.array([7, 1])))
        archive.writestr("t10k-images-idx3-ubyte.gz", _images(test))
        archive.writestr("t10k-labels-idx1-ubyte.gz", _labels(np.array([4])))
    table = read_mnist(buffer.getvalue())
    assert table.image_shape == (2, 2)
    assert list(table.train_idx) == [0, 1]
    assert list(table.test_idx) == [2]
    assert list(table.targets) == [7, 1, 4]
    assert list(table.features[0]) == [1, 2, 3, 4]
    assert list(table.features[2]) == [9, 0, 1, 2]


def _images(frames: np.ndarray) -> bytes:
    count, height, width = frames.shape
    header = struct.pack(">IIII", 2051, count, height, width)
    raw = np.ascontiguousarray(frames, dtype=np.uint8).tobytes()
    return gzip.compress(header + raw)


def _labels(labels: np.ndarray) -> bytes:
    header = struct.pack(">II", 2049, len(labels))
    raw = np.ascontiguousarray(labels, dtype=np.uint8).tobytes()
    return gzip.compress(header + raw)
