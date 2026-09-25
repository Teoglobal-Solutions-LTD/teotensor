"""Read the MNIST IDX archive into one dataset.

The archive is a zip of the official files
``train-images-idx3-ubyte.gz``, ``train-labels-idx1-ubyte.gz``, and the
matching ``t10k`` pair. Training rows and test rows stay in those roles.
Pixel values stay ``uint8`` until a scale step promotes them.
"""

from __future__ import annotations

import gzip
import io
import struct
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from teotensor.data.dataset import Dataset

_IMAGE_MAGIC = 2051
_LABEL_MAGIC = 2049


def read_mnist(source: str | Path | bytes) -> Dataset:
    """Load a zip archive, or a directory that holds the IDX files.

    Parameters
    ----------
    source : str, Path, or bytes
        Path to a ``.zip``, raw zip bytes, or a directory of ``.gz`` members.
    """
    members = _members(source)
    train_images, image_shape = _images(_must(_find(members, train=True, images=True)))
    train_labels = _labels(_must(_find(members, train=True, images=False)))
    test_blob = _find(members, train=False, images=True, required=False)
    test_labels_blob = _find(members, train=False, images=False, required=False)
    if (test_blob is None) != (test_labels_blob is None):
        msg = "MNIST test images and test labels must be provided together."
        raise ValueError(msg)
    if train_images.shape[0] != train_labels.shape[0]:
        msg = "MNIST training images and labels have different lengths."
        raise ValueError(msg)
    features = train_images
    targets = train_labels
    n_train = int(train_images.shape[0])
    test_idx = np.empty(0, dtype=np.int64)
    if test_blob is not None and test_labels_blob is not None:
        test_images, test_shape = _images(test_blob)
        test_labels = _labels(test_labels_blob)
        if test_shape != image_shape:
            msg = "MNIST train and test images have different frame sizes."
            raise ValueError(msg)
        if test_images.shape[0] != test_labels.shape[0]:
            msg = "MNIST test images and labels have different lengths."
            raise ValueError(msg)
        features = np.concatenate([train_images, test_images], axis=0)
        targets = np.concatenate([train_labels, test_labels], axis=0)
        n_test = int(test_images.shape[0])
        test_idx = np.arange(n_train, n_train + n_test, dtype=np.int64)
    height, width = image_shape
    names = tuple(f"p{i}" for i in range(height * width))
    return Dataset(
        features=features.reshape(features.shape[0], height * width),
        targets=targets,
        feature_names=names,
        target_name="digit",
        image_shape=image_shape,
        train_idx=np.arange(n_train, dtype=np.int64),
        test_idx=test_idx,
    )


def _members(source: str | Path | bytes) -> dict[str, bytes]:
    """Map a lower-case file name to its uncompressed bytes."""
    if isinstance(source, bytes):
        return _zip_members(source)
    path = Path(source)
    if path.is_dir():
        found: dict[str, bytes] = {}
        for child in path.iterdir():
            if child.is_file():
                found[child.name.lower()] = _maybe_gunzip(child.read_bytes())
        return found
    return _zip_members(path.read_bytes())


def _zip_members(payload: bytes) -> dict[str, bytes]:
    """Read a zip into name -> uncompressed bytes."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        msg = "MNIST source must be a zip archive or a directory of IDX files."
        raise ValueError(msg) from exc
    found: dict[str, bytes] = {}
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name.lower()
            found[name] = _maybe_gunzip(archive.read(info))
    return found


def _must(payload: bytes | None) -> bytes:
    """Return ``payload`` or raise when a required member was absent."""
    if payload is None:
        msg = "MNIST archive is missing a required file."
        raise ValueError(msg)
    return payload


def _find(
    members: dict[str, bytes],
    *,
    train: bool,
    images: bool,
    required: bool = True,
) -> bytes | None:
    """Pick one member by the usual MNIST file name."""
    want = "train" if train else "test"
    for name, payload in members.items():
        if _role(name) != want:
            continue
        if images != ("image" in name):
            continue
        return payload
    if required:
        role = "train" if train else "test"
        kind = "images" if images else "labels"
        msg = f"MNIST archive is missing {role} {kind}."
        raise ValueError(msg)
    return None


def _role(name: str) -> str | None:
    """Classify an archive member as train or test from its file name."""
    lowered = name.lower()
    if "t10k" in lowered or "test" in lowered:
        return "test"
    if "train" in lowered:
        return "train"
    return None


def _images(payload: bytes) -> tuple[NDArray[np.uint8], tuple[int, int]]:
    """Parse an image IDX record into ``(n, height, width)``."""
    magic, n_images, height, width = struct.unpack(">IIII", payload[:16])
    if magic != _IMAGE_MAGIC:
        msg = f"Unexpected image magic {magic}; expected {_IMAGE_MAGIC}."
        raise ValueError(msg)
    count = int(n_images * height * width)
    raw = np.frombuffer(payload[16 : 16 + count], dtype=np.uint8)
    if raw.size != count:
        msg = "Image IDX file is shorter than its header claims."
        raise ValueError(msg)
    frames = np.array(raw, copy=True).reshape(n_images, height, width)
    return frames, (int(height), int(width))


def _labels(payload: bytes) -> NDArray[np.uint8]:
    """Parse a label IDX record."""
    magic, n_labels = struct.unpack(">II", payload[:8])
    if magic != _LABEL_MAGIC:
        msg = f"Unexpected label magic {magic}; expected {_LABEL_MAGIC}."
        raise ValueError(msg)
    raw = np.frombuffer(payload[8 : 8 + int(n_labels)], dtype=np.uint8)
    if raw.size != int(n_labels):
        msg = "Label IDX file is shorter than its header claims."
        raise ValueError(msg)
    return np.array(raw, copy=True)


def _maybe_gunzip(payload: bytes) -> bytes:
    """Decompress a gzip blob. Other bytes pass through."""
    if payload[:2] == b"\x1f\x8b":
        return gzip.decompress(payload)
    return payload
