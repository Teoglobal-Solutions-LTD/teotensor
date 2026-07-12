"""Smoke tests ensuring the package imports and basic invariants hold."""

from __future__ import annotations

import re

import teotensor


def test_package_imports() -> None:
    assert teotensor is not None


def test_version_is_pep440_like() -> None:
    assert isinstance(teotensor.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", teotensor.__version__)
