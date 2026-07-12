"""Backends layer: pluggable compute implementations behind stable interfaces.

Hosts the backend-abstraction API and, over time, accelerated kernels
(Numba, Cython, Pythran, ANN libraries). A pure-NumPy baseline is always the
reference; accelerated backends must never change public behavior.
"""

from __future__ import annotations
