# Contributing to TeoTensor

Thank you for your interest in TeoTensor. The framework is built incrementally:
core API first, then algorithms, observability, and performance layers.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,viz,docs]"
pre-commit install
```

## Engineering rules

- **API discipline first.** Follow the sklearn-style estimator invariants:
  hyperparameters set in `__init__` (no logic there), learned attributes end
  with `_`, stochastic algorithms take `random_state`.
- **Observability is not optional.** Every model exposes `report()` and
  `diagnose()`; iterative models add `trace()`. Models never draw plots
  directly — they return data or a `FigureSpec`.
- **Correctness before speed.** Ship a pure-NumPy baseline first; add
  accelerated backends only after profiling proves the need.
- **Typing everywhere.** Code must pass `mypy --strict`.
- **English docs.** All docstrings and comments are written in English, NumPy
  docstring style.

## Definition of Done for a module

Working code + unit tests + at least one example + a docs page + `report()` +
`diagnose()` (+ `trace()` if iterative, + `visualize()` if visual).

## Before opening a PR

```powershell
ruff check .
ruff format --check .
mypy
pytest
```

All four must pass. Update `CHANGELOG.md` under `[Unreleased]`.

## Documentation site

- Preview locally with `mkdocs serve`.
- Public docs are built by `.github/workflows/docs.yml` on every push to
  `develop`/`main` and published to GitHub Pages.
- One-time setup in the GitHub UI: **Settings → Pages → Source =
  GitHub Actions**.
