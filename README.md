# TeoTensor

**A fast, inspectable ML systems framework for Python.**

> Guiding principle: *every model is inspectable, every hot path is compilable.*

TeoTensor is an educational yet production-minded machine-learning framework
that unifies three ideas: a single sklearn-style API, deep observability of a
model's internal state (`trace` / `report` / `diagnose` / `visualize`), and
performance-critical paths designed to be accelerated later (Numba, Cython,
Pythran, native backends) without breaking the public interface.

## Status

Pre-alpha. The framework is being built incrementally; the public API and
documentation grow with each release.

## Installation (development)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,viz,docs]"
```

## Quality gates

```powershell
ruff check .        # lint
ruff format --check .  # formatting
mypy                # strict static typing
pytest              # tests
```

## Layout

```text
teotensor/          # the importable framework package
  core/             # base estimators, mixins, validation
  artifacts/        # trace / diagnostic / report / figure specs
  observability/    # shared inspection protocol
  backends/         # pluggable / accelerated compute
  classical/        # PCA, KMeans, DBSCAN, trees, ensembles, metrics
  nn/               # tensor, autograd, layers, losses, optim, training
  vision/ attention/ transformers/ rag/ integrations/ viz/
tests/ examples/ benchmarks/ docs/
```

## Documentation

- Local preview: `mkdocs serve` → http://127.0.0.1:8000
- Public site (GitHub Pages): after enabling Pages → Source **GitHub Actions**,
  each push to `develop`/`main` rebuilds and publishes the docs via the
  `Docs` workflow (`.github/workflows/docs.yml`).
  Expected URL:
  https://teoglobal-solutions-ltd.github.io/teotensor/

## License

Apache-2.0. See [LICENSE](LICENSE).
