# Changelog

All notable changes to TeoTensor are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Dataset`: one index per row. CSV, MNIST IDX zip, and in-memory arrays
  fill it. Splits and minibatches move indices. `scale` is a per-row map;
  `Standardize` is fit on training indices only. The trainer batches with
  those indices, and the studio trains on `train_idx`.
- Multilayer perceptron (`MLPClassifier`, `MLPRegressor`) on a NumPy
  reverse-mode tape: layers, losses, penalties, SGD / Adam / RMSprop,
  schedules, traces, diagnostics, playback frames, and `.ttw` weight files.
  PCA can export the same weight file. `open_studio()` serves a localhost
  page for PCA and both MLPs.

### Removed

- The PCA test that compared results with scikit-learn. The `dev` extra no
  longer installs scikit-learn.

### Changed

- The studio dataset screen picks one source: pasted rows, a file, or an
  http(s) URL. Image rows preview in a table column sized from the frame.
- The studio walks a session: dataset, then model, then train or infer.
  Scaling is a divisor you type. Inference loads a `.ttw` file and scores
  every row.
- Point CI triggers, project URLs, and docs `repo_url` at the organization
  repository (`Teoglobal-Solutions-LTD/teotensor`) and the `develop` branch.
- Upgrade `export_html` to a denser one-page observation report with a
  Print / Save as PDF action, print stylesheet, full-width solo panels,
  redesigned SVG charts (area fills, soft grid), hover tooltips on chart
  marks, compact metric formatting, section ledes, and `?` help balloons.
  HTML dashboards are the primary visualization surface; matplotlib
  remains an optional secondary backend.

### Added

- Phase 3 (start): `teotensor.classical.decomposition.PCA` — economy SVD
  baseline with sklearn-compatible attributes, variance-threshold
  `n_components`, reconstruction metrics, scree / cumulative figures,
  diagnostics, HTML dashboard example (`examples/pca_dashboard.py`), docs
  page, and synthetic parity tests vs scikit-learn.
- GitHub Actions workflow to build MkDocs and publish documentation to GitHub
  Pages (`.github/workflows/docs.yml`), plus `site_url` in `mkdocs.yml`.
- Phase 2 observability MVP: artifact types (`Trace`, `Diagnostic`, `Report`,
  `Observation`, `FigureSpec`, `TableSpec`), `export_json` /
  `export_markdown_summary` / `export_html` (browser dashboard with inline SVG),
  `ObservabilityMixin` on `BaseEstimator`, optional matplotlib `render_figure`,
  dummy examples/tests, and the Observability docs page.
- Phase 1 Core API: `BaseEstimator`, parameter mixins
  (`ParamMixin`, transformer/classifier/regressor/cluster/serializable),
  validation helpers (`check_array`, `check_X_y`, `check_random_state`,
  `check_is_fitted`), unit tests, `examples/core_dummy_estimators.py`, and
  the Core API docs page.
- Phase 0 bootstrap: project scaffolding, packaging (`pyproject.toml` with
  hatchling), tooling configuration (ruff, mypy strict, pytest, coverage),
  pre-commit hooks, GitHub Actions CI, MkDocs Material docs skeleton, and the
  full layered `teotensor/` package layout.
