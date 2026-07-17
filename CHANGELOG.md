# Changelog

All notable changes to TeoTensor are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Point CI triggers, project URLs, and docs `repo_url` at the organization
  repository (`Teoglobal-Solutions-LTD/teotensor`) and the `develop` branch.
- Upgrade `export_html` to a denser one-page observation report with a
  Print / Save as PDF action, print stylesheet, full-width solo panels,
  redesigned SVG charts (area fills, soft grid), and hover tooltips on chart
  marks. HTML dashboards are the primary visualization surface; matplotlib
  remains an optional secondary backend.

### Added

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
