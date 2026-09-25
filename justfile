# Developer commands. Activate the project environment first so `python`
# resolves to the interpreter that has the dev extra installed.

# Rewrite sources with Ruff format.
fmt:
    python -m ruff format .

# Run pytest. Extra words are passed through, for example `just test tests/nn -q`.
test *args:
    python -m pytest --cov=teotensor --cov-report=term-missing {{args}}
