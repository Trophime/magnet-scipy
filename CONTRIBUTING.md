# Contributing to magnet-scipy

Thanks for your interest in improving magnet-scipy. This guide covers how to
set up a development environment, the conventions the code follows, and what a
pull request needs before it can be merged.

## Development setup

```bash
git clone https://github.com/Trophime/magnet-scipy.git
cd magnet-scipy
uv sync                    # package + dev tools in .venv
```

Without [uv](https://docs.astral.sh/uv/):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
uv sync --extra jax        # JAX/diffrax stack, only for tests marked `jax`
uv sync --extra docs       # Sphinx toolchain   (pip: pip install -e ".[docs]")
```

If you change dependencies in `pyproject.toml`, run `uv lock` and commit the
updated `uv.lock` together with the change.

## Workflow

1. Fork the repository and create a branch from `main`
   (`git checkout -b feature/short-description`).
2. Make focused changes: one topic per pull request. Avoid reformatting or
   refactoring code unrelated to your change.
3. Add or update tests. For a bug fix, write a test that fails before the fix.
4. Run the checks below and make sure they pass.
5. Open a pull request describing *what* changed and *why*. For changes that
   affect simulation results, say how you validated them (test case,
   comparison with experimental data, …).

## Checks

```bash
uv run pytest                                     # test suite with coverage
uv run black magnet_scipy tests                   # formatting (line length 88)
uv run ruff check magnet_scipy tests              # linting + import order
uv run ty check                                   # type checking
uv run bandit -c pyproject.toml -r magnet_scipy   # security linting
```

`uv run ruff check --fix` sorts imports and removes unused ones
automatically. Ruff is configured in `[tool.ruff]` in `pyproject.toml`.

Useful test selections:

```bash
uv run pytest -m "not slow"     # skip slow tests
uv run pytest -m unit           # unit tests only
uv run pytest tests/test_csv_utils.py -k grid
```

Available markers are declared in `[tool.pytest.ini_options]` in
`pyproject.toml`; `--strict-markers` is enabled, so new markers must be
registered there.

## Code conventions

### Naming

New code follows PEP 8:

| Construct | Style | Example |
|---|---|---|
| Modules / files | `snake_case` | `csv_utils.py` |
| Classes | `PascalCase` | `RLCircuitPID` |
| Functions / methods / properties | `snake_case` | `get_pid_parameters` |
| Constants | `UPPER_SNAKE` | `DEFAULT_TIME_STEP` |
| Private helpers | `_snake_case` | `_validate_regions` |

Some legacy public methods use camelCase. Do not rename them outright, since
downstream code depends on them. Use a deprecation shim instead:

```python
import warnings

def get_keys(self) -> list[str]:
    """Return the list of data column keys."""
    ...  # real implementation

def getKeys(self) -> list[str]:  # noqa: N802
    warnings.warn(
        "getKeys() is deprecated, use get_keys() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.get_keys()
```

Update internal callers to the new name when you touch a file. Shims are
removed only on a major version bump.

### Docstrings

All docstrings use **NumPy style** (rendered by `sphinx.ext.napoleon`). Do not
use Google style or reST field lists (`:param:`).

- Public functions and classes: at least a summary line, `Parameters` and
  `Returns`; add `Raises` where relevant.
- Private helpers: a one-line summary is enough.
- Give physical units in brackets: `[A]`, `[V]`, `[Ω]`, `[H]`, `[s]`, `[K]`.
- Cross-reference well-known types, e.g. `` :class:`~numpy.ndarray` ``,
  `` :class:`~pandas.DataFrame` ``.

```python
def compute_voltage(current: float, resistance: float) -> float:
    """Compute the resistive voltage drop.

    Parameters
    ----------
    current : float
        Circuit current [A].
    resistance : float
        Circuit resistance [Ω].

    Returns
    -------
    float
        Voltage drop [V].
    """
```

## Documentation

The documentation uses Sphinx with MyST-Parser, so pages are Markdown files in
`docs/`. The API reference is generated from docstrings.

```bash
cd docs
make html SPHINXBUILD=../.venv/bin/sphinx-build   # output in docs/_build/html
```

Please make sure your change does not add new Sphinx warnings.

## Reporting issues

Open an issue on GitHub with:

- what you ran (command line or minimal script, plus the configuration file),
- what you expected and what happened (full traceback if any),
- your Python version and `pip list` / `uv pip list` output for the core
  dependencies (scipy, numpy, pandas, matplotlib).

For simulation discrepancies, attaching the input CSV files (or a reduced
version) makes the problem much easier to reproduce.
