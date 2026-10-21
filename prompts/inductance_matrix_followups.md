# Inductance matrix: remaining fixes

Follow-up to the inductance-matrix review (2026-10-21). Already done in that
session (not yet committed):

- `CoupledRLCircuitsPID.__init__` accepts either the packed upper-triangle vector
  or a full N×N matrix (diagonal checked against each circuit's `L`).
- `ConfigurationLoader` warns on unknown circuit keys (`"L"`, `"R"`, `"pid"`, …).
- `SampleConfigGenerator`, `examples/create_sample_config.py` and the README
  use `inductance`/`resistance`/`pid_params` and the packed vector.
- Tests: `tests/test_inductance_config.py`.

This plan covers what is left.

## Issues

| # | Issue | Where |
|---|---|---|
| A | `set_coupling_matrix` only takes a full N×N matrix, the constructor takes both forms; no diagonal check | [coupled_circuits.py:395](../magnet_scipy/coupled_circuits.py#L395) |
| B | `update_mutual_inductance` and `add_circuit` change `M` without re-validating; can leave a non-SPD matrix | [coupled_circuits.py:280-327](../magnet_scipy/coupled_circuits.py#L280-L327) |
| C | `M` diagonal and `circuit.L` can drift apart; nothing checks it after construction | `_validate_coupling_matrix` |
| D | Missing `mutual_inductances` key → loader returns `[]` → `None` guard in `coupled_main_refactor` is bypassed → misleading length error | [cli_core.py:295](../magnet_scipy/cli_core.py#L295), [coupled_main_refactor.py:119](../magnet_scipy/coupled_main_refactor.py#L119) |
| E | `create_example_coupled_circuits` passes a zero-diagonal N×N matrix (now rejected by the diagonal check); `create_custom_coupled_system` passes a 3rd argument the constructor does not take | [examples/create_coupled_circuits.py:66-113](../examples/create_coupled_circuits.py#L66-L113) |
| F | `np.linalg.eigvals` on a symmetric matrix; `eigvalsh` is the right call | [coupled_circuits.py:117](../magnet_scipy/coupled_circuits.py#L117) |
| G | `pinv` fallback in both vector fields is unreachable while `M` is SPD and would silently give a wrong `di/dt` if reached (also noted in `scipy_theory_review.md`) | [coupled_circuits.py:205-212, 248-253](../magnet_scipy/coupled_circuits.py#L205-L253) |

## Plan

Tests first in `tests/test_inductance_config.py`; each step ends with
`pytest tests/test_inductance_config.py` green.

1. **Tests** (all should fail first except where noted):
   - `set_coupling_matrix` accepts a packed vector and a full matrix; rejects
     a full matrix whose diagonal ≠ `L`.
   - `update_mutual_inductance` with `M_ij² ≥ L_i·L_j` raises and leaves `M`
     unchanged.
   - `add_circuit` with `L ≤ 0` raises and leaves the system unchanged.
   - Loader on a 2-circuit config without `mutual_inductances` → constructor
     raises with a message naming the key and the expected length.
   - `create_example_coupled_circuits(3, 0.05)` and
     `create_custom_coupled_system([...])` build without error
     (import from `examples/` via `sys.path`, like the manual check).

2. **Share the matrix builder (A).** Move the body that turns
   `mutual_inductances` into `M` out of `__init__` into a private
   `_build_inductance_matrix(self, mutual_inductances) -> np.ndarray`.
   `__init__` and `set_coupling_matrix` both call it, then
   `_validate_coupling_matrix()`. Update the `set_coupling_matrix` docstring
   (NumPy style).
   → verify: step-1 `set_coupling_matrix` tests pass.

3. **Validate on every change (B, C, F).**
   - `_validate_coupling_matrix`: add `np.allclose(np.diag(M), [c.L …])`,
     switch to `eigvalsh`.
   - `update_mutual_inductance`: keep the old value, assign, validate,
     restore the old value and re-raise on failure.
   - `add_circuit`: validate after rebuilding; on failure roll back the
     appended circuit/ID/`n_circuits`/`M`. (A block-diagonal extension with
     `L > 0` is always SPD, so in practice this only catches `L ≤ 0`.)
   - `remove_circuit`: no change (a principal submatrix of an SPD matrix is SPD).
   → verify: step-1 B/C tests pass.

4. **Missing coupling (D).** Loader: `config_data.get("mutual_inductances")`
   (returns `None` when absent), then `np.asarray` when present. Constructor:
   when `None` and N ≥ 2, raise
   `ValueError("mutual_inductances must be given: N(N-1)/2 = k values …")`.
   → verify: step-1 D test passes; M9/M19 example configs still load.

5. **Examples (E).** In `examples/create_coupled_circuits.py`:
   - `create_example_coupled_circuits`: pass
     `np.full(n * (n - 1) // 2, coupling_strength)`.
   - `create_custom_coupled_system`: drop the 3rd argument; when
     `mutual_inductances is None`, build the same packed vector from
     `coupling_strength`.
   - Convert both docstrings to NumPy style (`coupling_strength` is an absolute
     mutual inductance [H], not a coupling coefficient k).
   → verify: step-1 E tests pass.

6. **`pinv` fallback (G).** Replace with a `RuntimeError` naming `t` and the
   matrix condition. → verify: full test suite unchanged apart from the new
   tests (`test_csv_utils.py` has 10 failures + 4 errors already, unrelated).

7. **Docs.** Update README API section for `set_coupling_matrix` if it is
   listed; mention in `theory_documentation.md` §4 that `M` must be SPD and its
   diagonal equals the circuits' self-inductances.

## Decisions needed

1. **Missing `mutual_inductances` with N ≥ 2** — *error* (recommended: real
   nested magnets are never uncoupled, so silent zero coupling hides config
   mistakes) or *default to zero coupling with a warning*?
2. **`set_coupling_matrix` diagonal** — *must match `circuit.L`* (recommended:
   one source of truth) or *overwrite each `circuit.L` from the diagonal*?
3. **`add_circuit` coupling** — keep *zero coupling to existing circuits*
   (recommended, minimal) or add an optional `mutual_inductances` argument
   (length = current N)?
4. **`pinv` fallback (step 6)** — include here, or leave for the
   `scipy_theory_review.md` follow-up?

## Out of scope (noticed, not changed)

- `remove_circuit` docstring says "fewer than three remain" while the error
  says "need at least 2 circuits".
- Debug `print` calls in `__init__` (`loaded extra_diag terms`, `Setting mutual
  inductances …`).
