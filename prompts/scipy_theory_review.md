# Review: SciPy implementation vs `theory_documentation.md`

Date: 2026-10-20 — state of the working tree on `main` (uncommitted changes included).

Scope: model ([rlcircuitpid.py](../magnet_scipy/rlcircuitpid.py),
[coupled_circuits.py](../magnet_scipy/coupled_circuits.py),
[pid_controller.py](../magnet_scipy/pid_controller.py),
[csv_utils.py](../magnet_scipy/csv_utils.py)), solver path
([simulation_strategies.py](../magnet_scipy/simulation_strategies.py),
[single_circuit_adapter.py](../magnet_scipy/single_circuit_adapter.py),
[cli_simulation.py](../magnet_scipy/cli_simulation.py)) and post-processing
([plotting_strategies.py](../magnet_scipy/plotting_strategies.py)), compared with
[theory_documentation.md](../theory_documentation.md).

Numerical checks used a test reference `t = [0, 1, 2, 3, 5] s`,
`i_ref = [0, 100, 100, 1000, 1000] A`, three gain regions
(`(10, 5, 0.1) < 60 A`, `(15, 8, 0.05) < 800 A`, `(25, 12, 0.02)`),
`R = 0.5 Ω`, `L = 0.2 H`; coupled case: two identical circuits, `M₁₂ = 0.1 H`.

## What matches the note

- **Single-circuit equation:** `RLCircuitPID.vector_field` implements section 3 exactly:
  `(L+Kd)·di/dt = −(R+Kp)·i + Kp·i_ref + Ki·∫e + Kd·di_ref/dt`, state `[i, ∫e]`.
- **Coupled equation:** `CoupledRLCircuitsPID.vector_field` implements section 4,
  solving `(M + diag(Kd))·di/dt = …`. M is checked for symmetry and positive
  definiteness at construction and gains are validated `≥ 0`, so the "always
  solvable" claim holds at that point.
- **Gain scheduling:** region selected on `|i_ref|`, first threshold strictly greater
  (`<` comparison), as described.
- **`di_ref/dt`:** exact segment slope, right-continuous at breakpoints
  ([csv_utils.py:46-83](../magnet_scipy/csv_utils.py#L46-L83)).
- **Units:** Kp [Ω] adds to R, Kd [H] adds to L — dimensionally consistent.
- **Consistency:** voltage from the control law (with exact `de/dt`) equals
  `R·i + L·di/dt` to 3e-12 V.

## Code bugs (confirmed)

1. **Failed coupled run reported as success.**
   [cli_simulation.py:142-143](../magnet_scipy/cli_simulation.py#L142-L143) takes
   `success` from `result.metadata.get("success", True)`, but failed results carry
   `metadata={}` ([simulation_strategies.py:367-396](../magnet_scipy/simulation_strategies.py#L367-L396)).
   Reproduced: `success=True`, `strategy_type="failed"`, `error_message=None`.
   Fix: use `result.success` / `result.error_message`.

2. **Run past the end of the reference CSV fails.**
   `interp1d` is built without `bounds_error` / `fill_value`
   ([csv_utils.py:41](../magnet_scipy/csv_utils.py#L41)).
   `t_end = 5.5` on a 5 s reference →
   `ValueError: A value (5.000025…) in x_new is above the interpolation range's maximum value (5)`.
   With bug 1 a coupled run then fails silently. The derivative function clamps
   out-of-range times instead, so the two are inconsistent.
   `ValidationHelper.validate_time_range` ([cli_core.py:371-380](../magnet_scipy/cli_core.py#L371-L380))
   only prints warnings. Decide: clip `t_end` to the CSV range, or extrapolate as constant.

3. **Single-circuit initial current is ignored.**
   The note states `y₀ = [i₀, 0]`, but
   [cli_simulation.py:52](../magnet_scipy/cli_simulation.py#L52) hard-codes
   `initial_values=[0.0]`, and `load_and_validate_single_circuit` never uses its
   `initial_value` argument ([main_refactor.py:110-121](../magnet_scipy/main_refactor.py#L110-L121)).
   `--value_start` has no effect.

4. **Negative current: single and coupled models differ.**
   Single circuit passes the signed current to `get_resistance(i, T)`
   ([rlcircuitpid.py:428](../magnet_scipy/rlcircuitpid.py#L428), also
   `voltage_vector_field`); coupled uses `abs(i)`
   ([coupled_circuits.py:124](../magnet_scipy/coupled_circuits.py#L124)).
   With `i < 0`, the single circuit is clamped to the lowest current of the R table.

5. **Reconstructed voltage is inaccurate around discontinuities.**
   [plotting_strategies.py:431](../magnet_scipy/plotting_strategies.py#L431)
   computes `de/dt` with `np.gradient` on the non-uniform solver output times.
   Measured error: 18.1 V on 877 V (~2 %) at the 800 A gain switch. This voltage
   feeds the RMS/MAE comparison with experimental voltage.
   Exact alternative: `de/dt = di_ref/dt − vector_field(t, y)[0]` (one RHS call per
   output point; coupled case uses the first N components).

## Note out of date or inaccurate

| Note | Implementation |
|---|---|
| Default step reference ([lines 131-139](../theory_documentation.md#L131-L139)) | Does not exist: without a CSV, `reference_current` raises; PID mode requires `reference_csv`. |
| "`max_step = dt` … keeps the integrator from stepping over the discontinuities" ([line 182](../theory_documentation.md#L182)) | Overstated (see table below). Switching times are known before solving (note says so) but the code does not use them; restarting the solve at each one is correct and cheaper. |
| Voltage "reconstructed … from the control law" ([line 174](../theory_documentation.md#L174)) | Yes, but with a numerical derivative (bug 5). |
| Typical ranges R 1-10 Ω, L 0.01-1 H, … ([lines 199-204](../theory_documentation.md#L199-L204)) | Textbook values; inconsistent with the code defaults (60 A / 800 A thresholds) and with LNCMI magnets (tens of mΩ, kA). |
| Only the PID model is described | The **voltage-driven** model (`L·di/dt = u − R·i`, `M·di/dt = u − R⊙i`, state `[i]`) is implemented but undocumented. |
| `R(i, T)` | Not specified: T is the imposed **inlet** temperature (not a state); bilinear interpolation clamped at the table edges; grid stored as float32; **missing grid points silently set to 0 Ω** ([csv_utils.py:174](../magnet_scipy/csv_utils.py#L174)). |
| `--method` | CLI also accepts `RK23` and `DOP853`. |

Discontinuity handling, single circuit, `rtol=1e-6`, `atol=1e-9`, compared with a
reference solution restarted at every breakpoint and region switch
(`DOP853`, `rtol=atol=1e-12`):

| `max_step` | RHS evaluations | `|i(5) − i_exact|` [A] |
|---|---|---|
| ∞ | 1628 | 3.5e-4 |
| 0.1 s | 1694 | 2.6e-4 |
| 0.01 s | 3404 | 8.4e-5 |

## Open modelling question

All three gains switch together in `u = Kp·e + Ki·∫e + Kd·de/dt`, so `u` jumps at
every region change: 47.5 → 66.8 V at the 60 A switch and 563 → 877 V at the
800 A switch (the `ΔKi·∫e` part alone: 5.1 V and 78.9 V).
Digital regulators usually integrate `Ki·e` (bumpless transfer), which changes the
second state to `∫Ki·e dτ`. **Which form does the actual power-supply regulator
use?** The note should state the modelled form either way.

## Smaller points

- `update_mutual_inductance` and `add_circuit` modify M without re-running
  `_validate_coupling_matrix`
  ([coupled_circuits.py:264-311](../magnet_scipy/coupled_circuits.py#L264-L311)).
- The `pinv` fallback ([coupled_circuits.py:232-237](../magnet_scipy/coupled_circuits.py#L232-L237))
  is unreachable while M is valid and would silently give a wrong result if reached.
- `np.linalg.eigvals` → `eigvalsh` for the symmetric-matrix check
  ([coupled_circuits.py:101](../magnet_scipy/coupled_circuits.py#L101)).
- Coupled runner picks `hybrid` (→ `NotImplementedError`) when every circuit has
  both voltage and reference CSV; the single-circuit runner picks PID
  ([simulation_strategies.py:324-329](../magnet_scipy/simulation_strategies.py#L324-L329)).
- Dead `else` branch in `PIDControlStrategy.run_simulation`
  ([simulation_strategies.py:179-181](../magnet_scipy/simulation_strategies.py#L179-L181)).
- Closed-loop poles of the test case (`(L+Kd)s² + (R+Kp)s + Ki = 0`): about
  −35 / −61 / −115 s⁻¹ and −0.5 s⁻¹ per region — not stiff; tightly coupled M
  (coupling close to 1) may change that.

## Suggested next steps

1. Write regression tests for bugs 1-3 and 5, then fix them.
2. Answer the integral-form question, then update the note: remove the default
   step reference, correct the `max_step` claim, add the voltage-driven model,
   specify `R(i, T)`, replace the typical ranges.
3. Optionally: integrate piecewise between known breakpoints instead of relying
   on `max_step`.
