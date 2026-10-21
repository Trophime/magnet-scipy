# Diffrax demonstrator for the PID-regulated RL circuits

## Goal

Solve the **same** closed-loop ODE system as `scipy.integrate.solve_ivp` does today
(`RLCircuitPID.vector_field`, `CoupledRLCircuitsPID.vector_field`, see
`theory_documentation.md` §3–4) with diffrax, exploiting adaptive time stepping on
each **sequence** of the reference current — a time interval on which `i_ref(t)` is
constant (plateau) or linear (ramp).

The demonstrator is a comparison tool, not a new backend: the package stays
scipy-only; JAX/diffrax remain the optional `jax` extra.

Terminology (see `theory_documentation.md`, "Terminology"): *PID controller* = the
physical regulator (`Kp`, `Ki`, `Kd`); *step-size controller* = `diffrax.PIDController`.

## Why sequences

### What the reference looks like

`examples/M19/M19_Overview_240208-0941-Courants_Alimentations_Référence_GR2.csv`:

- 71 points → **70 sequences** over 16107 s
- durations: min 1 s, median 9.5 s, max 3625 s
- 33 plateaus (|slope| < 0.1 A/s), ramps up to 511 A/s
- `|i_ref|` from −1.7 A to 32 397 A, i.e. it crosses the 60 A and 800 A region thresholds

### What the closed loop looks like (M9Bitters_18MW, L = 12.54 mH, R ≈ 16.7 mΩ, Kd = 0)

Linearised state matrix `[[-(R+Kp)/L, Ki/L], [-1, 0]]`:

| Region | Kp, Ki | Fast τ | Slow τ | Stiffness ratio |
|---|---|---|---|---|
| low (< 60 A) | 5, 0.2 | 2.5 ms | 25 s | 1.0e4 |
| medium/high | 12, 1.0 | 1.04 ms | 12 s | 1.15e4 |

Consequences:

1. **The system is stiff.** An explicit solver (Tsit5, Dopri5, scipy RK45) is
   stability-limited to h ≲ 2–3 ms *everywhere*, i.e. ~10⁷ steps for the M19 run,
   whatever the adaptivity. Adaptive stepping only pays off with an **implicit**
   solver (`diffrax.Kvaerno5`, L-stable ESDIRK, Jacobian by autodiff): after the
   ~12 s transient of a sequence, steps can grow to the length of the plateau.
2. **Each sequence restarts the dynamics.** At a sequence boundary, `di_ref/dt`
   jumps (the `Kd·di_ref/dt` term, and the forcing of the integral state). Within a
   sequence, `i_ref` and `di_ref/dt` are smooth (affine / constant), so the only
   non-smoothness left is R(I, T) (bilinear in cells, mild).
3. **Steady ramp tracking error** of the PI loop is `slope · R / Ki` ≈ 8.5 A at
   511 A/s — a useful analytical check (see Validation).

## Design

### Sequence decomposition (pre-processing, NumPy)

1. Breakpoints = reference CSV times.
2. Within each ramp, add the times where `|i_ref(t)|` crosses a region threshold:
   `t* = t₀ + (±thr − a)/b` when it falls inside `(t₀, t₁)`. Gains are then constant
   on every sequence (gain scheduling depends on `i_ref`, not on `i`, so these
   times are known before solving).
3. For coupled circuits, the breakpoints are the union over circuits.

Each sequence `k` is fully described by `t₀, t₁` and, per circuit,
`a = i_ref(t₀)`, `b = slope`, `Kp, Ki, Kd` → passed as diffrax `args`.

### Vector field (JAX)

Same equations as the scipy code, with on a sequence:
`i_ref(t) = a + b·(t − t₀)`, `di_ref/dt = b` (exact, no interpolation).

```
(M + diag(Kd)) · di/dt = -(R(i,T) + Kp) ⊙ i + Kp ⊙ i_ref + Ki ⊙ ∫e + Kd ⊙ b
d(∫e)/dt = i_ref - i
```

- Linear solve: `M + diag(Kd)` is SPD → Cholesky (`jax.scipy.linalg.cho_factor`
  once per sequence, `cho_solve` in the vector field). No `pinv` fallback.
- R(I, T): bilinear interpolation re-implemented in JAX on the grid already built by
  `create_2d_function_from_csv` (`current_range`, `temp_range`, `R_grid`), with the
  same clamping rules, so both backends evaluate the same R.
- Temperature: constant per circuit (the config `temperature`). Variable `T(t)` from
  CSV is **out of scope** for the first version (would add breakpoints).
- Single circuit = coupled system with N = 1 (one code path).

### Integration loop

- One `diffrax.diffeqsolve` per sequence, `y0` = final state of the previous one.
- The solve function is `eqx.filter_jit`-compiled **once**: all sequences have the
  same array shapes, only `args`/`t0`/`t1`/`y0` change.
- Python loop over sequences (70 iterations; `lax.scan` not needed for a demonstrator).
- Solver: `Kvaerno5` by default, `Tsit5` selectable to exhibit the stiffness.
- Step-size controller: `PIDController(rtol, atol)` (same defaults as scipy:
  1e-6 / 1e-9); `dtmax` optional; `max_steps` sized generously.
- Output: `SaveAt(steps=True)` to expose the accepted steps (the point of the
  demonstrator), plus `SaveAt(ts=...)` on the experimental time grid for comparison.

### Reusing the package

- Load the system with `ConfigurationLoader.load_single_circuit` /
  `load_coupled_circuits` (`magnet_scipy/cli_core.py`) and extract arrays from it
  (`L`/`M`, PID regions `thresholds`/`kp_values`/…, R grid, reference data).
  No second JSON parser.
- Reference scipy run: the existing `PIDControlStrategy.run_simulation` on the same
  system object.

## Deliverable

- `examples/diffrax_demonstrator.py` — standalone script, NumPy-style docstrings,
  PEP 8. CLI: `--config-file`, `--solver {kvaerno5,tsit5}`, `--rtol`, `--atol`,
  `--compare-scipy`, `--scipy-method`, `--show-plots`/`--save-plots`.
- Report printed per run: number of sequences, accepted/rejected steps (total and per
  sequence), wall time (excluding JIT compile, reported separately), max |Δi| vs scipy.
- Plots: `i(t)` and `i_ref(t)`; step size `h(t)` on a log scale with sequence
  boundaries marked; difference vs scipy.
- No change to the `magnet_scipy` package.

## Validation

1. **Constant R, Kd = 0, single circuit**: on each sequence the system is LTI with
   affine forcing → exact solution by matrix exponential (augmented 4×4 system).
   Compare diffrax to it at sequence ends: error ≲ tolerance.
2. **Ramp steady state**: tracking error on long ramps → `slope · R / Ki`.
3. **scipy agreement** on `examples/M19/M9Bitters_18MW-pid.json` (single) and
   `examples/M19/2circuits-pid.json` (coupled, N = 2), with scipy `Radau` at tight
   tolerance as reference.
4. Small pytest for (1) in `tests/`, skipped when diffrax is not installed.

## Steps

1. Sequence decomposition + exact LTI reference → verify: test (1) passes.
2. JAX vector field + per-sequence solve, single circuit → verify: M19 single vs scipy.
3. Coupled N = 2 → verify: `2circuits-pid.json` vs scipy.
4. Report + plots → verify: step-size plot shows h growing on plateaus.

## Open questions

- Location: standalone `examples/diffrax_demonstrator.py` (proposed) or a
  `magnet_scipy/diffrax_backend.py` module usable from the CLI later?
- Should the scipy comparison keep today's `max_step = time_step` (what users run) or
  use a tight-tolerance `Radau` reference (what measures accuracy)? Proposed: both.
- Variable temperature `T(t)`: needed in the first version (e.g. `circuits-pid-temp.json`)?
- Plateau steps: leave `dtmax` unbounded (shows the gain best) or cap it to keep
  smooth plots?
