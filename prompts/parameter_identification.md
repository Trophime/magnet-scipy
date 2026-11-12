# Identifying R and L (M) from experimental data

## Goal

Estimate, from site records:

- the per-element resistances `R_a(I, T)` [Ω],
- the inductive coupling seen by each element, `K = M_elem · P` [H]
  (n_elem × N, see [element_voltages.md](element_voltages.md)), and the
  circuit matrix `M = Pᵀ M_elem P`,

with uncertainties. Then run a sensitivity analysis of the simulator to these
parameters. Compare the results with the magnettools inductances and with the
`R_<circuit>.csv` tables.

No adjoint solver. The parameter count is small (p ≈ n_elem + n_elem·N), and the
voltages are measured, so most of the problem is **linear least squares
(equation error)**. The ODE is used for validation and for sensitivities
(forward mode).

## Data

| Source | Rate | Signals | Limits |
|---|---|---|---|
| pupitre `.txt` | 1 Hz, timestamps to 1 s (checked: `M9_2023.09.16---17:55:22.txt`, 1553 × 1 s steps, one 11 s gap) | `Idcct*`, `Icoil*`, `Ucoil1..16` (per element), `Tin1`, `Tin2`, `Tout` (**mixed**), `Flow1/2`, `HP/BP`, `teb/tsb/debitbrut` | `Ucoil` written to 10 mV; time skew between I and U unknown within ±0.5 s; ramps ≤ 70 A/s in that run |
| pigbrother `*_Archive_*.tdms` | 120 Hz (`wf_increment` = 1/120 s; M19: 1 940 880 samples = 16 174 s) | supply currents/references, supply voltages, `Tensions_Aimant` | no temperatures; 445 MB per file; no TDMS reader in the package (`nptdms` not installed) |
| pigbrother `*_Overview_*.tdms` | 1 Hz | same channels as Archive, same `wf_start_time` | |

### Channel correspondence (M9 housing, M23072801 + M9Bitters_18MW)

Sources: `python_magnetrun/pupitre-defs.json`, `pigbrother-defs.json`
(`aliases`), and `M9-housing-config.json` (housing-dependent GR roles). The
channel lists were checked in the 2023 M9 Overview TDMS files.

| Quantity | pupitre | pigbrother | magnet-scipy element |
|---|---|---|---|
| supply currents | `Idcct1..4` | `Courants_Alimentations/Courant_A1..A4` | |
| GR1 current (helices) | `IH = Idcct1 + Idcct2` (= `Icoil1..7`) | `Courant GR1` | circuit `M23072801` |
| GR2 current (Bitters) | `IB = Idcct3 + Idcct4` (= `Icoil15/16`) | `Courant GR2` | circuit `M9Bitters_18MW` |
| reference GR1/GR2 | — | `Référence_GR1/2` (= `Référence_A1+A2`, `A3+A4`) | `i_ref` of the PID |
| helix pair taps | `Ucoil1..7` | `Tensions_Aimant/Interne1..7` | `H1H2 … H13H14` (`R_14Helices.csv` order) |
| Inner Bitter tap | `Ucoil15` | `Tensions_Aimant/Externe1` | `M9_Bi` |
| Outer Bitter tap | `Ucoil16` | `Tensions_Aimant/Externe2` | `M9_Be` |
| circuit voltage H | `UH` | `Tensions_Aimant/ALL_internes` | |
| circuit voltage B | `UB` | `Tensions_Aimant/ALL_externes` | |
| supply voltages | — | `Tensions_Alimentations/Tension_A1..A4` | (includes busbars) |
| field | `Field` [T] | `Courants_Alimentations/Champ_magn` [**mT**] | |
| inlet temperatures | `Tin1` (GR1 = H), `Tin2` (GR2 = B) | — | `Tinit[C]` |
| outlet temperature | `Tout` (mixed) | — | |
| flows | `Flow1` (H), `Flow2` (B) [l/s] | — | |

Notes:

- `Ucoil8..14` / `Icoil8..14` are used for 14 single-helix taps. With 7 pairs
  they are 0 (checked in `M9_2023.09.16---17:55:22.txt`).
- **Bitter order mismatch**: `R_M9Bitters_18MW.csv` has columns
  `R_M9_Be, R_M9_Bi` (outer first), while taps are `Externe1` = Inner,
  `Externe2` = Outer. The mapping must be by name, not by position (this also
  affects the positional `B_k` rule in [element_voltages.md](element_voltages.md)).
- Channel names on disk differ from the defs keys (`ALL internes` with a space
  vs `ALL_internes`; housing prefixes such as `M9Externe1` in multi-housing
  sites). Reuse python_magnetrun's normalisation rather than reimplementing it.
- The channel set depends on the site configuration (see M19 below). Check
  every Archive file before using it for per-element K.
- python_magnetrun shifts the time stamps by half a sample (`t_offsets`:
  Overview 0.5 s, Archive 1/240 s in `readers/tdms_reader.py`). That implies
  the samples are interval averages stamped at the interval centre. If so,
  the integral form of B is exact on them (decision 4).
- The Archive and Overview files share `wf_start_time`, so they are
  synchronous. Pupitre is a separate system with a wall clock to 1 s
  (decision 5).

### M19 (2024-02-08): M9 + M10, Bitters only

M19 was a test run for the new 30 MW installation: the M9 and M10 housings
were operated at the same time, each assembly holding only a Bitter magnet
(Inner + Outer, no helix insert). `Infos/Pmax` = 31.5 MW.

Wiring read from `M19_Overview_240208-0941.tdms` (checked on the data):

| Supply group | Magnet | Taps | Circuit voltage | Peak |
|---|---|---|---|---|
| GR2 (`A3 + A4`) | M9 Bitters | `M9Externe1/2` | `ALL externes` | 553 V × 32.4 kA ≈ **17.9 MW**, R ≈ 17.1 mΩ |
| GR1 (`A1 + A2`) | M10 Bitters | `M10Externe1/2` | `ALL externes 2` | 411 V × 31.7 kA ≈ **13.1 MW**, R ≈ 13.0 mΩ |

- `Externe1 + Externe2 = ALL externes` to 6e-5 V, for both magnets.
  `ALL internes` and `ALL internes 2` are identically 0 (no inserts).
- Correlation of the taps with the group currents: `M9Externe1`/GR2 0.9992,
  `M10Externe1`/GR1 0.9994 (vs 0.97 for the crossed pairs).
- `Tension_A3/A4` − `ALL externes` ≈ 10 V at peak (busbars).
- **Power vs. the stated split (12.5 MW M9, 18 MW M10)**: the channels named
  `M9*` carry ≈ 18 MW and `M10*` ≈ 13 MW. Either the M9/M10 labels in the
  TDMS are swapped, or the split is the other way round. The existing
  example name `M9Bitters_18MW` agrees with the data labels (decision 6).
- `examples/M19/M19.json` refers to `Rtot_M10Bitters.csv`, which is not in
  the repo. Its M10 `inductance` (0.01254 H) is copied from M9, so it is a
  placeholder.

What M19 gives for identification:

- 120 Hz on both magnets: K is 4 × 2 (Inner/Outer × M9/M10 circuits). This is
  the best Archive available today for method B.
- **Built-in null test**: two separate housings, so the cross coupling
  `M_{M9,M10}` is ≈ 0. The estimate of the cross column of K must be 0
  within its confidence interval. If it isn't, that points to skew,
  cross-talk or a grounding issue, not to physics.
- GR1 and GR2 currents are 0.97-correlated over the run, so check from the
  SVD that the cross column is identifiable at all. If it isn't, fix it to 0
  and use the M9/M10 diagonal blocks only.
- No temperatures: R comes from method A only if pupitre files for M9 and
  M10 exist for that date.

The circuit time constant is τ = L/R ≈ 12.54 mH / 16.7 mΩ ≈ 0.75 s (M9 Bitters).
The closed loop is much faster than τ (≈ 1–2.5 ms, see
[diffrax_demonstrator.md](diffrax_demonstrator.md)).

## Key constraints

### 1. Timing skew is aliased into L (why L can't come from pupitre)

If the recorded current lags the true one by δ, then
`I_m(t) ≈ I(t) − δ·dI/dt`, and

```
U = R·I_m + (L + R·δ)·dI/dt
```

So the fitted inductance is `L + R·δ`, a relative bias of **δ/τ**. With pupitre's
1 s timestamps (δ unknown in ±0.5 s), the error on L can reach ±70 %. δ and L
are confounded: both terms are ∝ dI/dt, so a fit can't separate them.

At 120 Hz on a single acquisition system, δ ≲ 8 ms and the bias is ≲ 1 %.

→ **R from pupitre plateaus** (dI/dt = 0, so δ doesn't matter, and temperatures
are available). **L/M from pigbrother** (R treated as a slowly varying nuisance).

On plateaus, 1 Hz is not a limitation: there is no inductive term. On ramps,
the 1 Hz rate only matters near the corners (`dI/dt` jumps). Elsewhere `L·dI/dt`
is a constant offset.

### 2. What is identifiable

- All elements of a circuit carry the same current, so they can't be excited
  independently. The full `M_elem` (n_elem × n_elem) is **not identifiable**.
  Only `K = M_elem·P` (one column per circuit) is, from the element taps. That
  is exactly what the element-voltage prediction needs (`U_L_own`, `U_M_other`).
- Each column of K (`K[:, B]`) needs `dI_B/dt` to vary **independently** of the
  other circuits. Corners of GR1 and GR2 at the same time, with proportional
  slopes, make the columns collinear. The SVD of the regressor matrix gives this
  verdict per record.
- Circuit level: `L_A = Σ_{a∈A} K[a, A]`, `M_AB = Σ_{a∈A} K[a, B]`. Two
  estimates of `M_AB` (from A's taps and from B's taps) give a symmetry check.

### 3. Temperature

R depends on the conductor temperature, not on `Tin`. `Tout` is the mixed
outlet of both circuits, so per-circuit outlet temperatures must be
reconstructed by calorimetry:
`Tout_k = Tin_k + P_k / (ρ·c_p·Q_k)`, with `P_k = Σ_a U_a·I` measured. Check
against the mixed `Tout` using `Flow1/2`. This ties in with tier 1 of
[cooling_model.md](cooling_model.md). The R fit is an *effective* relation in
`(I, Tin_k, Tout_k)`. Use plateau tails (after thermal settling) to stay out of
the conductor's thermal transient.

## Methods

### A. R from pupitre plateaus (direct, no inverse problem)

Plateau windows: `|dI/dt| < ε` for ≥ `t_settle`, keeping the tail of each window.
On each window, `R_a = ⟨U_a⟩ / ⟨I⟩`. Then fit a model
`R_a(I, T̄)` (form: decision 3) and compare it with `R_<circuit>.csv`.

### B. K from pigbrother (integral equation error)

For element `a` and windows `W_k = [t_k, t_k + Δ]` (Δ ≈ a few s, overlapping
the corners), use sub-intervals `[s_j, s_{j+1}] ⊂ W_k`:

```
∫ U_a dt = R_{a,k} ∫ I_A dt + Σ_B K[a,B] · (I_B(s_{j+1}) − I_B(s_j)) + c_{a,k}·(s_{j+1} − s_j)
```

- Unknowns: `R_{a,k}` (one per window, as a nuisance, since there is no
  temperature), `c_{a,k}` (an optional tap offset), and the shared `K[a, :]`.
  This is linear, and one row of K per element is an independent problem.
- No numerical differentiation: only the increments ΔI and the integrals of
  U and I.
- The information is at the corners: the jump in `U_a` equals
  `Σ_B K[a,B] · Δ(dI_B/dt)` while `R·I` stays continuous.
- Output: K, its covariance from the residuals (bootstrap over windows), and the
  SVD/condition number of the regressor. The skew is assumed ≲ 1 sample
  (same acquisition system). δ stays confounded with K, as in constraint 1
  (only weakly separable through the variation of R across windows), so do not
  fit it here.

Optional: use K from B to estimate pupitre's δ, then the pupitre ramps become
usable as a cross-check.

### C. Validation with the ODE

Drive the existing `VoltageInputStrategy` with the measured circuit voltages
(open loop, so there are no PID-model errors), with the identified R and K.
Compare the simulated I with the measured one, and the element voltages
(`element_voltages`) with the taps. Then run the closed-loop PID simulation
against the pigbrother references.

### D. Sensitivity analysis

Forward sensitivities `S = ∂y/∂p`, with p = (scale of `R_a`, entries of K, and
optionally `Kp`, `Ki`):

- diffrax: `jax.jacfwd` through `diffeqsolve(Kvaerno5)` on sequences (reuses the
  [diffrax_demonstrator.md](diffrax_demonstrator.md) vector field), or
- scipy fallback: the augmented system `dS/dt = J_y S + J_p` (Jacobians by hand).

Use: rank parameters by their effect on I(t) and U_a(t); Fisher information
`SᵀS/σ²` for experiment design (which reference profile makes `M_HB`
identifiable). If a **global** analysis is wanted instead (sampling with SALib),
it needs no derivatives.

### E. Output error (only if needed)

If the residuals of B/C show structure: run `scipy.optimize.least_squares`
on the simulated vs measured I/U, with `jac` from D, starting from A+B.
L stays SPD through a Cholesky parameterization, and log scales are used.
An adjoint is only warranted if p grows a lot (an R grid per element, or a
T(t) field from the cooling model). Then use diffrax `RecursiveCheckpointAdjoint`,
**not** `BacksolveAdjoint` (unstable on stiff problems).

## Plan

New module `magnet_scipy/identification.py`. Tests go in
`tests/test_identification.py`.

1. **Synthetic recovery tests (fail first).** Simulate 2 circuits (2 + 1
   elements) with known R and `M_elem` in voltage mode, using a reference with
   non-simultaneous corners. Sample at 120 Hz with noise, and at 1 Hz with a lag
   δ.
   - A recovers R_a to < 0.5 %.
   - B recovers K to < 2 % at 120 Hz. The SVD flags rank deficiency when the
     corners are simultaneous and the slopes proportional.
   - At 1 Hz with δ = 0.3 s, B's L estimate is biased by ≈ δ/τ (documents
     constraint 1).
   → verify: tests written, failing for the right reason.
2. **Readers** (per decision 1): pupitre `.txt` → DataFrame with a time
   index; pigbrother → per-channel arrays. Use the tap ↔ element mapping
   (decision 2).
   → verify: load `M9_2023.09.16---17:55:22.txt`; `Σ Ucoil(H) ≈ UH`.
3. **Plateau detection + method A.**
   → verify: synthetic test passes; real M9 run, compare `R_a` against
   `R_14Helices.csv` at the measured `(I, Tin)`.
4. **Method B** (windows, regressor, LS, bootstrap, SVD report).
   Data: an M9 **Archive** file with `Interne1..7` is needed. The repo only
   has the M9 2023 Overview files (1 Hz) and the M19 2024 Archive (no
   `Interne*`). The DB lists one: record `M9_Overview_230916-1755` →
   `sources_archive = M9_Archive_230916-1755.tdms` (assembly `M9_A230801_00`).
   The case directory of [ode_config_from_magnetdb.md](ode_config_from_magnetdb.md)
   is the intended way to fetch it. Meanwhile, start with the M19 Archive (Bitters only, 4 × 2 K, with the
   cross-housing null test), then M9 helices when an Archive is available.
   → verify: synthetic tests pass; real pigbrother run, compare
   `M = Pᵀ K` with magnettools `<stem>_summary.csv`, check the symmetry of `M_AB`.
5. **Validation C**, then sensitivities D (diffrax first, if the demonstrator is in).
6. **Docs**: theory section (skew bias, identifiability of K vs M_elem).

## Decisions

Open:

1. **Readers**: python_magnetrun already has both readers, the defs, the
   housing configs and the channel-name normalisation. Use it as an optional
   dependency, or use CSV exports from it? (`nptdms` alone is not enough: the
   name normalisation and GR roles live in python_magnetrun.)
2. ~~Tap ↔ element mapping~~: resolved (2026-11-09) from python_magnetrun's
   defs, see "Channel correspondence". Still to confirm: is the
   `Externe1` = Inner (`M9_Bi`) / `Externe2` = Outer (`M9_Be`) wiring valid
   for every M9 assembly? The housing config has a single version since
   2018-11-09.
3. **R model**: `R_a0·[1 + α_a(T̄ − T0)]` plus a magnetoresistance term in I, or
   gridded `(I, T)` values?
4. **Sampling**: instantaneous samples or interval averages? python_magnetrun's
   half-sample offsets suggest averages for pigbrother. Pupitre is unknown. If
   U and I are averaged over the same window, the integral form of B is exact
   on the averages.
5. **Clock alignment** pupitre ↔ pigbrother (needed to combine temperatures
   with 120 Hz voltages): cross-correlation on the currents?
6. **M19 M9/M10 power split**: the data labels give M9 ≈ 18 MW, M10 ≈ 13 MW,
   which is the reverse of the stated 12.5 / 18 MW. Are the labels swapped?
   Are there pupitre files for M9 and M10 on 2024-02-08?

## To check

- Spectrum of U on a pigbrother plateau: if the converter ripple is not
  anti-alias filtered, it may fold onto low frequencies at 120 Hz.
- Calorimetric closure on long plateaus: `Σ U_a·I` vs `ρ c_p (Q1 + Q2)(Tout − T̄in)`.
  This checks the taps as much as the thermometry.
