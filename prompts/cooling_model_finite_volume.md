# Cooling model, tier 2b: method-of-lines / finite-volume loop network

## Relationship to the other two documents

This is an alternative numerical route to the same problem as
[`cooling_model_node_method.md`](cooling_model_node_method.md) (tier 2a) — 1D
advection of water temperature along the loop's actual pipe network, replacing
tier 1's fitted relaxation constant `τ` with a real transit delay. It reuses the
same graph topology and the same tier-1 pieces unchanged: `CoolingBranch` (magnet
+ derivation, applied as a zero-delay algebraic jump at a node), flow-weighted
mixing at merges, the calorimetric (UA-free) exchanger boundary. What differs from
tier 2a is purely *how the passive transport segments are numerically
represented* — spatially discretized cells integrated live, instead of an exact
delayed lookup computed as post-processing.

## The scheme: upwind finite-volume, method of lines

Each `TransportSegment` (still characterized only by its total volume `V`, exactly
as in tier 2a — see below) is chopped into `N` equal-volume cells, `ΔV = V/N`. Pure
advection with flow always positive (pump-driven, no reverse flow) means an upwind
(backward) difference is the only stable choice — central differencing is
unconditionally unstable for pure advection with no diffusion term to stabilize
it. That turns the segment's PDE into `N` coupled ODEs (method of lines):

```
dT_k/dt = -(Flow(t)/ΔV)·(T_k(t) - T_{k-1}(t)) + S_k(t)/(ρ·Cp·ΔV)          k = 1..N
```

`T_0` is whatever arrives at the segment's inlet node right now (from a merge,
split, or branch outlet — the same junction-coupling logic as tier 2a). `S_k` is
zero for every cell in a pure transport segment; heat sources stay at nodes via
`CoolingBranch`, not distributed into cells, keeping this numerically identical to
tier 2a wherever the two should agree.

Working in **volume coordinates** (`Flow/ΔV` instead of `velocity/Δs`) means no
separate pipe length or cross-sectional area is needed — only the segment's total
volume, exactly the same single number tier 2a needs. That's deliberate: the two
tiers should require identical topology/geometry input, so choosing between them
is a modeling/numerics decision, not a data-availability one.

## Numerical behavior: diffusion instead of exactness

First-order upwind reproduces advection with **numerical diffusion**
(∝ `Flow·ΔV/2`, i.e. it shrinks as `N` grows) — a sharp step at the inlet arrives
at the outlet smeared over roughly `V/(N·Flow)` of extra spread. This is the
central trade-off against tier 2a: the node method reproduces a step exactly with
no smearing at any resolution; this scheme approaches that only in the `N → ∞`
limit, and too few cells per segment quietly reintroduces something that looks
like tier 1's exponential rounding — the exact failure mode this tier exists to
fix. `N` is therefore not a free/cosmetic parameter; it needs to be checked
against how sharp a transit-delay feature actually needs to look (see Tests).

Refinement path (not designed in detail here, just flagged): a flux-limited
scheme (e.g. minmod/van Leer, second-order upwind) would cut numerical diffusion
substantially for the same `N`, at real implementation cost — a reasonable
follow-up once first-order behavior is characterized against real data, not a
first-cut requirement.

## State vector integration — this is the real advantage over tier 2a

Every cell of every `TransportSegment` is one more ODE state, appended to the
*same* state vector already used by `RLCircuitPID`/`CoupledRLCircuitsPID`
(`[currents, integral_errors, ...transport cells...]`), following exactly the
pattern tier 1 established for its single lumped state — `get_initial_conditions`
gains `N_segment` more entries per segment, `vector_field` computes and appends
`N_segment` more derivatives per segment, plotting slices out the same trailing
block. Nothing new architecturally: just more scalar states of a kind the codebase
already knows how to carry through `solve_ivp`.

The practical consequence: **this scheme supports direct monolithic two-way
coupling** — a single `solve_ivp` call integrates currents, PID states, and every
transport cell together, `R(I,Tin)` feeding back live every step. Tier 2a's exact
delay lookup can't do that (evaluating a delay needs a value from the past, which
`solve_ivp` can't hand its own RHS mid-solve), so it needed a staggered/Picard
outer loop for predictive use. This scheme doesn't need that — at the cost of the
numerical diffusion above, and a materially larger state vector
(`Σ_segments N_segment` extra states vs. tier 2a's zero extra ODE states, since
tier 2a is pure post-processing).

For diagnostic/sensor-checking use (known measured current history, no live
electrical coupling needed), this scheme can equally be run as an isolated,
decoupled thermal-only `solve_ivp` — it isn't exclusively a "predictive" tool,
just strictly more flexible about *how* it's coupled than tier 2a is.

## Stability / step-size guidance

Explicit integration of pure advection is CFL-constrained: the solver's step must
respect `Flow(t)·Δt / ΔV ≲ 1` per segment, and since `Flow(t)` tracks magnet
current (tier 1's confirmed flow-control-law), the constraint is time-varying, not
a single fixed number to set once. In practice: bound `max_step` conservatively
against the *fastest* expected flow across all segments, or use one of the
already-exposed implicit methods (`--method BDF`/`Radau`) for better robustness
without hand-tuning — both are existing knobs in this package's CLI, nothing new
to add.

## Comparison with tier 2a (node method) — decision aid

| | Tier 2a: node method | Tier 2b: finite-volume MOL |
|---|---|---|
| Transit-delay fidelity | Exact, no smearing at any resolution | First-order numerical diffusion; improves with `N` |
| Extra ODE states | None (pure post-processing) | `Σ N_segment` (can be large) |
| Predictive coupling | Staggered/Picard iteration required | Direct, monolithic, single `solve_ivp` call |
| Diagnostic-only coupling | Deterministic pass over known current history | Same, run as a decoupled thermal-only solve |
| New machinery vs. existing codebase | Cumulative-flow integration, monotonic inversion, topological walk — genuinely new | More states through already-existing `vector_field`/`get_initial_conditions` plumbing — same shape as tier 1 |
| Stability constraints | None (exact evaluation) | CFL-type step constraint, time-varying with flow |
| Geometry data needed | Segment volumes only | Segment volumes only (identical, via volume-coordinate formulation above) |

Rough recommendation once both are on the table: tier 2a for validating whether
real sensor timing looks like a genuine delay at all (the sharpest, least
ambiguous test of that question); tier 2b if/when a live, monolithically-coupled
predictive simulation (feeding a properly delayed `Tin` back into `R(I,Tin)`
within one solve) is actually needed, accepting the numerical-diffusion trade-off
and picking `N` deliberately rather than by default.

## Tests: `tests/test_cooling_finite_volume.py`

- `TestUpwindConvergence`: single segment, constant flow, step inlet — measure how
  the smeared output sharpens toward tier 2a's exact delayed step as `N` grows;
  establishes the `N` needed for a given sharpness tolerance rather than guessing.
- `TestCFLStability`: confirm the scheme is unstable/oscillatory when `max_step`
  violates the CFL bound for a given `N`/flow, and stable once respected — turns
  the stability guidance above into an executable check.
- `TestJunctionMixing`, `TestCoolingBranch`: reuse tier 1's hand-calculated cases
  unchanged (same physics, shared code).
- `TestConsistencyWithNodeMethod`: same scenario run through both tier 2a and
  tier 2b; confirm convergence as `N → large`, cross-validating both
  implementations against each other rather than only against hand calculations.
- `TestMonolithicCoupling`: a small predictive run (one magnet + one segment) with
  transport cells embedded directly in the electrical system's state vector —
  confirm a single `solve_ivp` call suffices, no outer iteration, in contrast to
  tier 2a's Picard loop.

## Open items for next discussion

- Choice of `N` per segment — needs the same unknown segment volumes as tier 2a,
  plus a decision about acceptable numerical smearing once real transit-delay
  magnitudes are known.
- Whether a higher-order/flux-limited scheme is worth the implementation cost, or
  first-order upwind with enough cells is good enough for this use case.
- Whether the monolithic-coupling advantage actually matters yet — it only pays
  off once predictive (not just diagnostic) use is needed; per tier 2a's open
  items, that's not confirmed as a near-term requirement.
- Not yet decided which of tier 2a / tier 2b to actually build, if either — both
  are next-tier alternatives to be picked between (or deferred entirely) after
  tier 1 is running against real sensor data.
