# Cooling model, tier 2: node-method loop network

## Relationship to the tier-1 plan

This builds on [`cooling_model.md`](cooling_model.md) (the 0D/lumped calorimetric
model) rather than replacing it. Tier 1 already established the pieces this plan
reuses unchanged:
- `CoolingBranch` — a magnet or the converter-derivation branch, each with a heat
  source `Q(t)` and a flow `Flow(t)`, producing `Tout_branch = Tin_local +
  Q(t)/(ρ·Cp·Flow(t))`. This algebra is unchanged here — the only thing that changes
  is what `Tin_local` means: in tier 1 it's one shared 0D loop state; here it's the
  temperature that has just arrived at that branch's specific point in the network.
- Flow-weighted mixing at merge points (`Σ Flow_k·T_k / Σ Flow_k`) — unchanged, just
  now applied at each real junction instead of assumed instantaneous everywhere.
- The calorimetric heat-exchanger boundary (`Q_hx` from measured `Teb`/`Tsb`/
  secondary flow, since UA is not reliably known) — unchanged, now a boundary
  condition on the network's exchanger node instead of a single algebraic step.
- `WaterProperties`, CSV-loading conventions, `experimental_data`/`exp_metrics`
  sensor-checking pattern — all reused as-is.

What tier 1 approximates with a single fitted relaxation time constant `τ` (an
exponential "rounding off" of any change), this tier replaces with the loop's
**actual transit delay** — a temperature change at a magnet arrives elsewhere in
the loop as a genuine delayed pulse, shaped by real pipe volumes and the
(current-dependent) flow through them. That's the specific gap flagged when
comparing the two: `τ` has no basis in loop geometry, and its response shape
(exponential) is qualitatively wrong if the real sensors show delay-then-arrival
rather than gradual rounding.

## Topology: a graph, not a ring

The loop is a small directed graph of pipe segments and junctions, not a single
closed curve: a header splits into the magnet branches (and the derivation
take-off) in parallel, they remerge into one pipe to the exchanger, and the return
pipe splits again to feed each magnet's inlet.

- **`TransportSegment`**: a passive pipe run between two nodes, characterized only
  by its volume `V` (m³) — pure plug-flow advection, no source term. Optional (off
  by default) ambient heat-loss coefficient for later refinement.
- **`CoolingBranch`** (from tier 1): sits at a node, not on a passive segment —
  reused as a zero-delay algebraic heat injection, since intra-magnet residence
  time is short relative to the loop's own transport segments and was already
  handled adequately by the tier-1 algebra. This is what keeps this tier additive
  rather than a rewrite: nothing about *how a magnet heats water* changes, only
  *how long water takes to travel between magnets and the exchanger*.
- **Merge nodes**: outgoing temperature = flow-weighted mix of whatever each
  incoming segment is delivering at that instant (tier 1's mixing formula).
- **Split nodes**: the arriving temperature propagates unchanged into every
  downstream segment; each then applies its own transit delay independently.
- **Exchanger node**: boundary condition via the tier-1 calorimetric balance,
  `T_out = T_mix_arriving − Q_hx(t)/(ρ·Cp·Flow_primary_total(t))`.

A concrete instance of this graph for the 2-magnet schematic: `header → (magnet_1
segment, magnet_2 segment, derivation segment)` in parallel, `merge → exchanger
segment`, `exchanger → return_header`, `return_header → (magnet_1 inlet, magnet_2
inlet)`.

## The node method: transit delay from time-varying flow

Pure advection with no diffusion means a segment's outlet at time `t` is exactly
its inlet at some earlier entrance time `t0(t)` — the classic **node method** used
for exactly this problem in district-heating network simulation (plug-flow pipes
with time-varying flow). No spatial discretization, no numerical diffusion:

```
W(t)  = ∫₀ᵗ Flow(τ) dτ                      # cumulative flow-volume ("flow clock")
t0(t) = W⁻¹( W(t) − V )                     # entrance time for the parcel exiting now
T_out_segment(t) = T_in_segment(t0(t))      # pure delayed lookup, no smoothing
```

`W` is monotonically increasing (flow > 0), so it's invertible by construction —
compute it by cumulative trapezoidal integration on a time grid, then build the
inverse via a monotonic interpolant of `(W values) → (t values)`.

### Why this is a post-processing pass, not a new solve_ivp state

Evaluating the network at time `t` needs the inlet's value at an *earlier* time
`t0(t) < t` — this is a delay-differential relationship, and `solve_ivp`'s explicit
steppers only ever hand the RHS the current state `y(t)`, not a history. Rather
than building a custom DDE integrator, treat the network as **deterministic
post-processing over an already-known current history**:

1. Get `I_i(t)` as a dense, evaluable function of time — either `sol.sol(t)` dense
   output from a prior electrical (or tier-1) simulation, or a measured `current`
   CSV directly.
2. Derive every branch's `Flow_i(t)` from it (flow-vs-current curves, tier 1).
3. For each `TransportSegment`, compute `W(t)` and its inverse on a fixed output
   grid, then evaluate the delayed lookup.
4. Propagate through merges/splits/the exchanger boundary in topological order to
   get every node's temperature time series, including each magnet's `Tin_i(t)`.

No new ODE states, no DDE solver — just `scipy.integrate.cumulative_trapezoid` +
monotonic `interp1d` + a topological walk over the graph, all on top of a current
history that's already fully known before this pass starts.

### Two operating modes

- **Diagnostic (primary — matches "check the T* sensors")**: feed the network real
  measured `current`/flow histories directly; compare its predicted `Tout`,
  `Talim`, `Tin_i` against the real sensor CSVs via the same
  `experimental_data`/`exp_metrics` machinery as tier 1. No iteration, no coupling
  back into the electrical model — purely deterministic.
- **Predictive (secondary, optional)**: to close the loop into `R(I,Tin)` for
  scenarios without measured data, wrap the above in an outer **Picard iteration**:
  (a) solve the electrical model with an initial `Tin` guess (e.g. tier 1's
  τ-based estimate, or a constant), (b) run the node-method pass on the resulting
  current history to get a refined `Tin_i(t)`, (c) re-solve the electrical model
  using that as an exogenous `temperature_func` (same mechanism as today's
  `temperature_csv`), (d) repeat until `Tin_i(t)` stops changing beyond tolerance
  (typically 2-3 passes, since `Tin` only weakly perturbs `R` relative to the
  current's own PID-driven dynamics). This is a staggered/partitioned coupling,
  not a monolithic one — an explicit, acceptable approximation, not a limitation
  worth hiding.

## New module: `magnet_scipy/cooling_network.py`

```python
class TransportSegment:
    def __init__(self, segment_id: str, upstream_node: str, downstream_node: str,
                 volume: float, ambient_heat_loss: float = 0.0): ...
    def transit_delay_function(self, flow_history: Callable[[float], float],
                                t_grid: np.ndarray) -> Callable[[float], float]:
        """Builds t0(t) via cumulative-flow inversion (the node method)."""

class CoolingNetwork:
    """Graph of TransportSegments + tier-1 CoolingBranches at nodes. Deterministic
    evaluator over a known current history — no new ODE states."""
    def __init__(self, nodes: List[str], segments: List[TransportSegment],
                 branches: Dict[str, CoolingBranch],   # node_id -> branch at that node
                 exchanger_node: str, water: WaterProperties = None): ...
    def evaluate(self, t_grid: np.ndarray, current_history: Dict[str, Callable[[float], float]],
                 secondary_supply, secondary_return, secondary_flow) -> Dict[str, np.ndarray]:
        """Returns every node's temperature time series, topologically ordered."""
    def solve_predictive(self, electrical_system, params, max_iterations: int = 3,
                          tol: float = 0.1) -> Tuple[SimulationResult, Dict[str, np.ndarray]]:
        """Optional Picard-iteration driver for the predictive mode."""
```

Depends on `cooling_model.py` for `CoolingBranch`, `WaterProperties`, and the
calorimetric exchanger balance — imports, doesn't duplicate.

## Data requirements beyond tier 1

The one substantive new cost of this tier: **a volume (or length × cross-section)
for every transport segment** in the network — header-to-magnet runs, merge-to-
exchanger, exchanger-to-return-header, return-header-to-magnet. Order-of-magnitude
estimates are enough to start, since what's being validated is whether measured
sensor timing looks like a genuine delay (this model) or an exponential rounding
(tier 1) — getting the delay's magnitude roughly right is what matters for that
comparison.

## Numerical caveats to carry into implementation

- Assumes flow is always positive on every segment (pump-driven, no reverse flow),
  which is what makes `W(t)` invertible. Worth an explicit guard/assertion.
- Ambient heat loss along transport is off by default; if insulation losses turn
  out to matter, it's a cheap multiplicative decay factor along the delay lookup
  (standard in district-heating pipe models) — not designed in detail here.
- This tier is only worth the extra data/complexity if tier 1's fitted `τ` doesn't
  actually reproduce the real sensors' timing shape. That's an empirical call to
  make once tier 1 is running against real `T*` logs — the plan is sequenced this
  way on purpose (build tier 1, look at the residuals, then decide whether tier 2
  is warranted).

## Tests: `tests/test_cooling_network.py`

- `TestTransitDelay`: constant-flow case has an exact closed form (`delay =
  V/Flow`) — verify the cumulative-integral/inversion machinery reproduces it
  exactly. Step-changing flow case checked against a hand-derived two-piece
  calculation.
- `TestJunctionMixing`: reuse tier 1's hand-calculated mixing cases at merge nodes.
- **Headline validation test**: a minimal 2-node network (one `TransportSegment` +
  one magnet branch) driven by a step current change with constant flow — assert
  the computed `Tin` is an *exact delayed step* (delay = `V/Flow`), not a smoothed
  exponential. This is the specific behavior tier 1 cannot produce, so it's the
  test that actually justifies this tier's existence.
- `TestConsistencyWithTier1`: for a slowly-varying (quasi-steady) current profile
  where transit delay is negligible relative to the timescale of change, the
  node-method network's steady-state values should converge to tier 1's
  equilibrium values — a cross-check that the two tiers agree where they should.

## Open items for next discussion

- Exact segment volumes/lengths for a first real network (not yet available).
- Whether the derivation branch needs its own transport segment (a delay between
  the header and the converter cooling zone) or can stay zero-delay like the
  magnet branches — unclear without knowing that branch's physical pipe run.
- Whether ambient heat loss along the transport segments is worth modeling for a
  first pass, or genuinely negligible for this installation.
- Whether the predictive/Picard mode is needed at all in the near term, given the
  primary stated purpose is sensor-checking (diagnostic mode) — could be deferred
  indefinitely if diagnostic-only use covers what's needed.
