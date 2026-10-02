# Coupling RLCircuitPID with PSIM

Status: draft, not started. Discussion from 2026-09-11.

## Goal

Couple the Python ODE circuit model (`RLCircuitPID` / `CoupledRLCircuitsPID`) with
PSIM for power-electronics-level co-simulation.

## Why not a direct Python link

PSIM has no native Python interface. Two PSIM-supported co-simulation paths exist:

- **SimCoupler Module**: couples PSIM with MATLAB/Simulink. Not useful here unless
  MATLAB is already in the loop.
- **DLL Block ("General DLL")**: PSIM calls a compiled C function once per
  simulation step. This is the practical route since our ODE right-hand side
  (`solve_ivp`-based, see `simulation_strategies.py`) is a plain Python/NumPy
  function that can be ported to C.

Loose/offline alternative if tight coupling isn't worth the effort: run PSIM and
the Python model standalone, exchange boundary data via CSV (`csv_utils.py`
already handles CSV I/O).

## Chosen architecture (if we proceed)

Don't reimplement `solve_ivp`/RK45 integration in C. PSIM already has native
Integrator, Gain, and Summing-junction blocks — let those own the state
integration (`i`, `integral_error`). The DLL block only supplies the nonlinear
pieces PSIM can't express natively:

- `R(I, T)` bilinear lookup (currently `csv_utils.create_2d_function_from_csv`,
  used via `RLCircuitPID.get_resistance` / `load_resistance_from_csv`,
  `rlcircuitpid.py:241-270`)
- Adaptive PID gain schedule (`PIDController.get_pid_parameters`, region
  thresholds in `rlcircuitpid.py:220-239`)
- For the coupled (N-circuit) case: the `M_ = diag(Kd) + M` linear solve done
  per step in `CoupledRLCircuitsPID.vector_field` (`coupled_circuits.py:186-227`)

Wiring in PSIM:
- Two Integrator blocks hold `i` and `integral_error`, fed back as DLL inputs.
- `i_ref` and `di_ref_dt` come from PSIM's own reference source (lookup-table
  block reading the same CSV, plus a derivative/exact-slope source), not from
  the Python `reference_func` — PSIM drives the reference in a real coupled run.
- DLL outputs `di/dt` and `d(integral_error)/dt`, feeding directly back into
  the two integrators.

Reference: `RLCircuitPID.vector_field` (`rlcircuitpid.py:380-407`) is the
single-circuit equation being ported; `CoupledRLCircuitsPID.vector_field`
(`coupled_circuits.py:186-227`) is the N-circuit version.

### Sketch (single circuit, illustrative — verify exact exported symbol name and
calling convention against the installed PSIM version's "DLL Block" docs; this
has changed across PSIM releases)

```c
#include <math.h>

#define NI 32
#define NT 8
static double I_grid[NI], T_grid[NT], R_grid[NI][NT]; // generated offline from resistance_csv

static double interp_R(double i, double T) { /* bilinear, mirrors csv_utils.create_2d_function_from_csv */ }

static void pid_gains(double i_ref, double *Kp, double *Ki, double *Kd) {
    double a = fabs(i_ref);
    if (a < 60.0)      { *Kp=10.0; *Ki=5.0;  *Kd=0.1;  }
    else if (a < 800.0){ *Kp=15.0; *Ki=8.0;  *Kd=0.05; }
    else               { *Kp=25.0; *Ki=12.0; *Kd=0.02; }
}

// in[]: i, integral_error, i_ref, di_ref_dt, T   out[]: di_dt, d(integral_error)/dt
__declspec(dllexport) void simuser(double t, double delt, double *in, double *out)
{
    double i = in[0], integral_error = in[1], i_ref = in[2], di_ref_dt = in[3], T = in[4];
    static const double L = 0.1; // RLCircuitPID.L

    double R = interp_R(i, T);
    double Kp, Ki, Kd; pid_gains(i_ref, &Kp, &Ki, &Kd);

    double numerator = -(R + Kp) * i + Kp * i_ref + Ki * integral_error + Kd * di_ref_dt;
    out[0] = numerator / (L + Kd);
    out[1] = i_ref - i;
}
```

For N circuits: extend `in[]`/`out[]` to N currents + N integral errors, replace
the scalar division with a small NxN Gauss-elimination solve for `M_` (N is
typically 2-4, so a naive solve is fine).

## Remaining work / open questions

1. Codegen script to dump `I_grid` / `T_grid` / `R_grid` from `resistance_csv`
   (via `csv_utils.py`) into a C header, so the DLL table matches whatever CSV
   is currently used in the Python model.
2. Confirm the actual PSIM DLL Block calling convention/export name for the
   PSIM version in use (varies by release — check the PSIM User Manual "DLL
   Block" chapter).
3. Decide how `i_ref` / `di_ref_dt` are sourced on the PSIM side (lookup table
   block vs. some other reference generation) so they match the CSV-driven
   reference currently used in Python (`reference_csv`, `voltage_csv`).
4. Work out the coupled (N-circuit) DLL block: NxN solve for `M_ = diag(Kd) + M`
   in C, plus how `mutual_inductances` (`CoupledRLCircuitsPID.__init__`) gets
   exported alongside the resistance table.
5. Decide whether this is worth the integration cost vs. the loose/offline CSV
   exchange alternative, given no PSIM model exists yet on this project.
