# Theory: RL Circuit with PID Controller as Controlled Differential Equations

## Overview

This document describes the mathematical model for an electrical RL (Resistor-Inductor) circuit with a PID (Proportional-Integral-Derivative) controller for current regulation. The PID controller is part of the **physical model**: its control law is substituted into the circuit equation, which yields a system of ordinary differential equations (ODEs) in time.

The resulting ODE system is solved with `scipy.integrate.solve_ivp`. Alternative backends (e.g. JAX/diffrax) can solve the very same system; see [Numerical Solution](#numerical-solution).

## Terminology: two unrelated "PID controllers"

Two different objects called "PID controller" appear around this model. They must not be confused:

| | Physical PID controller | Step-size "PID" controller |
|---|---|---|
| What it is | The current regulator of the power supply | An internal heuristic of adaptive ODE solvers |
| Acts on | The applied voltage `u(t)` | The time step `h` of the integrator |
| Driven by | The tracking error `e = i_ref - i` | The local truncation error estimate of the solver |
| Parameters | `Kp`, `Ki`, `Kd` (physical gains, per operating region) | Exponents on successive error estimates (`pcoeff`, `icoeff`, `dcoeff` in diffrax), plus `rtol`, `atol` |
| Changes the solution? | **Yes** — it defines the ODE system to solve | **No** — only how accurately/efficiently the ODE is integrated |
| Where | `PIDController` in `magnet_scipy/pid_controller.py`, used in `vector_field` | Inside the solver (`solve_ivp` uses a fixed integral-type controller; diffrax exposes it as `diffrax.PIDController`) |

In this document, **"PID controller" always means the physical controller** unless explicitly stated otherwise. When the solver's step-size controller is meant, it is called the **step-size controller**.

> When using diffrax, import its controller under an explicit alias to avoid name clashes, e.g.
> `from diffrax import PIDController as StepSizeController`.

## Circuit Description

The system consists of:
- An RL circuit with resistance R (Ω) and inductance L (H)
- A PID controller that regulates the current i(t) to follow a reference current i_ref(t)
- The control signal is the applied voltage u(t)

## Mathematical Model

### 1. Basic RL Circuit Dynamics

The fundamental equation governing an RL circuit is Kirchhoff's voltage law:

```
L · di/dt + R · i = u(t)
```

Where:
- `i(t)` = current through the circuit (A)
- `u(t)` = applied voltage (control input) (V)
- `L` = inductance (H)
- `R` = resistance (Ω), possibly `R(i, T)` when variable resistance is enabled
- `t` = time (s)

### 2. PID Controller (physical)

The PID controller generates the control voltage based on the current error:

```
e(t) = i_ref(t) - i(t)
```

The PID control law is:

```
u(t) = Kp · e(t) + Ki · ∫₀ᵗ e(τ) dτ + Kd · de/dt
```

Where:
- `Kp` = proportional gain (V/A)
- `Ki` = integral gain (V/(A·s))
- `Kd` = derivative gain (V·s/A, i.e. homogeneous to an inductance)
- `e(t)` = current error
- `i_ref(t)` = reference current (from CSV file or default function)

**Gain scheduling.** The gains are not constant: they are selected per operating region according to `|i_ref(t)|` (first region whose threshold exceeds `|i_ref|`). Since the selection depends on the prescribed reference and not on the computed current, the switching times are known before solving. The gains are piecewise constant in time, so the right-hand side of the ODE is discontinuous at those switching times.

### 3. Closed-Loop ODE System

Substituting the control law into the circuit equation, with `de/dt = di_ref/dt - di/dt`:

```
L · di/dt + R · i = Kp · (i_ref - i) + Ki · ∫e + Kd · (di_ref/dt - di/dt)
```

The `Kd · di/dt` term is moved to the left-hand side, so the derivative action does **not** require an extra state variable:

```
(L + Kd) · di/dt = -(R + Kp) · i + Kp · i_ref + Ki · ∫e + Kd · di_ref/dt
```

The only additional state is the integral of the error.

**State Vector:** `y = [i, ∫e]ᵀ`

Where:
- `y₁ = i` = current
- `y₂ = ∫₀ᵗ e(τ) dτ` = integral of error (A·s)

**System of Differential Equations:**

```
dy₁/dt = [ -(R + Kp) · y₁ + Kp · i_ref + Ki · y₂ + Kd · di_ref/dt ] / (L + Kd)
dy₂/dt = i_ref(t) - y₁
```

`di_ref/dt` is the exact derivative of the reference: for a piecewise-linear (CSV-interpolated) reference, it is the piecewise-constant slope of the current segment.

### 4. Coupled Circuits

For `N` magnetically coupled circuits, `L` is replaced by the (symmetric positive definite) mutual inductance matrix `M`, and the gains become per-circuit vectors. The state vector is `y = [i₁ … i_N, ∫e₁ … ∫e_N]ᵀ` and:

```
(M + diag(Kd)) · di/dt = -(R + Kp) ⊙ i + Kp ⊙ i_ref + Ki ⊙ ∫e + Kd ⊙ di_ref/dt
d(∫e)/dt = i_ref - i
```

where `⊙` is the element-wise product. Because `M` is SPD and `Kd ≥ 0`, `M + diag(Kd)` is SPD and the linear system is always solvable.

### 5. Possible Extension: Filtered Derivative (not implemented)

A real controller usually filters the derivative action, `Kd · s / (1 + τ·s)`, for noise rejection. This adds one state per circuit, `d`, the filtered derivative of the error:

```
τ · dd/dt + d = de/dt
u(t) = Kp · e + Ki · ∫e + Kd · d
```

With this form, `Kd · di/dt` no longer appears directly in the circuit equation, and the state vector becomes `y = [i, ∫e, d]ᵀ`. This is **not** the model currently implemented (see section 3).

## Reference Current Function

The reference current `i_ref(t)` can be:

### Default Step Function:
```
i_ref(t) = {
    0.0,  if t < 0.5
    2.0,  if 0.5 ≤ t < 1.5
    1.0,  if 1.5 ≤ t < 2.5
    3.0,  if t ≥ 2.5
}
```

### CSV-Based Function:
Linear interpolation from tabulated data:
```
i_ref(t) = interp(t, t_data, i_data)
```

Where `t_data` and `i_data` are arrays loaded from a CSV file with columns `'time'` and `'current'`. Its derivative `di_ref/dt` is piecewise constant and jumps at every data point.

## Implementation

### Vector Field Function

The closed-loop system of section 3 is implemented as a vector field (see `RLCircuitPID.vector_field` and `CoupledRLCircuitsPID.vector_field`):

```python
def vector_field(t, y, di_ref_dt):
    i, integral_error = y

    # Model inputs
    T = get_temperature(t)
    R = get_resistance(i, T)
    i_ref = reference_current(t)

    # Physical PID gains for the current operating region
    Kp, Ki, Kd = get_pid_parameters(i_ref)

    # Closed-loop dynamics (derivative action moved to the left-hand side)
    di_dt = (-(R + Kp) * i + Kp * i_ref + Ki * integral_error + Kd * di_ref_dt) / (L + Kd)
    dintegral_dt = i_ref - i

    return [di_dt, dintegral_dt]
```

The applied voltage is not a state; it is reconstructed after the solve from the control law of section 2.

## Numerical Solution

The ODE system of section 3 is solved with `scipy.integrate.solve_ivp`:

- **Solver**: adaptive method selected by `--method` (default `RK45`, Dormand–Prince); `Radau`, `BDF` or `LSODA` for stiff cases
- **Error Control**: the solver's internal **step-size controller**, driven by `rtol` (default 1e-6) and `atol` (default 1e-9). It is unrelated to the physical PID controller.
- **Maximum step**: `max_step = dt`, which also keeps the integrator from stepping over the discontinuities caused by gain scheduling and by the piecewise-linear reference
- **Initial Conditions**: `y₀ = [i₀, 0]` (initial current, zero integral error)

**Alternative backend (diffrax).** The same vector field can be integrated with diffrax (e.g. `Tsit5`/`Dopri5`, or `Kvaerno5` if stiff). There, the step-size controller is the class `diffrax.PIDController(rtol=..., atol=...)`. Despite its name, it only adapts the time step and has nothing to do with `Kp`, `Ki`, `Kd`. The physical gains are passed to the vector field through `args`. The known discontinuity times (gain-region switches, reference breakpoints) can be given to the step-size controller with `jump_ts`, instead of limiting the step size globally.

## Physical Interpretation

### Circuit Behavior:
- The inductance L causes current to change gradually (prevents instantaneous changes)
- The resistance R provides steady-state voltage drop proportional to current
- The PID controller adjusts voltage to minimize current tracking error

### PID Controller Action:
- **Proportional (P)**: Provides immediate response proportional to current error
- **Integral (I)**: Eliminates steady-state error by accumulating past errors
- **Derivative (D)**: Provides predictive action based on error rate of change; in the closed-loop equation it acts as an additional inductance `Kd`

### Typical Parameter Ranges:
- **R**: 1-10 Ω (circuit resistance)
- **L**: 0.01-1 H (circuit inductance)
- **Kp**: 10-50 (proportional gain)
- **Ki**: 5-20 (integral gain)
- **Kd**: 0.01-1 (derivative gain)

## Applications

This model is useful for:

1. **Control System Design**: Tuning PID parameters for desired response
2. **Circuit Analysis**: Understanding RL circuit transient behavior
3. **Educational Purposes**: Demonstrating controlled differential equations
4. **Simulation Studies**: Testing different reference current profiles
5. **Parameter Optimization**: Fitting the physical PID gains or circuit parameters to measurements (e.g. with gradients from a differentiable backend)

## Extensions

The model can be extended to include:

- **Nonlinear Elements**: Saturation (power supply voltage limit, with anti-windup), hysteresis, or nonlinear inductance
- **Discrete-time Controller**: Sampled PID with zero-order hold, as in a digital power supply regulator
- **Filtered Derivative**: See section 5
- **Noise Models**: Measurement noise or process disturbances
- **Multiple Loops**: Cascade or multi-variable control

## References

1. Ogata, K. "Modern Control Engineering" - PID Controller Theory
2. Franklin, G. F. "Feedback Control of Dynamic Systems" - State-Space Methods
3. Hairer, E., Wanner, G. "Solving Ordinary Differential Equations I/II" - Step-size control
4. SciPy Documentation - `scipy.integrate.solve_ivp`
5. Diffrax Documentation - Differential Equation Solvers in JAX
