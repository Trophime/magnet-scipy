# Theory: RL Circuit with PID Controller as Controlled Differential Equations

## Overview

This document describes the mathematical model for an electrical RL (Resistor-Inductor) circuit with a PID (Proportional-Integral-Derivative) controller for current regulation. The system is formulated as a set of controlled differential equations (CDEs) suitable for numerical solution using JAX and the diffrax library.

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
- `R` = resistance (Ω)
- `t` = time (s)

Rearranging for the derivative:

```
di/dt = (-R/L) · i + (1/L) · u(t)
```

### 2. PID Controller

The PID controller generates the control voltage based on the current error:

```
e(t) = i_ref(t) - i(t)
```

The PID control law is:

```
u(t) = Kp · e(t) + Ki · ∫₀ᵗ e(τ) dτ + Kd · de/dt
```

Where:
- `Kp` = proportional gain
- `Ki` = integral gain  
- `Kd` = derivative gain
- `e(t)` = current error
- `i_ref(t)` = reference current (from CSV file or default function)

### 3. State-Space Representation

To solve this as a system of differential equations, we introduce state variables:

**State Vector:** `y = [i, ∫e, e_prev]ᵀ`

Where:
- `x₁ = i` = current
- `x₂ = ∫₀ᵗ e(τ) dτ` = integral of error
- `x₃ = e_prev` = previous error (for derivative approximation)

**System of Differential Equations:**

```
dx₁/dt = (-R/L) · x₁ + (1/L) · u(t)
dx₂/dt = e(t) = i_ref(t) - x₁  
dx₃/dt = e(t) - x₃
```

**Control Law:**
```
u(t) = Kp · (i_ref - x₁) + Ki · x₂ + Kd · (x₃ - x₃_prev)
```

### 4. Enhanced Model with Filtered Derivative

For better numerical stability and noise rejection, the derivative term can be filtered:

**Enhanced State Vector:** `y = [i, ∫e, e_filtered]ᵀ`

Where `e_filtered` is the output of a first-order filter applied to the derivative of the error.

**Filtered Derivative Equation:**
```
τ · d(e_filtered)/dt + e_filtered = de/dt
```

Rearranging:
```
d(e_filtered)/dt = (de/dt - e_filtered) / τ
```

Where `τ` is the derivative filter time constant.

**Complete Enhanced System:**
```
dx₁/dt = (-R/L) · x₁ + (1/L) · u(t)
dx₂/dt = i_ref(t) - x₁
dx₃/dt = (de/dt - x₃) / τ
```

With `de/dt = -dx₁/dt` (since `i_ref` is typically piecewise constant).

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

Where `t_data` and `i_data` are arrays loaded from a CSV file with columns `'time'` and `'current'`.

## Implementation in JAX/Diffrax

### Vector Field Function

The system is implemented as a vector field function compatible with diffrax:

```python
def vector_field(t, y, args):
    i, integral_error, derivative_term = y
    
    # Reference current (JAX-compatible)
    i_ref = reference_func(t)
    
    # Current error
    error = i_ref - i
    
    # PID control signal
    u = Kp * error + Ki * integral_error + Kd * derivative_term
    
    # System dynamics
    di_dt = (-R * i + u) / L
    dintegral_dt = error
    dderivative_dt = calculate_derivative_term(...)
    
    return jnp.array([di_dt, dintegral_dt, dderivative_dt])
```

### JAX Integration

The model leverages JAX features:

1. **JIT Compilation**: All functions are JIT-compiled for performance
2. **Automatic Differentiation**: Gradients can be computed through the system
3. **Vectorization**: Efficient evaluation over time arrays
4. **Pure Functions**: No side effects, suitable for optimization

## Numerical Solution

The differential equation system is solved using diffrax with:

- **Solver**: Dormand-Prince (Dopri5) adaptive Runge-Kutta method
- **Error Control**: PID stepsize controller with relative tolerance ~1e-6
- **Time Span**: Typically 0 to 4-5 seconds
- **Initial Conditions**: `y₀ = [0, 0, 0]` (zero current, zero integral error, zero derivative)

## Physical Interpretation

### Circuit Behavior:
- The inductance L causes current to change gradually (prevents instantaneous changes)
- The resistance R provides steady-state voltage drop proportional to current
- The PID controller adjusts voltage to minimize current tracking error

### PID Controller Action:
- **Proportional (P)**: Provides immediate response proportional to current error
- **Integral (I)**: Eliminates steady-state error by accumulating past errors
- **Derivative (D)**: Provides predictive action based on error rate of change

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
5. **Parameter Optimization**: Using JAX gradients for automatic tuning

## Extensions

The model can be extended to include:

- **Nonlinear Elements**: Saturation, hysteresis, or nonlinear inductance
- **Noise Models**: Measurement noise or process disturbances  
- **Multiple Loops**: Cascade or multi-variable control
- **Adaptive Control**: Time-varying PID parameters
- **Robust Control**: Uncertainty modeling and robust design

## References

1. Ogata, K. "Modern Control Engineering" - PID Controller Theory
2. Franklin, G. F. "Feedback Control of Dynamic Systems" - State-Space Methods
3. JAX Documentation - Automatic Differentiation and JIT Compilation
4. Diffrax Documentation - Differential Equation Solvers in JAX