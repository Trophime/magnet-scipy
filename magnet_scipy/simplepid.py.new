import numpy as np
from scipy.integrate import solve_ivp

def simulate_pid(circuit, experimental_data, args):
    
    from simple_pid import PID

    dt = args.time_step
    t0 = args.time_start

    
    # Initial conditions [current, integral_error, prev_error]
    i0 = args.value_start
    print(f"Initial current: {i0} A at t={t0} s")
    i0_ref = circuit.reference_current(t0)
    print(f"init ref: {i0_ref:.3f} A at t={t0:.3f} s")
    if experimental_data is not None:
        v0 = np.interp(t0, experimental_data["time"], experimental_data["voltage"])
        print(f"init exp: {v0:.3f} V at t={t0:.3f} s")

    closest_index = np.argmin(np.abs(circuit.time_data - t0))
    print(f"t0: {t0}, closest_index={closest_index}")
    print("\nPID controller: run pid for each time step...")
    t = args.time_start
    # create simple pid for PIDController
    Kp, Ki, Kd = circuit.get_pid_parameters(i0_ref)
    pid = PID(Kp, Ki, Kd, setpoint=i0_ref)
    print(f"simple pid created: Kp={Kp}, Ki={Ki}, Kd={Kd}", flush=True)
    region_index = circuit.get_current_region(i0)
    print(f"region_index: {region_index}", flush=True)
    t_span = t, t + dt
    print(f"t_span: {t_span}", flush=True)
    u = v0 if v0 is not None else 0
    print(f"t={t}, u={u}", flush=True)

    voltage = []
    voltage.append(u)

    while t < args.time_end:
        sol = solve_ivp(
            lambda t, current: circuit.voltage_vector_field(t, current, u),
            t_span,
            np.array([i0]),
            method="RK45",
            dense_output=True,
            rtol=1e-6,
            atol=1e-9,
            max_step=dt,
        )
        print(f"t={t}, solv_ivp with u={u}, i0={i0}, t_span={t_span}, ", end="", flush=True)

        current = sol.y.squeeze()
        
        ifinal = float(current.squeeze()[-1])
        print(f"ifinal={ifinal}", end=", ", flush=True)

        pid.setpoint = circuit.reference_current(t + dt)
        print(f"setpoint={pid.setpoint}", end=", ", flush=True)

        new_region_index = circuit.get_current_region(ifinal)
        if new_region_index != region_index:
            Kp, Ki, Kd = circuit.get_pid_parameters(ifinal)
            pid.tuning(Kp, Ki, Kd)
            region_index = new_region_index
            print(f"change region to {region_index}")

        u = pid(ifinal, dt=args.time_step)
        voltage.append(u)
        print(f"new u={u}", flush=True)

        i0 = ifinal
        i0_ref = circuit.reference_current(t)
        t += dt
        t_span = t, t + args.time_step
        if t>1:
            exit(1)
