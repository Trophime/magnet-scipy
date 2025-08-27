import numpy as np
import pandas as pd

def create_sample_coupled_data(n_circuits: int = 3):
    """Create sample CSV files for testing coupled circuits"""

    time = np.linspace(0, 5, 1000)

    # Create different reference currents for each circuit
    references = []

    # Circuit 1: Step changes
    current1 = np.zeros_like(time)
    current1[time >= 0.5] = 15.0
    current1[time >= 1.5] = 150.0
    current1[time >= 2.5] = 75.0
    current1[time >= 3.5] = 400.0
    current1[time >= 4.5] = 100.0
    references.append(current1)

    # Circuit 2: Sinusoidal with offset
    current2 = 50.0 + 40.0 * np.sin(2 * np.pi * time * 0.8) * (time > 0.5)
    current2 = np.maximum(current2, 0.0)
    references.append(current2)

    # Circuit 3: Ramp with steps
    current3 = np.zeros_like(time)
    current3[time >= 1.0] = 20.0 * (time[time >= 1.0] - 1.0)
    current3[time >= 3.0] = 200.0
    current3 = np.minimum(current3, 300.0)
    references.append(current3)

    # Add more circuits if needed
    for i in range(3, n_circuits):
        # Create varied patterns
        if i % 2 == 0:
            current = 30.0 + 25.0 * np.sin(2 * np.pi * time * (0.5 + 0.2 * i))
        else:
            current = np.zeros_like(time)
            current[time >= 0.5 + 0.3 * i] = 50.0 + 20.0 * i
            current[time >= 2.0 + 0.2 * i] = 100.0 + 30.0 * i

        current = np.maximum(current, 0.0)
        references.append(current)

    # Save reference files
    for i in range(min(n_circuits, len(references))):
        df = pd.DataFrame({"time": time, "current": references[i]})
        filename = f"sample_reference_circuit_{i+1}.csv"
        df.to_csv(filename, index=False)
        print(f"Created {filename}")

    # Create sample resistance data for different circuits
    current_vals = np.linspace(0, 500, 40)
    temp_vals = np.linspace(20, 60, 25)

    for circuit_idx in range(n_circuits):
        data = []
        for temp in temp_vals:
            for curr in current_vals:
                # Different resistance models for each circuit
                R0 = 1.2 + 0.3 * circuit_idx  # Different base resistance
                alpha = 0.003 + 0.001 * circuit_idx  # Different temperature coefficient
                beta = 0.00008 + 0.00002 * circuit_idx  # Different current coefficient

                resistance = R0 * (1 + alpha * (temp - 25) + beta * curr)
                data.append(
                    {"current": curr, "temperature": temp, "resistance": resistance}
                )

        df = pd.DataFrame(data)
        filename = f"sample_resistance_circuit_{circuit_idx+1}.csv"
        df.to_csv(filename, index=False)
        print(f"Created {filename}")

