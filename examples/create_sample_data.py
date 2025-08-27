import numpy as np
import pandas as pd


def sample_reference_csv(filename: str = "reference_current.csv"):
    """Create a sample CSV file with reference current data spanning all three regions"""

    # Create sample time series data
    time = np.linspace(0, 8, 2000)  # Extended time for more regions

    # Create reference signal that covers all three current regions
    current = np.zeros_like(time)

    # Low current region (|I| < 60)
    current[time >= 0.5] = 30.0  # Low current
    current[time >= 1.5] = 45.0  # Low current (negative)

    # Medium current region (60 <= |I| < 800)
    current[time >= 2.5] = 150.0  # Medium current
    current[time >= 3.5] = 200.0  # Medium current (negative)
    current[time >= 4.5] = 400.0  # Medium current

    # High current region (|I| >= 800)
    current[time >= 5.5] = 1000.0  # High current
    current[time >= 6.5] = 1200.0  # High current (negative)
    current[time >= 7.5] = 900.0  # High current

    # Add some sinusoidal variation
    current += 20 * np.sin(2 * np.pi * time) * (time > 1.0)

    # Create DataFrame and save
    df = pd.DataFrame({"time": time, "current": current})

    df.to_csv(filename, index=False)
    print(f"Sample reference current saved to {filename}")
    print("Current ranges: Low (<60), Medium (60-800), High (>800)")
    return filename


def create_sample_data():
    """Create sample CSV files for testing if they don't exist"""

    # Create sample reference current
    time = np.linspace(0, 5, 1000)
    current = np.zeros_like(time)
    current[time >= 0.5] = 10.0  # Low current
    current[time >= 1.5] = 100.0  # Medium current
    current[time >= 2.5] = 50.0  # Back to medium
    current[time >= 3.5] = 500.0  # High current
    current[time >= 4.5] = 200.0  # Medium current

    # Add some variation
    current += 10 * np.sin(2 * np.pi * time) * (time > 1.0)
    current = np.maximum(current, 0.0)  # Ensure non-negative

    df_ref = pd.DataFrame({"time": time, "current": current})
    df_ref.to_csv("sample_reference.csv", index=False)
    print("Created sample_reference.csv")

    # Create sample resistance data
    current_vals = np.linspace(0, 1000, 50)
    temp_vals = np.linspace(20, 80, 30)

    data = []
    for temp in temp_vals:
        for curr in current_vals:
            # Simple resistance model: R = R0 * (1 + alpha*T + beta*I)
            R0 = 1.5
            alpha = 0.004  # Temperature coefficient
            beta = 0.0001  # Current coefficient
            resistance = R0 * (1 + alpha * (temp - 25) + beta * curr)
            data.append(
                {"current": curr, "temperature": temp, "resistance": resistance}
            )

    df_res = pd.DataFrame(data)
    df_res.to_csv("sample_resistance.csv", index=False)
    print("Created sample_resistance.csv")

    # Create sample experimental voltage data
    exp_time = np.linspace(0, 5, 500)  # Less dense than simulation
    # Simulate some "experimental" voltage with noise and slight differences
    exp_voltage = np.zeros_like(exp_time)
    exp_voltage[exp_time >= 0.5] = 12.0 + 2.0 * np.random.normal(
        0, 0.5, np.sum(exp_time >= 0.5)
    )
    exp_voltage[exp_time >= 1.5] = (
        18.0
        + 3.0 * np.sin(4 * np.pi * exp_time[exp_time >= 1.5])
        + np.random.normal(0, 1.0, np.sum(exp_time >= 1.5))
    )
    exp_voltage[exp_time >= 2.5] = 15.0 + np.random.normal(
        0, 0.8, np.sum(exp_time >= 2.5)
    )
    exp_voltage[exp_time >= 3.5] = (
        25.0
        + 5.0 * np.sin(2 * np.pi * exp_time[exp_time >= 3.5])
        + np.random.normal(0, 1.5, np.sum(exp_time >= 3.5))
    )
    exp_voltage[exp_time >= 4.5] = 20.0 + np.random.normal(
        0, 1.0, np.sum(exp_time >= 4.5)
    )

    df_exp = pd.DataFrame({"time": exp_time, "voltage": exp_voltage})
    df_exp.to_csv("sample_experimental.csv", index=False)
    print("Created sample_experimental.csv")
