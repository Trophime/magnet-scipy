import json

import numpy as np


def create_sample_config(n_circuits: int = 3, filename: str = "sample_config.json"):
    """Create a sample configuration file"""
    config = {"circuits": [], "mutual_inductances": None}

    # Create circuit configurations
    for i in range(n_circuits):
        circuit_config = {
            "circuit_id": f"circuit_{i+1}",
            "resistance": 1.0 + 0.2 * i,
            "inductance": 0.08 + 0.02 * i,
            "temperature": 25.0 + 5.0 * i,
            "reference_csv": f"sample_reference_circuit_{i+1}.csv",
            "resistance_csv": f"sample_resistance_circuit_{i+1}.csv",
            "pid_params": {
                "Kp_low": 15.0 + 2.0 * i,
                "Ki_low": 8.0 + 1.0 * i,
                "Kd_low": 0.08 + 0.01 * i,
                "Kp_medium": 18.0 + 2.0 * i,
                "Ki_medium": 10.0 + 1.0 * i,
                "Kd_medium": 0.06 + 0.01 * i,
                "Kp_high": 22.0 + 2.0 * i,
                "Ki_high": 12.0 + 1.0 * i,
                "Kd_high": 0.04 + 0.01 * i,
                "low_threshold": 60.0,
                "high_threshold": 200.0 + 100.0 * i,
            },
        }
        config["circuits"].append(circuit_config)

    # Create upper-triangle mutual inductances M[i, j] (i < j)
    coupling_strength = 0.02
    mutual_inductances = np.full(n_circuits * (n_circuits - 1) // 2, coupling_strength)
    config["mutual_inductances"] = mutual_inductances.tolist()

    # Save configuration
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created sample configuration: {filename}")

