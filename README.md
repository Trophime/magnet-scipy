# Magnet Scipy

**Magnetic coupling simulation for RL circuits with adaptive PID control using Scipy**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)

## Overview

Magnet Scipy is a simulation package for modeling RL (resistor-inductor) circuits with:

- **Adaptive PID Control**: Dynamic PID parameter adjustment based on current magnitude
- **Variable Resistance**: Temperature and current-dependent resistance modeling
- **Magnetic Coupling**: Multi-circuit systems with mutual inductance effects  
- **CSV Integration**: Load experimental data for validation and comparison
- **Comprehensive Visualization**: Rich plotting and analysis tools

## Features

### Single Circuit Simulation
- Adaptive PID control with configurable current regions
- Variable resistance R(I,T) from CSV data or analytical models
- Reference current tracking from CSV or analytical functions
- Comprehensive performance analysis and visualization

### Coupled Circuit Systems  
- Multiple magnetically coupled RL circuits
- Independent PID controllers for each circuit
- Configurable mutual inductance matrices
- Cross-coupling analysis and visualization

### Advanced Capabilities
- Flexible CSV data integration
- Extensive plotting and analysis tools
- Command-line interfaces for batch processing

## Installation

### From Source
```bash
git clone https://github.com/Trophime/magnet_scipy.git
cd magnet_scipy
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/Trophime/magnet_scipy.git
cd magnet_scipy
pip install -e ".[dev]"
```

### Dependencies
- **Core**: ScyPy (≥1.14.0), NumPy (≥1.21.0), Pandas (≥1.3.0)
- **Plotting**: Matplotlib (≥3.5.0), Seaborn (≥0.11.0)
- **Development**: pytest, black, flake8, mypy

## Quick Start

### Single Circuit Example

### Coupled Circuits Example  

### Loading Data from CSV
```python
# Circuit with CSV data
circuit = RLCircuitPID(
    R=1.0, L=0.1,
    reference_csv="reference_current.csv",     # time, current columns
    resistance_csv="resistance_data.csv",     # current, temperature, resistance  
    temperature=30.0,
    circuit_id="experimental_motor"
)

# The CSV files should have the following format:
# reference_current.csv: time,current
# resistance_data.csv: current,temperature,resistance
```

## Command Line Usage

### Single Circuit Simulation
# Basic simulation
```bash
magnet-scipy --inductance 0.1 --resistance 1.5 --show-plots
```

# With CSV data  
```bash
magnet-scipy --reference-csv data.csv --resistance-csv resistance.csv --show-analytics
```

# Custom PID parameters
```bash
magnet-scipy --custom-pid --kp-low 20 --ki-low 15 --show-plots
```

### Coupled Circuits Simulation  

# Default coupled system

# From configuration file
```bash
magnet-scipy-coupled --config-file config.json --show-coupling --save-results results.npz
```

# Create sample configuration
```bash
magnet-scipy-coupled --create-config sample_config.json
```

## Configuration Files

### JSON Configuration for Coupled Systems
```json
{
  "circuits": [
    {
      "circuit_id": "motor_1",
      "R": 1.0,
      "L": 0.08, 
      "temperature": 25.0,
      "reference_csv": "motor1_reference.csv",
      "resistance_csv": "motor1_resistance.csv",
      "pid_params": {
        "Kp_low": 15.0, "Ki_low": 8.0, "Kd_low": 0.08,
        "Kp_high": 25.0, "Ki_high": 15.0, "Kd_high": 0.04,
        "low_threshold": 60.0, "high_threshold": 200.0
      }
    }
  ],
  "mutual_inductances": [[0.0, 0.02], [0.02, 0.0]]
}
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# With coverage
pytest --cov=magnet_scipy

# Run specific test categories  
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

Test structure:
```
tests/
├── test_pid_controller.py     # PID controller tests
├── test_rlcircuitpid.py      # Single circuit tests  
├── test_coupled_circuits.py  # Coupled system tests
├── test_jax_csv_utils.py     # CSV utilities tests
├── conftest.py               # Test configuration
└── data/                     # Test data files
```

## API Reference

### Core Classes

#### `RLCircuitPID`
Main class for single RL circuit with adaptive PID control.

**Parameters:**
- `R`: Base resistance (Ω)  
- `L`: Inductance (H)
- `pid_controller`: PIDController instance
- `reference_csv`: Path to reference current CSV
- `resistance_csv`: Path to resistance data CSV  
- `temperature`: Operating temperature (°C)
- `circuit_id`: Unique circuit identifier

#### `CoupledRLCircuitsPID`  
Container for multiple magnetically coupled RL circuits.

**Parameters:**
- `circuits`: List of RLCircuitPID instances
- `mutual_inductances`: NxN coupling matrix
- `coupling_strength`: Default coupling value

#### `PIDController`
Flexible adaptive PID controller with region-based parameters.

**Key Methods:**
- `get_pid_parameters(i_ref)`: Get PID gains for reference current
- `get_current_region_name(i_ref)`: Get operating region name
- `add_region(name, config)`: Add new current region

## Performance


## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Magnet Scipy in your research, please cite:

```bibtex
@software{magnet_scipy,
  author = {MagnetDB Team},
  title = {Magnet Scipy: Magnetic coupling simulation for RL circuits with adaptive PID control},
  url = {https://github.com/Trophime/magnet_scipy},
  version = {0.1.0},
  year = {2024}
}
```

## todos

* [ ] test for extrapolation of csv 1D tabulated data
* [ ] implement extrapolation for 2D tabulated data
* [ ] test couple_main for one circuit with regular modeling
* [ ] test couple_main for one circuit with pid modeling
* [ ] test couple_main for two circuits with regular modeling - case M = 0 
* [ ] test couple_main for one circuits with pid modeling - case M = 0
* [ ] test couple_main for two circuits with regular modeling - case M!=0 
* [ ] test couple_main for one circuits with pid modeling - case M != 0
* [ ] set args.time_start, args.time_end from input csv and/or exp_data when available
* [ ] args.time_start: cannot be less that t0 from input csv
* [ ] args.time_end: cannot be greater that t1 from input csv
* [ ] read from csv using name of columns for input/output
* [ ] add notes on creating input data from magnetrun, magnetapi
* [ ] add intermediate voltage taps
* [ ] rework experimental data (add a dict per circuit_id for voltage, intermediate voltages, current)
* [ ] enable the choice of the solver method in solver_ivp
* [ ] use a different solver backend (eg diffrax/Jax)
* [ ] performance benchmarks
* [ ] add estimate Tout from pupitre data
* [ ] couple with cooling model for estimate of Tin
* [ ] add symbol, unit for field (try to use a single package to deal with that in all magnet related package)
* [ ] when loading data from csv use text into bracket to get unit, always convert to SI
* [ ] can display when a given unit (eg MW for Power instead of W)

## Acknowledgments


## Support

- **Documentation**: [https://magnet-scipy.readthedocs.io](https://magnet-scipy.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/Trophime/magnet_scipy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Trophime/magnet_scipy/discussions)
- **Email**: team@magnetscipy.com