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

#### Basic Usage
```bash
# Basic simulation with configuration file
magnet-scipy --config-file circuit_config.json --show-plots

# Create sample configuration first
magnet-scipy --create-sample
```

#### Configuration File Required
The single circuit simulation requires a JSON configuration file. Use `--create-sample` to generate a template:

```bash
magnet-scipy --create-sample
```

#### Available Arguments
```bash
# Core simulation options
magnet-scipy --config-file CONFIG_FILE [OPTIONS]

# Time control
--time-start FLOAT          # Simulation start time (default: 0.0)
--time-end FLOAT            # Simulation end time (default: 5.0) 
--time-step FLOAT           # Time step size (default: 0.001)
--method {RK45,RK23,DOP853,Radau,BDF,LSODA}  # ODE solver method

# Initial conditions
--value-start FLOAT         # Initial current value in Amperes (default: 0.0)

# Simulation strategy
--strategy {voltage,pid,auto}  # Simulation mode (default: auto)
--list-strategies           # Show available strategies and exit

# Output options
--show-plots               # Display interactive plots
--save-plots PATH          # Save plots to PNG file
--show-analytics           # Show detailed performance analytics
--save-results PATH        # Save simulation data to NPZ file
--debug                    # Enable debug output

# Plotting customization
--plot-profile {default,publication,presentation,debug,minimal}
--plot-config PATH         # Custom plotting configuration JSON
--comparison-mode          # Enable comparison plotting mode
--benchmark-plotting       # Enable plotting performance profiling

# Utilities
--wd PATH                  # Set working directory
--version                  # Show version information
```

#### Examples
```bash
# Basic simulation with plots
magnet-scipy --config-file motor_config.json --show-plots

# High-quality output for publication
magnet-scipy --config-file config.json --save-plots motor_analysis.png --plot-profile publication

# Performance analysis with debug info
magnet-scipy --config-file config.json --show-analytics --debug

# Custom time parameters
magnet-scipy --config-file config.json --time-end 10.0 --time-step 0.0001 --method BDF
```

### Coupled Circuits Simulation

#### Basic Usage
```bash
# Simulation with configuration file (required)
magnet-scipy-coupled --config-file coupled_config.json --show-plots

# Create sample configuration
magnet-scipy-coupled --create-sample
```

#### Available Arguments
```bash
# Core simulation options
magnet-scipy-coupled --config-file CONFIG_FILE [OPTIONS]

# Time control (same as single circuit)
--time-start FLOAT
--time-end FLOAT
--time-step FLOAT
--method {RK45,RK23,DOP853,Radau,BDF,LSODA}

# Initial conditions
--value-start FLOAT [FLOAT ...]  # Initial current values (one per circuit)

# Simulation strategy
--strategy {voltage,pid,auto}    # Simulation mode (default: auto)
--list-strategies               # Show available strategies

# Output options (same as single circuit)
--show-plots
--save-plots PATH
--show-analytics  
--save-results PATH
--debug

# Plotting customization (same as single circuit)
--plot-profile {default,publication,presentation,debug,minimal}
--plot-config PATH
--comparison-mode
--benchmark-plotting

# Utilities
--wd PATH
--version
```

#### Examples
```bash
# Basic coupled simulation
magnet-scipy-coupled --config-file motors_config.json --show-plots

# Set different initial currents for each circuit  
magnet-scipy-coupled --config-file config.json --value-start 0.0 1.5 0.8 --show-plots

# Analysis mode with detailed output
magnet-scipy-coupled --config-file config.json --show-analytics --save-results coupling_analysis.npz

# High-quality plots for presentation
magnet-scipy-coupled --config-file config.json --save-plots coupling_plots.png --plot-profile presentation
```

### Simulation Strategies

Both commands support automatic strategy detection based on your configuration:

- **voltage**: Direct voltage input simulation (requires `voltage_csv` in config)
- **pid**: PID control simulation (requires `reference_csv` and PID parameters)
- **auto**: Automatically detects based on available data (default)

Use `--list-strategies` to see detailed information about each strategy.

### Configuration Files

#### Single Circuit Configuration
```json
{
  "circuit_id": "test_motor",
  "R": 1.0,
  "L": 0.1,
  "temperature": 25.0,
  "reference_csv": "reference_current.csv",
  "resistance_csv": "resistance_data.csv",
  "pid_params": {
    "Kp_low": 10.0, "Ki_low": 5.0, "Kd_low": 0.1,
    "Kp_high": 25.0, "Ki_high": 12.0, "Kd_high": 0.02,
    "low_threshold": 60.0, "high_threshold": 800.0
  }
}
```

#### Coupled Circuits Configuration
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
    },
    {
      "circuit_id": "motor_2", 
      "R": 1.2,
      "L": 0.09,
      "temperature": 30.0,
      "reference_csv": "motor2_reference.csv"
    }
  ],
  "mutual_inductances": [0.09]
}
```

### Working Directory

Both commands support setting a working directory for relative paths:

```bash
magnet-scipy --wd /path/to/project --config-file configs/motor.json --show-plots
```

All relative paths in the configuration file will be resolved relative to the working directory.

### Output Files

#### Plot Files
- PNG format with configurable DPI (300 for saved plots)
- Multiple plot profiles for different use cases
- Custom styling via JSON configuration files

#### Results Files  
- NPZ format containing all simulation data
- Includes time arrays, current values, voltages, and metadata
- Can be loaded with `numpy.load()` for further analysis

#### Analytics
- Detailed performance metrics when using `--show-analytics`
- PID controller performance analysis
- Circuit parameter summaries
- Error analysis and statistics

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
├── test_pid_controller.py    # PID controller tests
├── test_rlcircuitpid.py      # Single circuit tests  
├── test_coupled_circuits.py  # Coupled system tests
├── test_csv_utils.py         # CSV utilities tests
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
- `temperature_csv`: Path to Operating temperature (°C)
- `circuit_id`: Unique circuit identifier

#### `CoupledRLCircuitsPID`  
Container for multiple magnetically coupled RL circuits.

**Parameters:**
- `circuits`: List of RLCircuitPID instances
- `mutual_inductances`: Nx(N+1)/2 -N mutual inductances as an array

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