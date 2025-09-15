# Command Line Usage

## Single Circuit Simulation

### Basic Usage
```bash
# Basic simulation with configuration file
magnet-scipy --config-file circuit_config.json --show-plots

# Create sample configuration first
magnet-scipy --create-sample
```

### Configuration File Required
The single circuit simulation requires a JSON configuration file. Use `--create-sample` to generate a template:

```bash
magnet-scipy --create-sample
```

### Available Arguments
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

### Examples
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

## Coupled Circuits Simulation

### Basic Usage
```bash
# Simulation with configuration file (required)
magnet-scipy-coupled --config-file coupled_config.json --show-plots

# Create sample configuration
magnet-scipy-coupled --create-sample
```

### Available Arguments
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

### Examples
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

## Simulation Strategies

Both commands support automatic strategy detection based on your configuration:

- **voltage**: Direct voltage input simulation (requires `voltage_csv` in config)
- **pid**: PID control simulation (requires `reference_csv` and PID parameters)
- **auto**: Automatically detects based on available data (default)

Use `--list-strategies` to see detailed information about each strategy.

## Configuration Files

### Single Circuit Configuration
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

### Coupled Circuits Configuration
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
  "mutual_inductances": [
    [0.08, 0.02],
    [0.02, 0.09]
  ]
}
```

## Working Directory

Both commands support setting a working directory for relative paths:

```bash
magnet-scipy --wd /path/to/project --config-file configs/motor.json --show-plots
```

All relative paths in the configuration file will be resolved relative to the working directory.

## Output Files

### Plot Files
- PNG format with configurable DPI (300 for saved plots)
- Multiple plot profiles for different use cases
- Custom styling via JSON configuration files

### Results Files  
- NPZ format containing all simulation data
- Includes time arrays, current values, voltages, and metadata
- Can be loaded with `numpy.load()` for further analysis

### Analytics
- Detailed performance metrics when using `--show-analytics`
- PID controller performance analysis
- Circuit parameter summaries
- Error analysis and statistics