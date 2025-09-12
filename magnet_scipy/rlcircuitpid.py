import sys
import numpy as np
from typing import Tuple, List, Dict

# Import the new PID controller and CSV utilities
from .pid_controller import PIDController, create_adaptive_pid_controller
from .csv_utils import create_function_from_csv, create_2d_function_from_csv


class RLCircuitPID:
    """RL Circuit with Adaptive PID Controller and variable resistance from CSV"""

    def __init__(
        self,
        R: float = 1.0,
        L: float = 0.1,
        pid_controller: PIDController = None,
        reference_csv: str = None,
        voltage_csv: str = None,
        resistance_csv: str = None,
        temperature: float = 25.0,
        temperature_csv: str = None,
        circuit_id: str = None,
        experimental_data: List[dict] = None,  # NEW: Experimental data list
        # Backward compatibility: individual PID parameters (deprecated)
        **pid_kwargs,
    ):
        """
        Initialize circuit parameters with adaptive PID controller

        Args:
            R: Constant resistance (Ohms) - used if resistance_csv is None
            L: Inductance (Henry)
            pid_controller: PIDController instance for managing adaptive PID gains
            reference_csv: Path to CSV file with reference current data
            resistance_csv: Path to CSV file with resistance data R(I, Tin)
            temperature: Temperature (°C) for resistance calculation
            temperature_csv: Path to CSV file with temperature Tin data
            circuit_id: Unique identifier for this circuit (required for coupled systems)
            experimental_data: List of ExperimentalData objects for experimental validation
            **pid_kwargs: Backward compatibility parameters for creating PID controller
        """
        self.L = L
        self.temperature = temperature

        # Set circuit ID
        self.circuit_id = circuit_id

        # Initialize experimental data
        self.experimental_data = (
            experimental_data if experimental_data is not None else []
        )

        # Load experimental data functions
        # self._load_experimental_data()

        # Initialize PID controller
        self.pid_controller = None
        if pid_controller is not None:
            self.pid_controller = pid_controller
        else:
            # Create PID controller from kwargs (backward compatibility)
            if voltage_csv is None:
                print(f"create PID from kwargs: {voltage_csv}")
                self.pid_controller = self._create_pid_from_kwargs(**pid_kwargs)

        self.voltage_csv = voltage_csv
        self.reference_csv = reference_csv

        # Handle resistance
        self.use_variable_resistance = False
        if resistance_csv:
            self.load_resistance_from_csv(resistance_csv)
        else:
            self.R_constant = R
            print(f"Using constant resistance: {R} Ω")

        # Always initialize these to None first - PREVENTS AttributeError
        self.temperature_func = None
        self.reference_func = None
        self.voltage_func = None
        self.use_csv_data = False

        # Load reference current from CSV if provided
        self.use_variable_temperature = False
        if temperature_csv:
            self.load_temperature_from_csv(temperature_csv)
        else:
            self.temperature = temperature
            print(f"Using constant temperature: {temperature} °C")

        if reference_csv:
            self.load_reference_from_csv(reference_csv)

        # Load input voltage from CSV if provided
        self.voltage_csv = None
        if voltage_csv:
            self.load_voltage_from_csv(voltage_csv)
            self.voltage_csv = voltage_csv

    def _load_experimental_data(self):
        """Load experimental data functions from files"""
        self.experimental_functions = {}

        for exp_data in self.experimental_data:
            try:
                func, time_data, values_data = create_function_from_csv(
                    exp_data["file"], "time", exp_data["key"], method="linear"
                )
                self.experimental_functions[f'{exp_data["type"]}_{exp_data["key"]}'] = {
                    "function": func,
                    "time_data": time_data,
                    "values_data": values_data,
                    "type": exp_data["type"],
                    "key": exp_data["key"],
                    "file": exp_data["file"],
                }
                print(
                    f'✓ Loaded experimental {exp_data["type"]} data from {exp_data["file"]}'
                )

            except Exception as e:
                print(
                    f'❌ Error loading experimental data from {exp_data["file"]}: {e}'
                )
                raise

    def add_experimental_data(self, exp_data: dict):
        """Add new experimental data entry"""
        # Validate new entry
        combination = (exp_data["type"], exp_data["key"])
        for existing in self.experimental_data:
            if (existing.type, existing.key) == combination:
                raise ValueError(
                    f'Experimental data entry already exists: type={exp_data["type"]}, key={exp_data["key"]}'
                )

        self.experimental_data.append(exp_data)
        self._load_single_experimental_data(exp_data)
        print(f'✓ Added experimental {exp_data["type"]} data: {exp_data["key"]}')

    def _load_single_experimental_data(self, exp_data: dict):
        """Load a single experimental data entry"""
        try:
            if exp_data["file"].endswith(".csv"):
                func, time_data, values_data = create_function_from_csv(
                    exp_data["file"], "time", exp_data["key"], method="linear"
                )
                self.experimental_functions[f'{exp_data["type"]}_{exp_data["key"]}'] = {
                    "function": func,
                    "time_data": time_data,
                    "values_data": values_data,
                    "type": exp_data["type"],
                    "key": exp_data["key"],
                    "file": exp_data["file"],
                }
            elif exp_data["file"].endswith(".tdms"):
                raise NotImplementedError(
                    f'TDMS file support not yet implemented for {exp_data["file"]}'
                )
            else:
                raise ValueError(f'Unsupported file format: {exp_data["file"]}')
        except Exception as e:
            raise RuntimeError(
                f'Error loading experimental data from {exp_data["file"]}: {e}'
            )

    def get_experimental_data(self, data_type: str, key: str, t: float) -> float:
        """Get experimental data value at time t"""
        func_key = f"{data_type}_{key}"
        if func_key not in self.experimental_functions:
            raise ValueError(
                f"No experimental data found for type='{data_type}', key='{key}'"
            )

        return self.experimental_functions[func_key]["function"](t)

    def list_experimental_data(self) -> List[Dict[str, str]]:
        """List all available experimental data entries"""
        return self.experimental_data

    def experimental_data_with_data_type(self, data_type: str):
        """returns keys associated to experimental_data with a given data_type"""
        return [exp["key"] for exp in self.experimental_data if exp["type"] == data_type]

    def has_experimental_data(self, data_type: str, key: str = None) -> bool:
        """Check if experimental data exists for given type and optionally key"""
        if key is None:
            return any(exp["type"] == data_type for exp in self.experimental_data)
        else:
            return any(
                exp["type"] == data_type and exp["key"] == key
                for exp in self.experimental_data
            )

    def remove_experimental_data(self, data_type: str, key: str):
        """Remove experimental data entry"""
        # Find and remove from list
        for i, exp_data in enumerate(self.experimental_data):
            if exp_data["type"] == data_type and exp_data["key"] == key:
                del self.experimental_data[i]
                break
        else:
            raise ValueError(
                f"No experimental data found for type='{data_type}', key='{key}'"
            )

        # Remove from functions dict
        func_key = f"{data_type}_{key}"
        if func_key in self.experimental_functions:
            del self.experimental_functions[func_key]

        print(f"✓ Removed experimental {data_type} data: {key}")

    def _create_pid_from_kwargs(self, **kwargs) -> PIDController:
        """
        Create PID controller from individual parameters for backward compatibility
        """
        # Extract PID parameters with defaults
        pid_params = {
            "Kp_low": kwargs.get("Kp_low", 10.0),
            "Ki_low": kwargs.get("Ki_low", 5.0),
            "Kd_low": kwargs.get("Kd_low", 0.1),
            "Kp_medium": kwargs.get("Kp_medium", 15.0),
            "Ki_medium": kwargs.get("Ki_medium", 8.0),
            "Kd_medium": kwargs.get("Kd_medium", 0.05),
            "Kp_high": kwargs.get("Kp_high", 25.0),
            "Ki_high": kwargs.get("Ki_high", 12.0),
            "Kd_high": kwargs.get("Kd_high", 0.02),
            "low_threshold": kwargs.get("low_current_threshold", 60.0),
            "high_threshold": kwargs.get("high_current_threshold", 800.0),
        }

        return create_adaptive_pid_controller(**pid_params)

    def load_resistance_from_csv(self, csv_file: str):
        """Load variable resistance from CSV file"""
        import pandas as pd

        try:
            pd.read_csv(csv_file, sep=None, engine="python")
            print(f"✓ Resistance CSV loaded: {csv_file}")
        except FileNotFoundError:
            print(f"❌ Resistance CSV file not found: {csv_file}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading resistance CSV: {e}")
            sys.exit(1)

        try:
            self.resistance_func, self.current_range, self.temp_range, self.R_grid = (
                create_2d_function_from_csv(
                    csv_file, "current", "temperature", "resistance", method="linear"
                )
            )
            self.use_variable_resistance = True
            print(f"Loaded variable resistance from {csv_file}")
            print(
                f"Current range: {float(self.current_range.min()):.3f} to {float(self.current_range.max()):.3f} A"
            )
            print(
                f"Temperature range: {float(self.temp_range.min()):.1f} to {float(self.temp_range.max()):.1f} °C"
            )
        except Exception as e:
            raise RuntimeError(f"Error loading resistance CSV file {csv_file}: {e}")

    def load_temperature_from_csv(self, csv_file: str):
        """Load temperature from CSV file using Scipy"""
        print(f"loading temperature from csv {csv_file}")
        try:
            self.temperature_func, self.temperature_time_data, self.temperature_data = (
                create_function_from_csv(csv_file, "time", "temperature", method="linear")
            )

            self.use_variable_temperature = True
            print(f"Loaded temperature {csv_file} using Scipy interpolation")
            print(
                f"Temperature Time range: {float(self.temperature_time_data[0]):.3f} to {float(self.temperature_time_data[-1]):.3f} seconds"
            )

        except Exception as e:
            raise RuntimeError(f"Error loading CSV file {csv_file}: {e}")

    def load_reference_from_csv(self, csv_file: str):
        """Load reference current from CSV file using Scipy"""
        print(f"loading reference from csv {csv_file}")
        try:
            self.reference_func, self.time_data, self.current_data = (
                create_function_from_csv(csv_file, "time", "current", method="linear")
            )

            self.use_csv_data = True
            print(f"Loaded reference current from {csv_file} using Scipy interpolation")
            print(
                f"Reference Time range: {float(self.time_data[0]):.3f} to {float(self.time_data[-1]):.3f} seconds"
            )

        except Exception as e:
            raise RuntimeError(f"Error loading CSV file {csv_file}: {e}")

    def load_voltage_from_csv(self, csv_file: str):
        """Load input voltage from CSV file using Scipy"""
        print(f"loading voltage from csv {csv_file}")
        try:
            self.voltage_func, self.time_data, self.voltage_data = (
                create_function_from_csv(csv_file, "time", "voltage", method="linear")
            )

            self.use_csv_data = True
            print(f"Loaded input voltage from {csv_file} using Scipy interpolation")
            print(
                f"Input Voltage Time range: {float(self.time_data[0]):.3f} to {float(self.time_data[-1]):.3f} seconds"
            )

        except Exception as e:
            raise RuntimeError(f"Error loading CSV file {csv_file}: {e}")

    def get_pid_parameters(self, i_ref: float) -> Tuple[float, float, float]:
        """
        Get PID parameters based on reference current magnitude
        Delegates to the PID controller

        Args:
            i_ref: Reference current value

        Returns:
            Tuple of (Kp, Ki, Kd) for the current operating region
        """
        return self.pid_controller.get_pid_parameters(i_ref)

    def get_current_region(self, i_ref: float) -> str:
        """
        Get the current operating region name for logging/plotting
        Delegates to the PID controller
        """
        return self.pid_controller.get_current_region_name(i_ref)

    def get_resistance(self, current: float, temperature: float = None) -> float:
        """Get resistance value based on current and temperature"""
        if self.use_variable_resistance:
            if temperature is None:
                temperature = self.temperature
            return self.resistance_func(current, temperature)
        else:
            return self.R_constant

    def get_temperature(self, t: float) -> float:
        """Get input temperature at time t"""
        if self.use_variable_temperature:
            return self.temperature_func(t)
        else:
            return self.temperature

    def reference_current(self, t: float) -> float:
        """Get reference current at time t"""
        return self.reference_func(t)

    def input_voltage(self, t: float) -> float:
        """Get input voltage at time t"""
        return self.voltage_func(t)

    def vector_field(self, t, y, di_ref_dt: float = 0):
        """
        Define the system dynamics as a vector field with variable resistance and adaptive PID

        State vector y = [i, integral_error]
        where:
        - i: current
        - integral_error: integral of error for PID
        """
        i, integral_error = y

        # Get parameters
        temperature = self.get_temperature(t)
        R_current = self.get_resistance(i, temperature)
        i_ref = self.reference_current(t)
        # di_ref_dt = self.reference_current_derivative
        Kp, Ki, Kd = self.get_pid_parameters(i_ref)

        # Analytical di/dt
        numerator = (
            -(R_current + Kp) * i + Kp * i_ref + Ki * integral_error + Kd * di_ref_dt
        )
        di_dt = numerator / (self.L + Kd)

        # Integral error evolution
        error = i_ref - i
        dintegral_dt = error

        return np.array([di_dt, dintegral_dt])

    def voltage_vector_field(self, t: float, y, u: float = None):
        """
        RL circuit ODE
        """

        i = y  # Current is a scalar now, not an array

        # Get voltage from CSV data
        if u is None:
            u = self.input_voltage(t)

        # Get current-dependent resistance
        temperature = self.get_temperature(t)
        R_current = self.get_resistance(i, temperature)

        # Circuit dynamics: L * di/dt = -R(i,T) * i + u
        di_dt = (-R_current * i + u) / self.L

        return di_dt

    def print_configuration(self):
        """Print circuit and PID controller configuration"""
        print(f"\n=== {self.circuit_id} Configuration ===")
        print(f"Circuit ID: {self.circuit_id}")
        print(f"Inductance (L): {self.L} H")
        if self.use_variable_temperature:
            print("Using variable temperature from CSV")
            temp_min = float(self.temperature_data.min())
            temp_max = float(self.temperature_data.max())
            print(f"Temperature range: {temp_min:.3f} to {temp_max:.3f} °C over time")

        else:
            print(f"Temperature: {self.temperature}°C")

        if self.use_variable_resistance:
            print("Using variable resistance from CSV")
            current_min = float(self.current_range.min())
            current_max = float(self.current_range.max())
            if self.use_variable_temperature:
                # find min and max of R(I,T)
                raise NotImplementedError(
                    "find Min/max for R over current range and temperature range"
                )
            # compute Resistance range for current range at given temperature
            else:
                R_min = self.get_resistance(current_min)
                R_max = self.get_resistance(current_max)
                print(
                    f"Resistance range: {R_min:.3f} to {R_max:.3f} Ω over current range (T = {self.temperature} °C)"
                )
        else:
            print(f"Constant resistance: {self.R_constant} Ω")

        # Print experimental data configuration
        if self.experimental_data:
            print(f"\nExperimental Data ({len(self.experimental_data)} entries):")
            for exp_data in self.experimental_data:
                print(f"  - {exp_data}")
        else:
            print("\nNo experimental data loaded")

        # Print PID controller configuration
        if self.pid_controller:
            self.pid_controller.print_summary()

    def update_pid_controller(self, pid_controller: PIDController):
        """Update the PID controller"""
        self.pid_controller = pid_controller

    def set_circuit_id(self, circuit_id: str):
        """Update the circuit ID"""
        old_id = self.circuit_id
        self.circuit_id = circuit_id
        print(f"Circuit ID changed from '{old_id}' to '{circuit_id}'")

    def copy(self, new_circuit_id: str = None):
        """Create a copy of this circuit with optionally different ID"""
        if new_circuit_id is None:
            import uuid

            new_circuit_id = f"circuit_{str(uuid.uuid4())[:8]}"

        # Create new circuit with same parameters
        new_circuit = RLCircuitPID(
            R=self.R_constant if not self.use_variable_resistance else 1.0,
            L=self.L,
            pid_controller=self.pid_controller,  # Share the same PID controller
            temperature=self.temperature,
            circuit_id=new_circuit_id,
            experimental_data=self.experimental_data.copy(),  # Copy experimental data list
        )

        # Copy resistance and reference functions if they exist
        if self.use_variable_resistance:
            new_circuit.resistance_func = self.resistance_func
            new_circuit.current_range = self.current_range
            new_circuit.temp_range = self.temp_range
            new_circuit.R_grid = self.R_grid
            new_circuit.use_variable_resistance = True

        new_circuit.reference_func = self.reference_func
        new_circuit.voltage_func = self.voltage_func
        new_circuit.use_csv_data = self.use_csv_data

        if hasattr(self, "time_data"):
            new_circuit.time_data = self.time_data
        if hasattr(self, "current_data"):
            new_circuit.current_data = self.current_data
        if hasattr(self, "voltage_data"):
            new_circuit.voltage_data = self.voltage_data

        # Copy experimental functions
        new_circuit.experimental_functions = self.experimental_functions.copy()

        return new_circuit

    def __repr__(self):
        """String representation of the circuit"""
        return f"RLCircuitPID(id='{self.circuit_id}', L={self.L}, R={'variable' if self.use_variable_resistance else self.R_constant}, T={self.temperature}°C)"

    def validate(self):
        import pandas as pd

        # Check if files exist when specified
        if self.reference_csv:
            try:
                pd.read_csv(self.reference_csv, sep=None, engine="python")
                print(f"✓ Reference CSV loaded: {self.reference_csv}")
            except FileNotFoundError:
                print(f"❌ Reference CSV file not found: {self.reference_csv}")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Error reading reference CSV: {e}")
                sys.exit(1)

        # Check if files exist when specified
        if self.voltage_csv:
            try:
                pd.read_csv(self.voltage_csv, sep=None, engine="python")
                print(f"✓ Input Voltage CSV loaded: {self.voltage_csv}")
            except FileNotFoundError:
                print(f"❌ Input voltage CSV file not found: {self.voltage_csv}")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Error reading reference CSV: {e}")
                sys.exit(1)

        if self.experimental_data:
            """Validate experimental data configuration"""
            valid_types = {"voltage", "current", "temperature"}
            for exp_data in self.experimental_data:
                if exp_data["type"] not in valid_types:
                    raise ValueError(
                        f'Invalid type {exp_data["type"]}. Must be one of: {valid_types}'
                    )

                if not exp_data["key"]:
                    raise ValueError("Key cannot be empty")

                if not exp_data["file"]:
                    raise ValueError("File path cannot be empty")
                else:
                    if exp_data["file"].endswith(".tdms"):
                        # For TDMS files, we'd need additional handling
                        # For now, raise a helpful error
                        raise NotImplementedError(
                            f'TDMS file support not yet implemented for {exp_data["file"]}: Please convert to CSV format.'
                        )

                    try:
                        pd.read_csv(exp_data["file"], sep=None, engine="python")
                        print(f'✓ Experimental Data CSV loaded: {exp_data["file"]}')
                    except FileNotFoundError:
                        print(
                            f'❌ Experimental Data CSV file not found: {exp_data["file"]}'
                        )
                        sys.exit(1)
                    except Exception as e:
                        print(f"❌ Error reading Experimental Data CSV: {e}")
                        sys.exit(1)
