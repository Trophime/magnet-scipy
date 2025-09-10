import numpy as np
from typing import List, Tuple


from .rlcircuitpid import RLCircuitPID


class CoupledRLCircuitsPID:
    """
    Multiple RL Circuits with Magnetic Coupling and Independent PID Controllers

    Each circuit is electrically independent but magnetically coupled through
    mutual inductances. Uses RLCircuitPID instances directly for maximum flexibility.
    """

    def __init__(
        self,
        circuits: List[RLCircuitPID],  # List of RLCircuitPID instances
        mutual_inductances: np.ndarray = None,
    ):
        """
        Initialize coupled RL circuits

        Args:
            circuits: List of RLCircuitPID instances
            mutual_inductances: NxN matrix of mutual inductances M[i,j] between circuits i and j
                               If None, creates symmetric coupling matrix
        """
        self.circuits = circuits
        self.n_circuits = len(circuits)

        # Validate that all circuits have circuit_id
        self.circuit_ids = []
        for i, circuit in enumerate(self.circuits):
            if not hasattr(circuit, "circuit_id") or circuit.circuit_id is None:
                raise ValueError(f"Circuit {i} must have a circuit_id")
            self.circuit_ids.append(circuit.circuit_id)

        # Check for duplicate circuit IDs
        if len(set(self.circuit_ids)) != len(self.circuit_ids):
            raise ValueError("All circuits must have unique circuit_id values")

        # Initialize mutual inductance matrix
        L_values = [circuit.L for circuit in self.circuits]
        self.M = np.zeros((self.n_circuits, self.n_circuits))
        np.fill_diagonal(self.M, L_values)

        if mutual_inductances is not None:
            extra_diag_terms = int(self.n_circuits * (self.n_circuits - 1) / 2)
            if extra_diag_terms != 0:
                if mutual_inductances.shape[0] != (extra_diag_terms):
                    raise ValueError(
                        f"Mutual inductance matrix must be {extra_diag_terms}: shape={mutual_inductances.shape[0]}"
                    )

                # create a symetric 2d numpy array from mutual_inductances
                M = np.array(mutual_inductances)
                print(f"loaded extra_diag terms: {M}")
                triu_indices = np.triu_indices(self.n_circuits, k=1)
                print(
                    f"Setting mutual inductances for {extra_diag_terms}: triu_indices={triu_indices}"
                )
                self.M[triu_indices] = M
                self.M = self.M + self.M.T - np.diag(np.diag(self.M))  # Make symmetric

        else:
            if self.n_circuits >= 1:
                raise RuntimeError("mutual_inductances must be defined")

        # Validate mutual inductance matrix
        self._validate_coupling_matrix()

        # Pid Controller params

        print(f"Initialized {self.n_circuits} coupled RL circuits")
        print(f"Circuit IDs: {self.circuit_ids}")

    def _validate_coupling_matrix(self):
        """Validate the mutual inductance matrix"""
        # Check symmetry
        if not np.allclose(self.M, self.M.T):
            raise RuntimeError("Mutual inductance matrix is not symmetric")

        # Check diagonal contains positive self-inductances (NOT zero!)
        if np.any(np.diag(self.M) <= 0):
            raise RuntimeError("Inductance matrix has non-positive diagonal elements")

        # Validate positive definiteness (for physical realizability)
        eigenvals = np.linalg.eigvals(self.M)
        if np.any(eigenvals <= 0):
            raise RuntimeError("Inductance matrix has non-positive eigenvalues. ")

    def get_resistance(self, circuit_idx: int, current: float) -> float:
        """Get resistance for a specific circuit"""
        circuit = self.circuits[circuit_idx]
        return circuit.get_resistance(abs(current))

    def get_reference_current(self, circuit_idx: int, t: float) -> float:
        """Get reference current for a specific circuit"""
        circuit = self.circuits[circuit_idx]
        return circuit.reference_current(t)

    def get_resistances(
        self, currents: List[float], temperatures: List[float] = None
    ) -> List[float]:
        """Get resistance for a specific circuit"""
        # print(f"get_resistances: currents={currents}, temperatures={temperatures}")
        if temperatures is not None:
            return [
                circuit.get_resistance(abs(currents[i]), temperatures[i])
                for i, circuit in enumerate(self.circuits)
            ]
        else:
            return [
                circuit.get_resistance(abs(currents[i]))
                for i, circuit in enumerate(self.circuits)
            ]

    def get_reference_currents(self, t: float) -> List[float]:
        """Get reference current for a specific circuit"""
        return [circuit.reference_current(t) for circuit in self.circuits]

    def get_pid_parameters(
        self, i_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get PID parameters for all circuits"""
        Kp = np.zeros(self.n_circuits)
        Ki = np.zeros(self.n_circuits)
        Kd = np.zeros(self.n_circuits)
        for n, circuit in enumerate(self.circuits):
            kp, ki, kd = circuit.get_pid_parameters(i_ref[n])
            Kp[n] = kp
            Ki[n] = ki
            Kd[n] = kd
        return Kp, Ki, Kd

    def get_current_region(self, circuit_idx: int, i_ref: float) -> str:
        """Get current region name for a specific circuit"""
        circuit = self.circuits[circuit_idx]
        return circuit.get_current_region(i_ref)

    def voltage_vector_field(self, t: float, y, u: np.ndarray = None):
        """
        RL circuit ODE
        """
        # print(f"DEBUG: t={t:.3f}, y={y}, y.shape={y.shape}")

        i = y  # Current is an array of ncircuit dimension

        # Get voltage from CSV data
        tutu = ""
        if u is None:
            voltages = [circuit.input_voltage(t) for circuit in self.circuits]
            u = np.array(voltages)
            tutu = " ****"
        # print(f"DEBUG: voltages u={u} {tutu}")

        # Get current-dependent resistance
        temperatures = [circuit.get_temperature(t) for circuit in self.circuits]
        """
        print(
            f"DEBUG: temperatures={temperatures} {tutu}"
        )
        """
        R_current = np.array(self.get_resistances(i.tolist(), temperatures))
        """
        print(
            f"DEBUG: R_current R={R_current} {tutu}"
        )
        """

        net_voltages = u - R_current * i  # Net voltages after resistive drop
        try:
            di_dt = np.linalg.solve(self.M, net_voltages)
        except np.linalg.LinAlgError as e:
            # raise RuntimeError(f"Singular inductance matrix: {e}")
            print(
                f"Warning: Singular inductance matrix encountered ({e}). Using pseudo-inverse."
            )
            di_dt = np.linalg.pinv(self.M) @ net_voltages
        return di_dt

    # TODO: by default make di_ref_dt: np.ndarray of dimensions ncircuits with zeros
    def vector_field(
        self, t: float, y: np.ndarray, di_ref_dt: np.ndarray = None
    ) -> np.ndarray:
        """
        Define the system dynamics as a vector field with variable resistance and adaptive PID

        State vector y = [i, integral_error]
        where:
        - i: current
        - integral_error: integral of error for PID
        """
        i = y[: self.n_circuits]
        integral_error = y[self.n_circuits :]
        # print(f"DEBUG: currents i={i} integram_error={integral_error}")

        # Get parameters
        temperatures = [circuit.get_temperature(t) for circuit in self.circuits]
        R_current = np.array(self.get_resistances(i, temperatures))
        i_ref = np.array(self.get_reference_currents(t))

        Kp, Ki, Kd = self.get_pid_parameters(i_ref)

        # Analytical di/dt
        numerator = (
            -(R_current + Kp) * i + Kp * i_ref + Ki * integral_error + Kd * di_ref_dt
        )

        M_ = np.zeros((self.n_circuits, self.n_circuits))
        np.fill_diagonal(M_, Kd)
        M_ += self.M
        try:
            di_dt = np.linalg.solve(M_, numerator)
        except np.linalg.LinAlgError as e:
            # raise RuntimeError(f"Singular matrix: {e}")
            print(f"Warning: Singular matrix encountered ({e}). Using pseudo-inverse.")
            di_dt = np.linalg.pinv(M_) @ numerator

        # Integral error evolution
        error = i_ref - i
        dintegral_dt = error

        return np.concatenate([di_dt, dintegral_dt])

    def get_initial_conditions(self, mode: str = "regular") -> np.ndarray:
        """Get initial conditions for all circuits"""
        # Each circuit: [current] if mode="regular", [current, integral_error] other wise
        if mode == "regular":
            y0 = np.zeros(self.n_circuits)
        else:
            y0 = np.zeros(self.n_circuits * 2)
        return y0

    def print_configuration(self):
        """Print configuration of all circuits and coupling"""
        print("\n=== Coupled RL Circuits Configuration ===")
        print(f"Number of circuits: {self.n_circuits}")
        print(f"Circuit IDs: {self.circuit_ids}")

        print("\nMutual Inductance Matrix (H):")
        for i in range(self.n_circuits):
            row_str = "  "
            for j in range(self.n_circuits):
                row_str += f"{float(self.M[i,j]):8.6f} "
            print(row_str)

        print("\nIndividual Circuit Configurations:")
        for i, circuit in enumerate(self.circuits):
            circuit.print_configuration()

        print("\n=== Coupled RL Circuits Configuration loaded ===")

    def update_mutual_inductance(self, i: int, j: int, M_ij: float):
        """Update a specific mutual inductance value"""
        if i >= self.n_circuits or j >= self.n_circuits:
            raise ValueError("Circuit indices out of range")

        # Update symmetrically
        self.M = self.M.at[i, j].set(M_ij)
        self.M = self.M.at[j, i].set(M_ij)

    def add_circuit(self, circuit) -> None:
        """
        Add a new circuit to the system (requires rebuilding coupling matrix)

        Args:
            circuit: RLCircuitPID instance to add
        """
        if not hasattr(circuit, "circuit_id") or circuit.circuit_id is None:
            raise ValueError("Circuit must have a circuit_id")

        if circuit.circuit_id in self.circuit_ids:
            raise ValueError(f"Circuit ID '{circuit.circuit_id}' already exists")

        self.circuits.append(circuit)
        self.circuit_ids.append(circuit.circuit_id)
        old_n_circuits = self.n_circuits
        self.n_circuits += 1

        # Rebuild mutual inductance matrix with default coupling to new circuit
        old_M = self.M

        # Create new matrix
        new_M = np.zeros((self.n_circuits, self.n_circuits))

        # Copy old matrix
        new_M = new_M.at[:old_n_circuits, :old_n_circuits].set(old_M)

        self.M = new_M
        print(f"Added circuit '{circuit.circuit_id}'")

    def remove_circuit(self, circuit_id: str) -> None:
        """
        Remove a circuit from the system

        Args:
            circuit_id: ID of the circuit to remove
        """
        if circuit_id not in self.circuit_ids:
            raise ValueError(f"Circuit ID '{circuit_id}' not found")

        if self.n_circuits <= 2:
            raise ValueError(
                "Cannot remove circuit - need at least 2 circuits for coupling"
            )

        # Find index of circuit to remove
        remove_idx = self.circuit_ids.index(circuit_id)

        # Remove circuit from list
        self.circuits.pop(remove_idx)
        self.circuit_ids.remove(circuit_id)
        self.n_circuits -= 1

        # Rebuild mutual inductance matrix
        indices_to_keep = [i for i in range(self.n_circuits + 1) if i != remove_idx]
        new_M = self.M[np.ix_(indices_to_keep, indices_to_keep)]
        self.M = new_M

        print(f"Removed circuit '{circuit_id}'")

    def get_circuit_names(self) -> List[str]:
        """Get list of circuit IDs"""
        return self.circuit_ids.copy()

    def get_coupling_strength(self, i: int, j: int) -> float:
        """Get mutual inductance between circuits i and j"""
        if i >= self.n_circuits or j >= self.n_circuits:
            raise ValueError("Circuit indices out of range")
        return float(self.M[i, j])

    def get_circuit_by_id(self, circuit_id: str):
        """Get a circuit by its ID"""
        for circuit in self.circuits:
            if circuit.circuit_id == circuit_id:
                return circuit
        raise ValueError(f"Circuit ID '{circuit_id}' not found")

    def get_circuit_by_index(self, index: int):
        """Get a circuit by its index"""
        if 0 <= index < self.n_circuits:
            return self.circuits[index]
        raise ValueError(f"Circuit index {index} out of range [0, {self.n_circuits-1}]")

    def get_circuit_index(self, circuit_id: str) -> int:
        """Get the index of a circuit by its ID"""
        try:
            return self.circuit_ids.index(circuit_id)
        except ValueError:
            raise ValueError(f"Circuit ID '{circuit_id}' not found")

    def set_coupling_matrix(self, mutual_inductances: np.ndarray):
        """Set the entire mutual inductance matrix"""
        if mutual_inductances.shape != (self.n_circuits, self.n_circuits):
            raise ValueError(
                f"Mutual inductance matrix must be {self.n_circuits}x{self.n_circuits}"
            )

        self.M = np.array(mutual_inductances)
        self._validate_coupling_matrix()
        print("Updated mutual inductance matrix")

    def get_coupling_matrix(self) -> np.ndarray:
        """Get the current mutual inductance matrix"""
        return np.array(self.M)

    def __repr__(self) -> str:
        """String representation of the coupled system"""
        return f"CoupledRLCircuitsPID(n_circuits={self.n_circuits}, ids={self.circuit_ids})"
