import cudaq
import pywt
import numpy as np
from tqdm import tqdm


class QuantumDynamics:
    def __init__(self, nsites, atoms, readouts, t_start, t_end, t_step, id, use_wavelet=False):
        self.nsites = nsites
        self.atoms = atoms
        self.readouts = readouts
        self.t_start = t_start
        self.t_end = t_end
        self.t_step = t_step
        self.id = id
        self.use_wavelet = use_wavelet
        self.dimensions = {i: 2 for i in range(len(atoms))}
        self.n_steps = int((t_end - t_start) / t_step) + 1
        self.steps = np.linspace(0, t_end, self.n_steps)
        self.schedule = cudaq.Schedule(self.steps, ["t"])



    def _match_features_to_sites(self, x):
        if len(x) > self.nsites:
            return x[:self.nsites]
        elif len(x) < self.nsites:
            return np.concatenate([x, np.zeros(self.nsites - len(x))])
        return x

    def _apply_wavelet_transform(self, x):
        if not self.use_wavelet:
            return np.array(x)
        wavelet = 'db1'
        coeffs = pywt.wavedec(x, wavelet, level=None)
        transformed_x = np.concatenate(coeffs)[:len(self.atoms)]
        max_val = np.max(np.abs(transformed_x))
        return transformed_x / max_val if max_val != 0 else transformed_x

    def apply_layer(self, datapoints, construct_hamiltonian):
        cudaq.set_target("dynamics")
        num_readouts = len(self.readouts)
        n_steps_minus_one = self.n_steps - 1
        num_results = len(datapoints)
        processed_results = np.zeros((num_results, num_readouts * n_steps_minus_one))

        for j, x in tqdm(enumerate(datapoints), total=num_results):
            x_matched = self._match_features_to_sites(x)
            hamiltonian = construct_hamiltonian(x_matched)
            evolution_result = cudaq.evolve(hamiltonian,
                                            dimensions=self.dimensions,
                                            schedule=self.schedule,
                                            initial_state=cudaq.State.from_data(
    np.ones((2**self.nsites, 2**self.nsites), dtype=np.complex128) / np.sqrt(2)),
                                            observables=self.readouts,
                                            collapse_operators=[],
                                            store_intermediate_results=True)
            ev_data = evolution_result.expectation_values()[1:]
            ev_matrix = np.array([[val.expectation() for val in step_vals] for step_vals in ev_data])
            processed_results[j, :] = ev_matrix.T.flatten(order='C')
        return processed_results



class RydbergSimulator(QuantumDynamics):
    def __init__(self, nsites, atoms, readouts, omega, t_start, t_end, t_step, t_rate, alpha, V_matrix, id, use_wavelet=False):
        super().__init__(nsites, atoms, readouts, t_start, t_end, t_step, id, use_wavelet)
        self.omega = omega
        self.t_rate = t_rate
        self.alpha = alpha
        self.V_matrix = V_matrix

    def construct_hamiltonian(self, x):
        hamiltonian = cudaq.SpinOperator()
        for j in range(len(self.atoms)):
            hamiltonian += (self.omega / 2) * cudaq.spin.x(j)
        for j in range(len(self.atoms)):
            for k in range(j + 1, len(self.atoms)):
                hamiltonian += self.V_matrix[j, k] * cudaq.spin.z(j) * cudaq.spin.z(k)
        for j in range(len(self.atoms)):
            delta_j = x[j] + self.alpha[j] * x[j]
            hamiltonian -= delta_j * cudaq.spin.z(j)
        return hamiltonian

    def apply_layer(self, datapoints):
        return super().apply_layer(datapoints, self.construct_hamiltonian)

    
class HeisenbergSimulator(QuantumDynamics):
    def __init__(self, nsites, atoms, readouts, J, h, t_start, t_end, t_step, id, use_wavelet=False):
        super().__init__(nsites, atoms, readouts, t_start, t_end, t_step, id, use_wavelet)
        self.J = J
        self.h = h

    def construct_hamiltonian(self, x):
        hamiltonian = cudaq.SpinOperator()
        for j in range(len(self.atoms) - 1):
            hamiltonian += -self.J * (cudaq.spin.x(j) * cudaq.spin.x(j + 1) +
                                      cudaq.spin.y(j) * cudaq.spin.y(j + 1) +
                                      cudaq.spin.z(j) * cudaq.spin.z(j + 1))
        for j in range(len(self.atoms)):
            hamiltonian += -self.h * cudaq.spin.z(j)
        return hamiltonian

    def apply_layer(self, datapoints):
        return super().apply_layer(datapoints, self.construct_hamiltonian)


class IsingSimulator(QuantumDynamics):
    def __init__(self, nsites, atoms, readouts, h_x, J, t_start, t_end, t_step, id, use_wavelet=False):
        super().__init__(nsites, atoms, readouts, t_start, t_end, t_step, id, use_wavelet)
        self.h_x = h_x
        self.J = J

    def construct_hamiltonian(self, x):
        hamiltonian = cudaq.SpinOperator()
        for j in range(len(self.atoms)):
            hamiltonian += self.h_x * cudaq.spin.x(j)
        for j in range(len(self.atoms)):
            for k in range(j + 1, len(self.atoms)):
                hamiltonian += self.J * cudaq.spin.z(j) * cudaq.spin.z(k)
        return hamiltonian

    def apply_layer(self, datapoints):
        return super().apply_layer(datapoints, self.construct_hamiltonian)
    
    
def generate_readouts(nsites, custom_readouts=None):
    """
    Generate a list of readouts using cudaq.spin with either a default configuration
    or based on a custom list of Pauli operator strings.

    Parameters:
        nsites (int): Number of sites.
        custom_readouts (list, optional): List of strings representing the Pauli operators.

    Returns:
        list: List of cudaq.spin operators.

    Raises:
        ValueError: If custom_readouts contains strings of length not equal to nsites.
    """

    if custom_readouts is not None:
        # Validate custom readouts
        for readout in custom_readouts:
            if len(readout) != nsites:
                raise ValueError(f"Each string in custom_readouts must have a length of {nsites}.")

        # Convert strings to cudaq.spin operators
        readouts = []
        for readout in custom_readouts:
            operator = None
            for i, pauli in enumerate(readout):
                if pauli == 'X':
                    term = cudaq.spin.x(i)
                elif pauli == 'Y':
                    term = cudaq.spin.y(i)
                elif pauli == 'Z':
                    term = cudaq.spin.z(i)
                else:
                    raise ValueError(f"Invalid character '{pauli}' in readout string. Allowed characters: 'X', 'Y', 'Z'.")

                operator = term if operator is None else operator * term
            readouts.append(operator)
        return readouts

    # Default configuration
    readouts = []

    # Single-site Pauli operators
    readouts.extend([cudaq.spin.x(i) for i in range(nsites)])
    readouts.extend([cudaq.spin.y(i) for i in range(nsites)])
    readouts.extend([cudaq.spin.z(i) for i in range(nsites)])

    # Two-site correlators (only unique combinations)
    for i in range(nsites):
        for j in range(i + 1, nsites):
            readouts.extend([
                cudaq.spin.x(i) * cudaq.spin.x(j),  # XᵢXⱼ
                cudaq.spin.x(i) * cudaq.spin.y(j),  # XᵢYⱼ
                cudaq.spin.x(i) * cudaq.spin.z(j),  # XᵢZⱼ
                cudaq.spin.y(i) * cudaq.spin.y(j),  # YᵢYⱼ
                cudaq.spin.y(i) * cudaq.spin.z(j),  # YᵢZⱼ
                cudaq.spin.z(i) * cudaq.spin.z(j)   # ZᵢZⱼ
            ])

    return readouts