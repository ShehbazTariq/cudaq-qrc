import cudaq
import pywt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm, rc
import torch
import torch.nn as nn

import cudaq
import pywt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm, rc
import torch
import torch.nn as nn

##########################################################################
# 1) QuantumSimulatorLayer - calls the child simulator
##########################################################################
class QuantumSimulatorLayer(nn.Module):
    def __init__(self, simulator):
        """
        simulator: An instance of RydbergSimulator (or HeisenbergSimulator, IsingSimulator).
        """
        super().__init__()
        self.simulator = simulator

    def forward(self, x):
        """
        x: (batch_size, feature_dim) Torch tensor
        We'll call simulator.apply_layer(datapoints, show_progress=False).
        """
        x_np = x.detach().cpu().numpy()
        # Child's signature = (datapoints, show_progress=True)
        results_np = self.simulator.apply_layer(
            x_np,
            show_progress=False  # No TQDM
        )
        return torch.tensor(results_np, dtype=torch.float32, device=x.device)


##########################################################################
# 2) Base Class - QuantumDynamics
##########################################################################
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

    def apply_layer(self, datapoints, construct_hamiltonian, show_progress=True):
        """
        This parent expects: (datapoints, construct_hamiltonian, show_progress).
        BUT typically the child class will hide 'construct_hamiltonian' from the user.
        """
        cudaq.set_target("dynamics")
        num_readouts = len(self.readouts)
        n_steps_minus_one = self.n_steps - 1
        num_results = len(datapoints)
        processed_results = np.zeros((num_results, num_readouts * n_steps_minus_one), dtype=np.float64)

        # Precompute initial state
        initial_state = cudaq.State.from_data(
            np.ones((2**self.nsites, 2**self.nsites), dtype=np.complex128) / np.sqrt(2)
        )

        # Wrap with tqdm if desired
        if show_progress:
            data_iter = tqdm(enumerate(datapoints), total=num_results)
        else:
            data_iter = enumerate(datapoints)

        for j, x in data_iter:
            x_matched = self._match_features_to_sites(x)
            hamiltonian = construct_hamiltonian(x_matched)
            evolution_result = cudaq.evolve(
                hamiltonian,
                dimensions=self.dimensions,
                schedule=self.schedule,
                initial_state=initial_state,
                observables=self.readouts,
                collapse_operators=[],
                store_intermediate_results=True
            )
            ev_data = evolution_result.expectation_values()[1:]
            ev_matrix = np.array([[val.expectation() for val in step_vals] for step_vals in ev_data])
            processed_results[j, :] = ev_matrix.T.flatten(order='C')

        return processed_results


##########################################################################
# 3) Child Classes - Rydberg, Heisenberg, Ising
##########################################################################
class RydbergSimulator(QuantumDynamics):
    def __init__(self, nsites, atoms, readouts, omega, t_start, t_end, t_step,
                 t_rate, alpha, V_matrix, id, use_wavelet=False):
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

    def apply_layer(self, datapoints, show_progress=True):
        """
        Child signature: (datapoints, show_progress=True).
        The child automatically supplies 'construct_hamiltonian'
        to the parent, so the user doesn't have to pass it explicitly.
        """
        return super().apply_layer(datapoints, self.construct_hamiltonian, show_progress=show_progress)


class HeisenbergSimulator(QuantumDynamics):
    def __init__(self, nsites, atoms, readouts, J, h, t_start, t_end, t_step, id, use_wavelet=False):
        super().__init__(nsites, atoms, readouts, t_start, t_end, t_step, id, use_wavelet)
        self.J = J
        self.h = h

    def construct_hamiltonian(self, x):
        hamiltonian = cudaq.SpinOperator()
        for j in range(len(self.atoms) - 1):
            hamiltonian += -self.J * (
                cudaq.spin.x(j) * cudaq.spin.x(j + 1)
                + cudaq.spin.y(j) * cudaq.spin.y(j + 1)
                + cudaq.spin.z(j) * cudaq.spin.z(j + 1)
            )
        for j in range(len(self.atoms)):
            hamiltonian += -self.h * cudaq.spin.z(j)
        return hamiltonian

    def apply_layer(self, datapoints, show_progress=True):
        return super().apply_layer(datapoints, self.construct_hamiltonian, show_progress=show_progress)


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

    def apply_layer(self, datapoints, show_progress=True):
        return super().apply_layer(datapoints, self.construct_hamiltonian, show_progress=show_progress)

##########################################################################
# 4) generate_readouts & plot_3d_lattice
##########################################################################
def generate_readouts(nsites, custom_readouts=None):
    """
    Generate a list of readouts using cudaq.spin with either a default configuration
    or based on a custom list of Pauli operator strings.
    """
    if custom_readouts is not None:
        for readout in custom_readouts:
            if len(readout) != nsites:
                raise ValueError(f"Each string in custom_readouts must have length {nsites}.")
        readouts = []
        for readout in custom_readouts:
            operator = None
            for i, pauli in enumerate(readout):
                if pauli == 'I':
                    continue
                elif pauli == 'X':
                    term = cudaq.spin.x(i)
                elif pauli == 'Y':
                    term = cudaq.spin.y(i)
                elif pauli == 'Z':
                    term = cudaq.spin.z(i)
                else:
                    raise ValueError(f"Invalid character '{pauli}'. Allowed: I,X,Y,Z.")
                operator = term if operator is None else operator * term
            readouts.append(operator if operator else cudaq.spin.identity())
        return readouts

    # Default readouts
    readouts = []
    # Single-site Pauli operators
    for i in range(nsites):
        readouts.append(cudaq.spin.x(i))
    for i in range(nsites):
        readouts.append(cudaq.spin.y(i))
    for i in range(nsites):
        readouts.append(cudaq.spin.z(i))
    # Two-site correlators
    for i in range(nsites):
        for j in range(i + 1, nsites):
            readouts.extend([
                cudaq.spin.x(i) * cudaq.spin.x(j),
                cudaq.spin.x(i) * cudaq.spin.y(j),
                cudaq.spin.x(i) * cudaq.spin.z(j),
                cudaq.spin.y(i) * cudaq.spin.y(j),
                cudaq.spin.y(i) * cudaq.spin.z(j),
                cudaq.spin.z(i) * cudaq.spin.z(j)
            ])
    return readouts

def plot_3d_lattice(nsites, d, atoms, alpha, V_matrix, save_as_pdf=False, filename="3d_lattice.pdf"):
    """
    Plot a 3D lattice of atoms with interaction strengths visualized as colored lines using a gradient.
    """
    rc('text', usetex=True)
    rc('font', family='serif')

    V_min, V_max = V_matrix.min(), V_matrix.max()
    V_normalized = (V_matrix - V_min) / (V_max - V_min)

    x_positions = np.zeros(nsites)
    y_positions = np.zeros(nsites)
    z_positions = np.zeros(nsites)

    for i in range(1, nsites):
        strongest_interaction = np.max(V_normalized[:i, i])
        distance = d * (1.5 - strongest_interaction)
        x_positions[i] = x_positions[i - 1] + np.random.uniform(-distance, distance)
        y_positions[i] = y_positions[i - 1] + np.random.uniform(-distance, distance)
        z_positions[i] = z_positions[i - 1] + np.random.uniform(-distance, distance)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_positions, y_positions, z_positions, s=200, c='blue')
    for i, (xx, yy, zz) in enumerate(zip(x_positions, y_positions, z_positions)):
        ax.text(xx, yy, zz, f"{i}", ha='center', va='center', fontsize=10,
                color='white', bbox=dict(boxstyle="circle", fc="blue"))

    cmap = plt.colormaps.get_cmap('bwr')
    for i in range(nsites):
        for j in range(i + 1, nsites):
            interaction_strength = V_normalized[i, j]
            color = cmap(interaction_strength)
            ax.plot(
                [x_positions[i], x_positions[j]],
                [y_positions[i], y_positions[j]],
                [z_positions[i], z_positions[j]],
                color=color, linewidth=1, linestyle="--"
            )

    ax.set_xlabel(r"$x$ ($\mu m$)")
    ax.set_ylabel(r"$y$ ($\mu m$)")
    ax.set_zlabel(r"$z$ ($\mu m$)")

    max_range = np.array([
        x_positions.max() - x_positions.min(),
        y_positions.max() - y_positions.min(),
        z_positions.max() - z_positions.min()
    ]).max() / 2.0
    mid_x = (x_positions.max() + x_positions.min()) * 0.5
    mid_y = (y_positions.max() + y_positions.min()) * 0.5
    mid_z = (z_positions.max() + z_positions.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.grid(True)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label(r"Interaction Strength", fontsize=10)

    if save_as_pdf:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Plot saved as {filename}")
    else:
        plt.show()
