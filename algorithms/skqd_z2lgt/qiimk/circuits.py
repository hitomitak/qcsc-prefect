"""Circuits."""
from collections.abc import Sequence
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit


def make_step_circuits(lattice, plaquette_energy, delta_t, basis_2q):
    """Make circuits."""
    circuits = []

    circuit = QuantumCircuit(lattice.num_links + lattice.num_plaquettes, lattice.num_links)
    circuit.compose(lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuit.compose(lattice.magnetic_evolution(plaquette_energy, delta_t, basis_2q=basis_2q),
                    inplace=True)
    circuit.compose(lattice.electric_evolution(delta_t), inplace=True)
    circuit.compose(lattice.magnetic_evolution(plaquette_energy, delta_t, basis_2q=basis_2q),
                    inplace=True)
    circuit.compose(lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuits.append(circuit)

    circuit = QuantumCircuit(lattice.num_links + lattice.num_plaquettes, lattice.num_links)
    circuit.compose(lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuit.compose(lattice.magnetic_evolution(plaquette_energy, delta_t, basis_2q=basis_2q),
                    inplace=True)
    circuits.append(circuit)

    circuit = QuantumCircuit(lattice.num_links + lattice.num_plaquettes, lattice.num_links)
    circuit.compose(lattice.magnetic_evolution(plaquette_energy, -delta_t, basis_2q=basis_2q),
                    inplace=True)
    circuit.compose(lattice.electric_evolution(-0.5 * delta_t), inplace=True)
    circuits.append(circuit)

    circuit = QuantumCircuit(lattice.num_links + lattice.num_plaquettes, lattice.num_links)
    circuit.measure(range(lattice.num_links), range(lattice.num_links))
    circuits.append(circuit)

    return circuits


def make_plaquette_circuits(dual_lattice, plaquette_energy, delta_t):
    """Make dual lattice circuits."""
    circuits = []

    circuit = QuantumCircuit(dual_lattice.num_plaquettes, dual_lattice.num_plaquettes)
    circuit.compose(dual_lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuit.compose(dual_lattice.magnetic_evolution(plaquette_energy, delta_t), inplace=True)
    circuit.compose(dual_lattice.electric_evolution(delta_t), inplace=True)
    circuit.compose(dual_lattice.magnetic_evolution(plaquette_energy, delta_t), inplace=True)
    circuit.compose(dual_lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuits.append(circuit)

    circuit = QuantumCircuit(dual_lattice.num_plaquettes, dual_lattice.num_plaquettes)
    circuit.compose(dual_lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuit.compose(dual_lattice.magnetic_evolution(plaquette_energy, delta_t), inplace=True)
    circuits.append(circuit)

    circuit = QuantumCircuit(dual_lattice.num_plaquettes, dual_lattice.num_plaquettes)
    circuit.compose(dual_lattice.magnetic_evolution(plaquette_energy, -delta_t), inplace=True)
    circuit.compose(dual_lattice.electric_evolution(-0.5 * delta_t), inplace=True)
    circuits.append(circuit)

    circuit = QuantumCircuit(dual_lattice.num_plaquettes, dual_lattice.num_plaquettes)
    circuit.measure(range(dual_lattice.num_plaquettes), range(dual_lattice.num_plaquettes))
    circuits.append(circuit)

    return circuits


def compose_trotter_circuits(
    step_circuit: QuantumCircuit,
    measure_circuit: QuantumCircuit,
    steps: int | Sequence[int]
):
    """Make Trotter evolution circuits from single-step and measurement circuits."""
    if isinstance(steps, int):
        steps = list(range(1, steps + 1))

    circuits = []
    trotter = step_circuit.copy_empty_like()
    nsteps = 0
    for _ in range(max(steps)):
        trotter.compose(step_circuit, inplace=True)
        nsteps += 1
        if nsteps in steps:
            circuits.append(trotter.compose(measure_circuit))
    return circuits


def trim_idle_qubits(circuit, layin_map):
    """Re-logicalize a transpiled circuit by trimming idle qubits."""
    num_logical_qubits = len(layin_map)
    dag = circuit_to_dag(circuit)
    subdag = next(d for d in dag.separable_circuits(remove_idle_qubits=True)
                  if d.num_qubits() == num_logical_qubits)
    trimmed_circuit = dag_to_circuit(subdag)
    qreg = circuit.qregs[0]
    compose_onto = [layin_map[qreg.index(q)] for q in trimmed_circuit.qubits]
    new_circ = QuantumCircuit(num_logical_qubits, len(circuit.cregs[0]))
    return new_circ.compose(trimmed_circuit, qubits=compose_onto)
