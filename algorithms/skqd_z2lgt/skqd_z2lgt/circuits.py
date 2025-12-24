"""Circuits."""
from collections.abc import Sequence
from numbers import Number
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import Target, PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveIdentityEquivalent
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.mwpm import minimum_weight_link_state


def make_step_circuits(
    lattice: TriangularZ2Lattice,
    plaquette_energy: float,
    delta_t: float,
    target: Target,
    charged_vertices: Optional[list[int]] = None,
    layout: Optional[dict[tuple[str, int], int] | list[int]] = None,
    optimization_level: int = 3
) -> tuple[tuple[QuantumCircuit, ...], list[int]]:
    """Make circuits.

    Returns:
        Circuits for initial, Trotter single, identity single, Trotter double,
        forward double, backward double, measurement
    """
    if not layout:
        layout = lattice.layout_heavy_hex(target=target, basis_2q='rzz')
    elif isinstance(layout, dict):
        layout = lattice.layout_heavy_hex(qubit_assignment=layout, target=target, basis_2q='rzz')

    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        target=target,
        initial_layout=layout,
    )
    pm.post_optimization = PassManager(
        [
            FoldRzzAngle(),
            Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
            RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
        ]
    )

    circuits = []

    # Initial
    circuit = QuantumCircuit(lattice.qubit_graph.num_nodes(), lattice.num_links)
    if charged_vertices:
        # Set the initial state to a link configuration corresponding to the charge distribution.
        # Here we assume that the ground state has a significant overlap with the minimum-weight
        # state, which should be a reasonble assumption in the weak-coupling regime.
        link_state = minimum_weight_link_state(charged_vertices, lattice)
        link_qubits = lattice.link_qubits()
        for link_id in np.nonzero(link_state[::-1])[0]:
            circuit.x(link_qubits[link_id])
    circuits.append(pm.run(circuit))

    # Trotter single
    circuit = circuit.copy_empty_like()
    circuit.compose(lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuit.compose(lattice.magnetic_evolution(plaquette_energy, delta_t, basis_2q='cz'),
                    inplace=True)
    circuit.compose(lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuits.append(pm.run(circuit))

    # Identity single
    param = Parameter('delta_t')
    circuit = circuit.copy_empty_like()
    circuit.compose(lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuit.compose(lattice.magnetic_evolution(plaquette_energy, param, basis_2q='cz'),
                    inplace=True)
    circuit.compose(lattice.electric_evolution(-0.5 * delta_t), inplace=True)
    circuits.append(pm.run(circuit).assign_parameters({param: 0.}))

    # Trotter double
    circuit = circuit.copy_empty_like()
    circuit.compose(lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuit.compose(lattice.magnetic_evolution(plaquette_energy, delta_t, basis_2q='rzz'),
                    inplace=True)
    circuit.compose(lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuit_tr = pm.run(circuit)
    circuit_tr.compose(circuit_tr, inplace=True)
    circuits.append(circuit_tr)

    # Identity double
    circuit = circuit.copy_empty_like()
    circuit.compose(lattice.electric_evolution(0.5 * delta_t), inplace=True)
    circuit.compose(lattice.magnetic_evolution(plaquette_energy, delta_t, basis_2q='rzz'),
                    inplace=True)
    circuit_tr = pm.run(circuit)
    circuit = circuit.copy_empty_like()
    circuit.compose(lattice.magnetic_evolution(plaquette_energy, -delta_t, basis_2q='rzz'),
                    inplace=True)
    circuit.compose(lattice.electric_evolution(-0.5 * delta_t), inplace=True)
    circuit_tr.compose(pm.run(circuit), inplace=True)
    circuits.append(circuit_tr)

    # Measurement
    circuit = circuit.copy_empty_like()
    circuit.measure(range(lattice.num_links), range(lattice.num_links))
    circuits.append(pm.run(circuit))

    return tuple(circuits), layout


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
    init: QuantumCircuit,
    single_step: QuantumCircuit,
    double_step: QuantumCircuit,
    measure: QuantumCircuit,
    steps: int | Sequence[int]
):
    """Make Trotter evolution circuits from single-step and measurement circuits."""
    if isinstance(steps, Number):
        steps = list(range(1, int(steps) + 1))

    circuits = []
    for nst in steps:
        circuit = init.copy()
        for _ in range(nst // 2):
            circuit.compose(double_step, inplace=True)
        if nst % 2 == 1:
            circuit.compose(single_step, inplace=True)
        circuit.compose(measure, inplace=True)
        circuits.append(circuit)
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
