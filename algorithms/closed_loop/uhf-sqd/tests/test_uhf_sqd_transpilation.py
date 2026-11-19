"""The QRMI compile hooks: configuration, PRE_INIT in prepare, SABRE layout and transpile."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from uhf_sqd import transpilation
from uhf_sqd.flow_params import CircuitParameters


def test_config_comes_from_circuit_parameters_and_is_validated():
    config = transpilation.transpile_config(CircuitParameters())
    assert config == {
        "optimization_level": 3,
        "sabre_max_iterations": 8,
        "sabre_swap_trials": 10,
        "sabre_layout_trials": 1024,
    }
    for bad in (
        {**config, "extra": 1},
        {**config, "optimization_level": 4},
        {**config, "sabre_swap_trials": 0},
        {**config, "sabre_layout_trials": 1.0},
    ):
        with pytest.raises(ValueError):
            transpilation.transpile(QuantumCircuit(1), None, bad)


def test_the_hook_is_importable_by_its_name():
    module, function = transpilation.TRANSPILER.split(":")
    assert module == "uhf_sqd.transpilation" and function == "transpile"


def test_transpile_maps_a_measured_circuit_onto_the_target():
    target = GenericBackendV2(num_qubits=5, seed=1).target
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure(range(3), range(3))
    config = {
        "optimization_level": 1,
        "sabre_max_iterations": 2,
        "sabre_swap_trials": 2,
        "sabre_layout_trials": 4,
    }
    prepared = transpilation.prepare(circuit)
    isa = transpilation.transpile(prepared.circuit, target, config)
    assert isa.num_qubits == 5
    assert set(isa.count_ops()) <= set(target.operation_names) | {"measure", "barrier"}
    again = transpilation.transpile(prepared.circuit, target, config)
    assert isa == again  # the layout search and transpile are seeded
