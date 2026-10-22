"""Offline compatibility checks using the unmodified SQD functions from PR #42.

Only CircuitParameters, create_ansatz_circuits and transpile_circuit are loaded
from the pinned source fixture. An identity @task decorator avoids starting a
Prefect flow/server. The function bodies and transpiler passes are unchanged.
"""

from __future__ import annotations

import logging
import socket
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("ffsim", reason="Install examples/qrmi_sqd_validation/requirements.txt")
pytest.importorskip("pyscf")
pytest.importorskip("pydantic_numpy")
pytest.importorskip("qiskit_ibm_runtime")

from pyscf import lib  # noqa: E402
from qiskit import qpy  # noqa: E402
from qiskit.providers.fake_provider import GenericBackendV2  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from examples.qrmi_sqd_validation.pr42 import chem, load_pr42, prepare_qpy_transport  # noqa: E402

LOGGER = logging.getLogger("qrmi-sqd-offline-validation")


@pytest.fixture(scope="module", autouse=True)
def block_network():
    def fail(*args, **kwargs):
        pytest.fail("This validation must not open network connections")

    with (
        patch.object(socket.socket, "connect", side_effect=fail),
        patch.object(socket.socket, "connect_ex", side_effect=fail),
        patch.object(socket, "create_connection", side_effect=fail),
    ):
        yield


@pytest.fixture(scope="module")
def pr42(block_network):
    return load_pr42()


@pytest.fixture(scope="module")
def h2_properties(block_network):
    # Use the same molecular-integral/CCSD utility as the PR, without running
    # its Prefect task wrapper. One thread avoids overhead for this tiny system.
    with lib.with_omp_threads(1), patch.object(chem, "get_run_logger", return_value=LOGGER):
        properties = chem.compute_molecular_integrals_from_geometry.fn(
            atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"
        )
    assert properties.num_orbitals == 2
    assert properties.num_electrons == (1, 1)
    assert np.isfinite(properties.t2).all()
    assert np.linalg.norm(properties.t2) > 0.01
    return properties


def circuit_parameters(pr42, use_reset_mitigation):
    # Reduce only the search budget, not the PR's pass pipeline or opt level.
    return pr42.CircuitParameters(
        use_reset_mitigation=use_reset_mitigation,
        sabre_layout_trials=4,
        sabre_swap_trials=2,
        sabre_max_iterations=2,
    )


def qpy_roundtrip(circuit, path):
    with path.open("wb") as stream:
        qpy.dump(circuit, stream)
    with path.open("rb") as stream:
        (restored,) = qpy.load(stream)
    return restored


def register_shape(circuit):
    return [(register.name, len(register)) for register in circuit.cregs]


def meas_probabilities(circuit):
    """Exact readout probabilities including physical layout and bit order.

    The optional test register measures the initial all-zero state in this
    fixture. Its measurements can be removed for noiseless state simulation;
    this does not validate noisy reset mitigation or hardware support for it.
    """
    meas = next(register for register in circuit.cregs if register.name == "meas")
    unmeasured = circuit.copy_empty_like()
    mapping = []
    for instruction in circuit.data:
        qubits = [circuit.find_bit(bit).index for bit in instruction.qubits]
        clbits = [circuit.find_bit(bit).index for bit in instruction.clbits]
        if instruction.operation.name == "measure":
            if instruction.clbits[0] in meas:
                mapping.append((qubits[0], list(meas).index(instruction.clbits[0])))
        else:
            unmeasured.append(instruction.operation, qubits, clbits)
    assert sorted(index for _, index in mapping) == list(range(len(meas)))
    probabilities = np.zeros(1 << len(meas))
    for basis, probability in enumerate(Statevector.from_instruction(unmeasured).probabilities()):
        outcome = sum(((basis >> qubit) & 1) << bit for qubit, bit in mapping)
        probabilities[outcome] += probability
    return probabilities


def fake_target(with_rzz):
    basis_gates = ["rz", "sx", "x", "cz"]
    if with_rzz:
        basis_gates.append("rzz")
    # More physical than logical qubits exercises ancilla allocation/layout.
    edges = [(i, i + 1) for i in range(5)] + [(i + 1, i) for i in range(5)]
    return GenericBackendV2(
        num_qubits=6, basis_gates=basis_gates, coupling_map=edges, seed=42
    ).target


def assert_isa(circuit, target):
    assert circuit.num_parameters == 0
    assert circuit.num_qubits == target.num_qubits
    for instruction in circuit.data:
        if instruction.operation.name == "barrier":
            continue
        qubits = tuple(circuit.find_bit(bit).index for bit in instruction.qubits)
        assert target.instruction_supported(
            operation_name=instruction.operation.name,
            qargs=qubits,
            parameters=instruction.operation.params,
        ), (instruction.operation.name, qubits, instruction.operation.params)


@pytest.mark.parametrize("use_reset_mitigation", [False, True], ids=["meas", "meas-and-test"])
def test_h2_lucj_qpy_preserves_readout(pr42, h2_properties, tmp_path, use_reset_mitigation):
    params = circuit_parameters(pr42, use_reset_mitigation)
    original = pr42.create_ansatz_circuits(params, h2_properties)
    restored = qpy_roundtrip(original, tmp_path / "lucj.qpy")
    expected_registers = [("meas", 4)] + ([("test", 4)] if use_reset_mitigation else [])
    assert original.num_qubits == restored.num_qubits == 4
    assert original.num_parameters == restored.num_parameters == 0
    assert register_shape(original) == register_shape(restored) == expected_registers
    assert original.count_ops() == restored.count_ops()
    expected = meas_probabilities(original)
    np.testing.assert_allclose(meas_probabilities(restored), expected, atol=1e-12, rtol=0)
    assert np.count_nonzero(expected > 1e-10) >= 2
    # JW spin-block ordering: one alpha and one beta electron for H2.
    for value, probability in enumerate(expected):
        if (value & 0b11).bit_count() != 1 or (value >> 2).bit_count() != 1:
            assert probability < 1e-12


@pytest.mark.parametrize("with_rzz", [False, True], ids=["cz-target", "rzz-target"])
@pytest.mark.parametrize("use_reset_mitigation", [False, True], ids=["meas", "meas-and-test"])
def test_pr_transpiler_accepts_prepared_qpy_lucj(
    pr42, h2_properties, tmp_path, use_reset_mitigation, with_rzz
):
    params = circuit_parameters(pr42, use_reset_mitigation)
    original = pr42.create_ansatz_circuits(params, h2_properties)
    prepared = prepare_qpy_transport(original)
    restored = qpy_roundtrip(prepared, tmp_path / "lucj_transport.qpy")
    target = fake_target(with_rzz)
    expected = meas_probabilities(original)
    assert register_shape(prepared) == register_shape(original)
    assert restored.count_ops() == prepared.count_ops()
    np.testing.assert_allclose(meas_probabilities(restored), expected, atol=1e-12, rtol=0)
    for name, circuit in [("original", original), ("restored", restored)]:
        isa = pr42.transpile_circuit(circuit, target, params, np.random.default_rng(24))
        assert_isa(isa, target)
        assert register_shape(isa) == register_shape(original)
        np.testing.assert_allclose(meas_probabilities(isa), expected, atol=1e-10, rtol=0)
        # Also check the transport after transpilation, which is useful if the
        # integration chooses to serialize an ISA circuit instead of raw LUCJ.
        reloaded_isa = qpy_roundtrip(isa, tmp_path / f"{name}_isa.qpy")
        assert_isa(reloaded_isa, target)
        assert register_shape(reloaded_isa) == register_shape(original)
        np.testing.assert_allclose(meas_probabilities(reloaded_isa), expected, atol=1e-10, rtol=0)


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="Raw LUCJ QPY loses ffsim Gate attributes required by PR #42's PRE_INIT",
)
@pytest.mark.parametrize("use_reset_mitigation", [False, True], ids=["meas", "meas-and-test"])
def test_raw_qpy_before_pr_transpiler_known_limitation(
    pr42, h2_properties, tmp_path, use_reset_mitigation
):
    """Keep the unsupported raw transport path visible; --runxfail reproduces it."""
    params = circuit_parameters(pr42, use_reset_mitigation)
    circuit = pr42.create_ansatz_circuits(params, h2_properties)
    restored = qpy_roundtrip(circuit, tmp_path / "raw_lucj.qpy")
    try:
        pr42.transpile_circuit(restored, fake_target(False), params, np.random.default_rng(24))
    except AttributeError as exc:
        # An unrelated AttributeError must fail normally, rather than being
        # hidden by this narrow, version-specific compatibility expectation.
        if str(exc) != "'Gate' object has no attribute 'norb'":
            raise AssertionError(f"Unexpected transpiler error: {exc}") from exc
        raise
