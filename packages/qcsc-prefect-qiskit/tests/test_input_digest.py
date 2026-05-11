from __future__ import annotations

from dataclasses import dataclass

from qcsc_prefect.integrations.qiskit import input_digest as digest_mod
from qcsc_prefect.integrations.qiskit.cache import qiskit_sampler_submit_cache_key
from qcsc_prefect.integrations.qiskit.input_digest import (
    build_qiskit_estimator_input_digest,
    build_qiskit_input_digest_payload,
    build_qiskit_sampler_input_digest,
    qiskit_input_digest_from_payload,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


def _bell_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit


def _x_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()
    return circuit


@dataclass
class _SamplerPubObject:
    circuit: QuantumCircuit
    parameter_values: list[float] | None = None
    shots: int | None = None


class _Backend:
    name = "ibm_kawasaki"


def test_sampler_input_digest_is_stable_for_same_inputs_in_same_scope():
    first = build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
        options={"b": 2, "a": {"z": [3, 1]}},
    )
    second = build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
        options={"a": {"z": [3, 1]}, "b": 2},
    )

    assert first == second
    assert first.startswith("qiskit-sampler-input-")


def test_sampler_input_digest_defaults_to_flow_scope(monkeypatch):
    monkeypatch.setattr(digest_mod, "_current_prefect_flow_key", lambda: "name:flow-a")
    first = build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
    )

    monkeypatch.setattr(digest_mod, "_current_prefect_flow_key", lambda: "name:flow-b")
    second = build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
    )

    assert first != second


def test_sampler_input_digest_can_use_global_cross_flow_scope(monkeypatch):
    monkeypatch.setattr(digest_mod, "_current_prefect_flow_key", lambda: "name:flow-a")
    first = build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
        cache_scope="global",
    )

    monkeypatch.setattr(digest_mod, "_current_prefect_flow_key", lambda: "name:flow-b")
    second = build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
        cache_scope="global",
    )

    assert first == second


def test_sampler_input_digest_changes_when_backend_changes():
    first = build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
    )
    second = build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_osaka",
        shots=1024,
    )

    assert first != second


def test_sampler_input_digest_changes_when_circuit_changes():
    assert build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
    ) != build_qiskit_sampler_input_digest(
        [_x_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
    )


def test_sampler_input_digest_changes_when_shots_change():
    assert build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
    ) != build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=2048,
    )


def test_sampler_input_digest_changes_when_options_change():
    assert build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
        options={"params": {"resilience_level": 1}},
    ) != build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
        options={"params": {"resilience_level": 2}},
    )


def test_sampler_input_digest_accepts_pub_tuple_and_pub_object():
    circuit = _bell_circuit()
    tuple_digest = build_qiskit_sampler_input_digest(
        [(circuit, [0.1], 100)],
        backend_name=_Backend(),
        shots=1024,
    )
    object_digest = build_qiskit_sampler_input_digest(
        [_SamplerPubObject(circuit=circuit, parameter_values=[0.1], shots=100)],
        backend_name=_Backend(),
        shots=1024,
    )

    assert tuple_digest == object_digest
    assert tuple_digest.startswith("qiskit-sampler-input-")
    assert object_digest.startswith("qiskit-sampler-input-")


def test_estimator_input_digest_changes_when_precision_changes():
    pub = (_bell_circuit(), SparsePauliOp.from_list([("ZZ", 1.0)]))

    assert build_qiskit_estimator_input_digest(
        [pub],
        backend_name="ibm_kawasaki",
        precision=0.01,
    ) != build_qiskit_estimator_input_digest(
        [pub],
        backend_name="ibm_kawasaki",
        precision=0.02,
    )


def test_estimator_input_digest_changes_when_observable_changes():
    first = (_bell_circuit(), SparsePauliOp.from_list([("ZZ", 1.0)]))
    second = (_bell_circuit(), SparsePauliOp.from_list([("ZI", 1.0)]))

    assert build_qiskit_estimator_input_digest(
        [first],
        backend_name="ibm_kawasaki",
        precision=0.01,
    ) != build_qiskit_estimator_input_digest(
        [second],
        backend_name="ibm_kawasaki",
        precision=0.01,
    )


def test_input_digest_payload_and_hash_are_canonical():
    payload_a = build_qiskit_input_digest_payload(
        program_type="sampler",
        pubs=[_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
        extra={"tags": {"b", "a"}},
    )
    payload_b = build_qiskit_input_digest_payload(
        program_type="sampler",
        pubs=[_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
        extra={"tags": {"a", "b"}},
    )

    assert payload_a == payload_b
    assert qiskit_input_digest_from_payload(payload_a) == qiskit_input_digest_from_payload(
        payload_b
    )


def test_input_digest_payload_stores_circuit_hash_not_qasm_body():
    payload = build_qiskit_input_digest_payload(
        program_type="sampler",
        pubs=[_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
    )
    circuit_payload = payload["pubs"][0]["circuit"]

    assert circuit_payload["format"] == "qasm3"
    assert "qasm3_sha256" in circuit_payload
    assert "qasm3_length" in circuit_payload
    assert "qasm3" not in circuit_payload


def test_sampler_submit_cache_key_can_use_generated_input_digest():
    input_digest = build_qiskit_sampler_input_digest(
        [_bell_circuit()],
        backend_name="ibm_kawasaki",
        shots=1024,
    )

    key = qiskit_sampler_submit_cache_key(
        None,
        {
            "runtime_block_name": "ibm-runtime",
            "shots": 1024,
            "input_digest": input_digest,
        },
    )

    assert key is not None
    assert key.startswith("qiskit-sampler-")
