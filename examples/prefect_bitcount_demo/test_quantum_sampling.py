import asyncio
import builtins

import pytest

from examples.prefect_bitcount_demo.quantum_sampling import (
    generate_random_bitstrings,
    resolve_shots,
    sample_bitstrings,
)


def test_resolve_shots_reads_sampler_option_value():
    shots = resolve_shots(
        sampler_options={"params": {"shots": 321}},
        default_shots=100_000,
    )

    assert shots == 321


def test_generate_random_bitstrings_is_reproducible():
    first = generate_random_bitstrings(bitlen=4, shots=5, seed=24)
    second = generate_random_bitstrings(bitlen=4, shots=5, seed=24)

    assert first == second
    assert len(first) == 5
    assert all(len(bits) == 4 for bits in first)


def test_generate_random_bitstrings_accepts_zero_shots():
    assert generate_random_bitstrings(bitlen=4, shots=0, seed=24) == []


def test_sample_bitstrings_random_does_not_import_real_device_dependencies(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "prefect_qiskit" or name.startswith("prefect_qiskit."):
            raise AssertionError("random mode must not import prefect_qiskit")
        if name == "qiskit" or name.startswith("qiskit."):
            raise AssertionError("random mode must not import qiskit")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    bitstrings = asyncio.run(
        sample_bitstrings(
            quantum_source="random",
            runtime_block_name="unused",
            sampler_options={"params": {"shots": 3}},
            bitlen=4,
            default_shots=100_000,
            random_seed=24,
        )
    )

    assert len(bitstrings) == 3
    assert all(len(bits) == 4 for bits in bitstrings)


def test_sample_bitstrings_real_device_missing_prefect_qiskit_has_clear_error(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "prefect_qiskit" or name.startswith("prefect_qiskit."):
            raise ModuleNotFoundError(
                "No module named 'prefect_qiskit'",
                name="prefect_qiskit",
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ModuleNotFoundError, match="requires prefect-qiskit"):
        asyncio.run(
            sample_bitstrings(
                quantum_source="real-device",
                runtime_block_name="ibm-runner",
                sampler_options={"params": {"shots": 3}},
                bitlen=4,
                default_shots=100_000,
                random_seed=24,
            )
        )
