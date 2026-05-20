from __future__ import annotations

import inspect
import random
from typing import Any, Literal

QuantumSource = Literal["real-device", "random"]


def resolve_shots(*, sampler_options: dict[str, Any], default_shots: int) -> int:
    params = sampler_options.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("'params' in sampler options must be a mapping.")
    return int(params.get("shots", default_shots))


def generate_random_bitstrings(
    *,
    bitlen: int,
    shots: int,
    seed: int,
) -> list[str]:
    if bitlen <= 0:
        raise ValueError("'bitlen' must be positive.")
    if shots < 0:
        raise ValueError("'shots' must be non-negative.")

    rng = random.Random(seed)
    return [format(rng.getrandbits(bitlen), f"0{bitlen}b") for _ in range(shots)]


def _build_ghz_circuit(bitlen: int) -> Any:
    from qiskit import QuantumCircuit

    qc_ghz = QuantumCircuit(bitlen)
    qc_ghz.h(0)
    qc_ghz.cx(0, range(1, bitlen))
    qc_ghz.measure_active()
    return qc_ghz


async def _resolve_loaded_block(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _load_real_device_dependencies() -> tuple[Any, Any, Any]:
    try:
        from qcsc_prefect.integrations.qiskit import QiskitRuntimeConfig, run_sampler_task
    except ModuleNotFoundError as exc:
        if exc.name == "qcsc_prefect":
            raise ModuleNotFoundError(
                "The BitCount real-device quantum source requires qcsc-prefect-qiskit. "
                'Install it with python -m pip install "qcsc-prefect[qiskit]" '
                "before using --quantum-source real-device, or use --quantum-source "
                "random for the lightweight tutorial path."
            ) from exc
        raise

    from qiskit.transpiler import generate_preset_pass_manager

    return QiskitRuntimeConfig, run_sampler_task, generate_preset_pass_manager


async def sample_bitstrings(
    *,
    quantum_source: QuantumSource,
    runtime_block_name: str,
    sampler_options: dict[str, Any],
    bitlen: int,
    default_shots: int,
    random_seed: int,
) -> list[str]:
    shots = resolve_shots(sampler_options=sampler_options, default_shots=default_shots)
    if quantum_source == "random":
        return generate_random_bitstrings(
            bitlen=bitlen,
            shots=shots,
            seed=random_seed,
        )

    QiskitRuntimeConfig, run_sampler_task, generate_preset_pass_manager = (
        _load_real_device_dependencies()
    )
    runtime_config = await _resolve_loaded_block(QiskitRuntimeConfig.load(runtime_block_name))
    backend = runtime_config.get_backend()
    qc_ghz = _build_ghz_circuit(bitlen)
    pm = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        seed_transpiler=123,
    )
    isa = pm.run(qc_ghz)
    output = await run_sampler_task.fn(
        [(isa,)],
        runtime_config=runtime_config,
        shots=shots,
        options=sampler_options,
        artifact_key="bitcount-qiskit-sampler",
    )
    results = output["result"]
    return results[0].data.meas.get_bitstrings()
