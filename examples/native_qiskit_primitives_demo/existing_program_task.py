from __future__ import annotations

import argparse
from typing import Any

from prefect import flow, get_run_logger, task


def build_bell_pubs(backend: Any) -> list[Any]:
    """Existing Qiskit program code, kept independent from Prefect."""

    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
        seed_transpiler=123,
    )
    return [pass_manager.run(circuit)]


def run_native_sampler_program(*, backend_name: str, shots: int) -> dict[str, Any]:
    """Existing native Qiskit execution code wrapped by a Prefect task below."""

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    sampler = SamplerV2(mode=backend)
    job = sampler.run(build_bell_pubs(backend), shots=shots)
    result = job.result()
    return {
        "job_id": job.job_id(),
        "shots": shots,
        "result": result,
    }


@task(name="run-existing-native-qiskit-program", retries=0)
def run_existing_native_qiskit_program_task(
    backend_name: str,
    shots: int,
) -> dict[str, Any]:
    """Use Prefect for orchestration while native Qiskit performs execution."""

    logger = get_run_logger()
    output = run_native_sampler_program(backend_name=backend_name, shots=shots)
    logger.info("Submitted native Qiskit job %s.", output["job_id"])
    return output


@flow(name="existing-native-qiskit-program-demo")
def existing_native_qiskit_program_demo(
    backend_name: str = "ibm_fez",
    shots: int = 100,
) -> dict[str, Any]:
    return run_existing_native_qiskit_program_task(
        backend_name=backend_name,
        shots=shots,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap an existing native Qiskit program as a Prefect task."
    )
    parser.add_argument("--backend-name", default="ibm_fez")
    parser.add_argument("--shots", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(
        existing_native_qiskit_program_demo(
            backend_name=args.backend_name,
            shots=args.shots,
        )
    )
