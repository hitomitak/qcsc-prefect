"""H2/STO-3G LUCJ -> QPY -> Slurm -> SQD transpiler -> task_runner (Batch)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from prefect import flow
from prefect.logging import get_run_logger
from pyscf import lib

from examples.prefect_bitcount_demo.qrmi_slurm_runtime import QrmiSlurmRuntime

from .pr42 import load_pr42, source_metadata


@flow(name="qrmi-slurm-h2-lucj-poc", retries=0)
async def h2_lucj_flow(
    *,
    shots: int = 100,
    work_dir: str = "/shared/qrmi-h2-poc",
    reset_mitigation: bool = False,
) -> dict:
    from qcsc_workflow_utility import chem

    if shots < 1:
        raise ValueError("shots must be positive")
    run_dir = Path(work_dir).expanduser().resolve() / f"run_{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    get_run_logger().info("H2 LUCJ work directory: %s", run_dir)
    pr42 = load_pr42()
    params = pr42.CircuitParameters(
        use_reset_mitigation=reset_mitigation,
        sabre_layout_trials=4,
        sabre_swap_trials=2,
        sabre_max_iterations=2,
    )
    with lib.with_omp_threads(1):
        properties = chem.compute_molecular_integrals_from_geometry.fn(
            atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"
        )
    circuit = pr42.create_ansatz_circuits(params, properties)
    runtime = QrmiSlurmRuntime(
        root_dir=run_dir / "quantum",
        transpile_strategy="sqd-pr42",
        sqd_circuit_parameters=params.model_dump(),
    )
    result = await runtime.sampler([(circuit,)], shots=shots)
    registers = {"meas": 4}
    if reset_mitigation:
        registers["test"] = 4
    counts = {}
    for name, width in registers.items():
        samples = getattr(result[0].data, name, None)
        if samples is None or samples.num_bits != width or samples.num_shots != shots:
            raise RuntimeError(f"Unexpected {name} readout; expected {shots} shots of {width} bits")
        counts[name] = samples.get_counts()
    summary = {
        "molecule": "H2",
        "basis": "sto-3g",
        "bond_length_angstrom": 0.74,
        "logical_qubits": 4,
        "quantum_mode": "batch",
        "transpile_strategy": "sqd-pr42",
        "source_commit": source_metadata()["commit"],
        "shots": shots,
        "registers": registers,
        "counts": counts,
        "slurm": result.metadata["qcsc_slurm"],
        "work_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--work-dir", default="/shared/qrmi-h2-poc")
    parser.add_argument("--reset-mitigation", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(h2_lucj_flow(**vars(args))), indent=2))
