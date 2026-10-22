"""Two Slurm jobs: QRMI Batch sampling through task_runner, then MPI BitCount."""

from __future__ import annotations

import argparse
import asyncio
import json
from array import array
from collections import Counter
from pathlib import Path
from uuid import uuid4

from prefect import flow
from prefect.artifacts import create_table_artifact
from prefect.logging import get_run_logger
from qiskit import QuantumCircuit

from .flow_optimized import BITLEN, run_hpc_bitcount_from_input
from .qrmi_slurm_runtime import QrmiSlurmRuntime


def check_counts(bitstrings: list[str], mpi_counts: dict[str, int]) -> dict[str, int]:
    quantum_counts = dict(Counter(bitstrings))
    if quantum_counts != mpi_counts:
        raise RuntimeError("MPI histogram does not match the original sampler bitstrings")
    return quantum_counts


@flow(name="qrmi-slurm-bitcount-poc", retries=0)
async def qrmi_slurm_bitcount_flow(
    *, shots: int = 100, work_dir: str = "/shared/qrmi-bitcount-poc"
) -> dict:
    run_dir = Path(work_dir).expanduser().resolve() / f"run_{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    get_run_logger().info("PoC work directory: %s", run_dir)

    circuit = QuantumCircuit(BITLEN)
    circuit.h(0)
    for index in range(BITLEN - 1):
        circuit.cx(index, index + 1)
    circuit.measure_all()

    runtime = QrmiSlurmRuntime(root_dir=run_dir / "quantum")
    result = await runtime.sampler([(circuit,)], shots=shots)
    bitstrings = result[0].data.meas.get_bitstrings()
    if result[0].data.meas.num_bits != BITLEN:
        raise RuntimeError("Sampler returned an unexpected bit width")

    # Reuse the existing 10-bit MPI executable and the existing BitCount task.
    mpi_dir = run_dir / "bitcount"
    mpi_dir.mkdir()
    with (mpi_dir / "input.bin").open("wb") as stream:
        array("I", (int(bits, 2) for bits in bitstrings)).tofile(stream)
    hpc = await run_hpc_bitcount_from_input(
        command_block_name="poc-bitcount-command",
        execution_profile_block_name="poc-bitcount-execution",
        hpc_profile_block_name="poc-bitcount-hpc",
        job_work_dir=str(mpi_dir),
        script_filename="bitcount.slurm",
    )
    counts = check_counts(bitstrings, hpc["counts"])
    await create_table_artifact(
        key="qrmi-slurm-bitcount-counts",
        table=[
            {"bitstring": bits, "sampler": count, "mpi": hpc["counts"][bits]}
            for bits, count in sorted(counts.items())
        ],
        description="QRMI Batch samples counted by a separate two-node MPI Slurm job.",
    )
    summary = {
        "quantum_mode": "batch",
        "quantum_slurm_job_id": result.metadata["qcsc_slurm"]["job_id"],
        "mpi_slurm_job_id": hpc["job_id"],
        "mpi_nodes": 2,
        "shots": len(bitstrings),
        "counts_match": True,
        "counts": counts,
        "work_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--work-dir", default="/shared/qrmi-bitcount-poc")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(qrmi_slurm_bitcount_flow(**vars(args))), indent=2))
