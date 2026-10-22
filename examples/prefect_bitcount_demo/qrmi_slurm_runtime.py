"""Minimal sampler facade using the existing Slurm executor, for this PoC only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from prefect import task
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger
from qcsc_prefect_executor.from_blocks import run_job_from_blocks
from qiskit import qpy
from qiskit.primitives import PrimitiveResult

try:
    from qiskit_ibm_runtime.decoders.result_decoder import ResultDecoder
except ModuleNotFoundError:
    from qiskit_ibm_runtime.utils.result_decoder import ResultDecoder


@dataclass
class QrmiSlurmRuntime:
    """Accept one measured circuit without unbound parameters; transpile on the node.

    This is a PoC facade, not ``prefect_qiskit.QuantumRuntime``. It supports
    ``await runtime.sampler([(circuit,)], shots=100)`` only.
    """

    root_dir: Path

    async def sampler(self, sampler_pubs, *, shots: int = 100) -> PrimitiveResult:
        if len(sampler_pubs) != 1 or len(sampler_pubs[0]) != 1:
            raise ValueError("PoC accepts only [(circuit,)]")
        circuit = sampler_pubs[0][0]
        if circuit.num_parameters or circuit.num_clbits == 0:
            raise ValueError("Use a measured circuit without unbound parameters")
        if shots < 1:
            raise ValueError("shots must be positive")
        return await _sample_on_slurm(circuit, shots, str(self.root_dir))


@task(name="quantum-sampling-qrmi-slurm", retries=0, cache_policy=NO_CACHE, persist_result=False)
async def _sample_on_slurm(circuit, shots: int, root_dir: str) -> PrimitiveResult:
    work_dir = Path(root_dir).expanduser().resolve() / uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=False)
    with (work_dir / "circuit.qpy").open("wb") as stream:
        qpy.dump(circuit, stream)

    # This executor renders a batch script, calls sbatch, and waits via scontrol.
    job = await run_job_from_blocks(
        command_block_name="poc-qrmi-command",
        execution_profile_block_name="poc-qrmi-execution",
        hpc_profile_block_name="poc-qrmi-hpc",
        work_dir=work_dir,
        script_filename="sampler.slurm",
        user_args=["--shots", str(shots)],
        watch_poll_interval=2.0,
        metrics_artifact_key="poc-qrmi-slurm-job",
    )
    result_path = work_dir / "result.json"
    if job.state != "COMPLETED" or job.exit_status != 0:
        raise RuntimeError(f"QRMI Slurm job {job.job_id} failed; see {work_dir}")
    if not result_path.is_file():
        raise RuntimeError(f"QRMI produced no result; see {work_dir}/output.err")

    result = ResultDecoder.decode(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, PrimitiveResult) or len(result) != 1:
        raise RuntimeError(f"Unexpected QRMI result format: {result_path}")
    if result[0].data.meas.num_shots != shots:
        raise RuntimeError(
            f"QRMI returned {result[0].data.meas.num_shots} shots; "
            f"expected {shots}; see {result_path}"
        )
    record = {"job_id": job.job_id, "work_dir": str(work_dir), "mode": "batch"}
    (work_dir / "slurm_job.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    result.metadata["qcsc_slurm"] = record
    get_run_logger().info("Quantum Slurm job %s completed: %s", job.job_id, work_dir)
    return result
