from __future__ import annotations

import argparse
import asyncio
from array import array
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from prefect import flow, task
from prefect.artifacts import create_table_artifact
from prefect.variables import Variable
from prefect_qiskit import QuantumRuntime
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from hpc_prefect_executor.from_blocks import run_job_from_blocks

BITLEN = 10
MAXVAL = 1 << BITLEN


def _write_input_u32(work_dir: Path, bitstrings: list[str]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    values = array("I", (int(bits, 2) for bits in bitstrings))
    with (work_dir / "input.bin").open("wb") as f:
        values.tofile(f)


def _read_hist_u64(path: Path) -> list[int]:
    hist = array("Q")
    with path.open("rb") as f:
        hist.frombytes(f.read())
    if len(hist) != MAXVAL:
        raise ValueError(f"Unexpected histogram size: expected {MAXVAL}, got {len(hist)}")
    return [int(v) for v in hist]


def _make_job_work_dir(base_work_dir: Path) -> Path:
    base_work_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_dir = base_work_dir / f"job_{timestamp}_{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


@task(name="quantum-sampling-task")
async def run_quantum_sampling_and_prepare_input(
    *,
    runtime_block_name: str,
    options_variable_name: str,
    default_shots: int,
    work_dir: str,
) -> str:
    runtime = await QuantumRuntime.load(runtime_block_name)
    options = await Variable.get(
        options_variable_name,
        default={"params": {"shots": default_shots}},
    )

    target = await runtime.get_target()

    qc_ghz = QuantumCircuit(BITLEN)
    qc_ghz.h(0)
    qc_ghz.cx(0, range(1, BITLEN))
    qc_ghz.measure_active()

    pm = generate_preset_pass_manager(
        optimization_level=3,
        target=target,
        seed_transpiler=123,
    )
    isa = pm.run(qc_ghz)

    results = await runtime.sampler([(isa,)], options=options)
    bitstrings = results[0].data.meas.get_bitstrings()

    resolved_base_work_dir = Path(work_dir).expanduser().resolve()
    job_work_dir = _make_job_work_dir(resolved_base_work_dir)
    _write_input_u32(job_work_dir, bitstrings)
    return str(job_work_dir)


@task(name="hpc-bitcount-task")
async def run_hpc_bitcount_from_input(
    *,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    job_work_dir: str,
    script_filename: str,
) -> dict[str, object]:
    resolved_job_work_dir = Path(job_work_dir).expanduser().resolve()
    result = await run_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=resolved_job_work_dir,
        script_filename=script_filename,
        watch_poll_interval=5.0,
        timeout_seconds=1800,
        metrics_artifact_key="bitcount-optimized-metrics",
    )
    if result.exit_status != 0:
        raise RuntimeError(f"Optimized BitCount job failed: exit_status={result.exit_status}")

    hist = _read_hist_u64(resolved_job_work_dir / "hist_u64.bin")
    counts = {
        format(i, f"0{BITLEN}b"): c
        for i, c in enumerate(hist)
        if c > 0
    }
    return {
        "job_id": result.job_id,
        "counts": counts,
        "work_dir": str(resolved_job_work_dir),
    }


@flow(name="prefect-bitcount-optimized-flow")
async def prefect_bitcount_optimized_flow(
    *,
    runtime_block_name: str = "ibm-runner",
    command_block_name: str = "cmd-bitcount-hist",
    execution_profile_block_name: str = "exec-bitcount-mpi",
    hpc_profile_block_name: str = "hpc-miyabi-bitcount",
    options_variable_name: str = "miyabi-bitcount-options",
    default_shots: int = 100000,
    work_dir: str = "./work/prefect_bitcount_optimized",
    script_filename: str = "bitcount_optimized.job",
):
    job_work_dir = await run_quantum_sampling_and_prepare_input(
        runtime_block_name=runtime_block_name,
        options_variable_name=options_variable_name,
        default_shots=default_shots,
        work_dir=work_dir,
    )
    hpc_result = await run_hpc_bitcount_from_input(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        job_work_dir=job_work_dir,
        script_filename=script_filename,
    )
    counts = hpc_result["counts"]

    await create_table_artifact(
        table=[list(counts.keys()), list(counts.values())],
        key="sampler-count-dict-optimized",
    )

    return {
        "mode": "optimized",
        "job_id": hpc_result["job_id"],
        "shots": int(sum(counts.values())),
        "num_unique_bitstrings": len(counts),
        "work_dir": hpc_result["work_dir"],
    }


miyabi_bitcount_optimized_flow = prefect_bitcount_optimized_flow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimized BitCount tutorial flow.")
    parser.add_argument("--runtime-block", default="ibm-runner")
    parser.add_argument("--command-block", default="cmd-bitcount-hist")
    parser.add_argument("--execution-profile-block", default="exec-bitcount-mpi")
    parser.add_argument("--hpc-profile-block", default="hpc-miyabi-bitcount")
    parser.add_argument("--options-variable", default="miyabi-bitcount-options")
    parser.add_argument("--shots", type=int, default=100000)
    parser.add_argument("--work-dir", default="./work/prefect_bitcount_optimized")
    parser.add_argument("--script-filename", default="bitcount_optimized.job")
    args = parser.parse_args()

    print(
        asyncio.run(
            prefect_bitcount_optimized_flow(
                runtime_block_name=args.runtime_block,
                command_block_name=args.command_block,
                execution_profile_block_name=args.execution_profile_block,
                hpc_profile_block_name=args.hpc_profile_block,
                options_variable_name=args.options_variable,
                default_shots=args.shots,
                work_dir=args.work_dir,
                script_filename=args.script_filename,
            )
        )
    )
