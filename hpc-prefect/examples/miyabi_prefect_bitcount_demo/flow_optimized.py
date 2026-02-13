from __future__ import annotations

import argparse
import asyncio
import inspect
from array import array
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from prefect import flow
from prefect.artifacts import create_table_artifact
from prefect.variables import Variable
from prefect_qiskit import QuantumRuntime
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from hpc_prefect_adapters.miyabi.builder import MiyabiJobRequest
from hpc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_executor.miyabi.run import run_miyabi_job

BITLEN = 10
MAXVAL = 1 << BITLEN


async def _resolve_loaded_block(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _resolve_queue_and_project(hpc_block: HPCProfileBlock, resource_class: str) -> tuple[str, str]:
    if resource_class == "gpu":
        return hpc_block.queue_gpu, hpc_block.project_gpu
    return hpc_block.queue_cpu, hpc_block.project_cpu


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


@flow(name="miyabi-bitcount-optimized-flow")
async def miyabi_bitcount_optimized_flow(
    *,
    runtime_block_name: str = "ibm-runner",
    command_block_name: str = "cmd-bitcount-hist",
    execution_profile_block_name: str = "exec-bitcount-mpi",
    hpc_profile_block_name: str = "hpc-miyabi-bitcount",
    options_variable_name: str = "miyabi-bitcount-options",
    default_shots: int = 100000,
    work_dir: str = "./work/miyabi_bitcount_optimized",
    script_filename: str = "bitcount_optimized.pbs",
):
    runtime = await QuantumRuntime.load(runtime_block_name)
    cmd = await _resolve_loaded_block(CommandBlock.load(command_block_name))
    profile_block = await _resolve_loaded_block(ExecutionProfileBlock.load(execution_profile_block_name))
    hpc_block = await _resolve_loaded_block(HPCProfileBlock.load(hpc_profile_block_name))

    if profile_block.command_name != cmd.command_name:
        raise ValueError(
            f"ExecutionProfileBlock '{profile_block.profile_name}' is for command "
            f"'{profile_block.command_name}', but command block is '{cmd.command_name}'."
        )

    executable = hpc_block.executable_map.get(cmd.executable_key)
    if not executable:
        raise KeyError(f"Executable key '{cmd.executable_key}' was not found in HPCProfileBlock.")

    queue, project = _resolve_queue_and_project(hpc_block, profile_block.resource_class)
    if not project:
        raise ValueError("Project is empty. Update HPCProfileBlock project_cpu/project_gpu.")

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

    exec_profile = ExecutionProfile(
        command_key=cmd.command_name,
        num_nodes=profile_block.num_nodes,
        mpiprocs=profile_block.mpiprocs,
        ompthreads=profile_block.ompthreads,
        walltime=profile_block.walltime,
        launcher=profile_block.launcher,
        mpi_options=list(profile_block.mpi_options),
        modules=list(profile_block.modules),
        environments=dict(profile_block.environments),
        arguments=list(cmd.default_args),
    )

    req = MiyabiJobRequest(
        queue_name=queue,
        project=project,
        executable=executable,
    )

    result = await run_miyabi_job(
        work_dir=job_work_dir,
        script_filename=script_filename,
        exec_profile=exec_profile,
        req=req,
        watch_poll_interval=5.0,
        timeout_seconds=1800,
        metrics_artifact_key="miyabi-bitcount-optimized-metrics",
    )
    if result.exit_status != 0:
        raise RuntimeError(f"Optimized BitCount job failed: exit_status={result.exit_status}")

    hist = _read_hist_u64(job_work_dir / "hist_u64.bin")
    counts = {
        format(i, f"0{BITLEN}b"): c
        for i, c in enumerate(hist)
        if c > 0
    }

    await create_table_artifact(
        table=[list(counts.keys()), list(counts.values())],
        key="sampler-count-dict-optimized",
    )

    return {
        "mode": "optimized",
        "job_id": result.job_id,
        "shots": int(sum(counts.values())),
        "num_unique_bitstrings": len(counts),
        "work_dir": str(job_work_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimized Miyabi BitCount tutorial flow.")
    parser.add_argument("--runtime-block", default="ibm-runner")
    parser.add_argument("--command-block", default="cmd-bitcount-hist")
    parser.add_argument("--execution-profile-block", default="exec-bitcount-mpi")
    parser.add_argument("--hpc-profile-block", default="hpc-miyabi-bitcount")
    parser.add_argument("--options-variable", default="miyabi-bitcount-options")
    parser.add_argument("--shots", type=int, default=100000)
    parser.add_argument("--work-dir", default="./work/miyabi_bitcount_optimized")
    parser.add_argument("--script-filename", default="bitcount_optimized.pbs")
    args = parser.parse_args()

    print(
        asyncio.run(
            miyabi_bitcount_optimized_flow(
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
