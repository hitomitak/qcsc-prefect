from __future__ import annotations

import json
import random
import struct
from pathlib import Path

from prefect import task

from hpc_prefect_adapters.miyabi.builder import MiyabiJobRequest
from hpc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_executor.miyabi.run import run_miyabi_job


async def resolve_loaded_block(value):
    if hasattr(value, "__await__"):
        return await value
    return value


def resolve_queue_and_project(hpc_block: HPCProfileBlock, resource_class: str) -> tuple[str, str]:
    if resource_class == "gpu":
        return hpc_block.queue_gpu, hpc_block.project_gpu
    return hpc_block.queue_cpu, hpc_block.project_cpu


def build_execution_profile(
    *,
    command_block: CommandBlock,
    execution_profile_block: ExecutionProfileBlock,
    user_args: list[str] | None,
) -> ExecutionProfile:
    arguments = list(command_block.default_args)
    if user_args:
        arguments.extend(user_args)

    return ExecutionProfile(
        command_key=command_block.command_name,
        num_nodes=execution_profile_block.num_nodes,
        mpiprocs=execution_profile_block.mpiprocs,
        ompthreads=execution_profile_block.ompthreads,
        walltime=execution_profile_block.walltime,
        launcher=execution_profile_block.launcher,
        mpi_options=list(execution_profile_block.mpi_options),
        modules=list(execution_profile_block.modules),
        environments=dict(execution_profile_block.environments),
        arguments=arguments,
    )


def to_u32_binary(bitstrings: list[str]) -> bytes:
    values = [int(bits, 2) for bits in bitstrings]
    if not values:
        return b""
    return struct.pack(f"<{len(values)}I", *values)


def from_json_counts(data: dict[str, int], bitlen: int) -> dict[str, int]:
    return {format(int(k), f"0{bitlen}b"): int(v) for k, v in data.items()}


def from_bin_counts(raw: bytes, bitlen: int) -> dict[str, int]:
    if len(raw) % 4 != 0:
        raise ValueError("output.bin format is invalid (not aligned to uint32).")

    total_bins = len(raw) // 4
    values = struct.unpack(f"<{total_bins}I", raw)
    return {
        format(i, f"0{bitlen}b"): int(v)
        for i, v in enumerate(values)
        if v > 0
    }


async def load_bitstrings(
    *,
    bit_source: str,
    bitlen: int,
    shots: int,
    quantum_runtime_block_name: str,
    sampler_variable_name: str,
) -> list[str]:
    normalized = bit_source.strip().lower()
    if normalized == "qiskit":
        from prefect.variables import Variable
        from prefect_qiskit import QuantumRuntime
        from qiskit import QuantumCircuit
        from qiskit.transpiler import generate_preset_pass_manager

        runtime = await resolve_loaded_block(QuantumRuntime.load(quantum_runtime_block_name))
        try:
            options = await Variable.get(sampler_variable_name)
        except Exception:
            options = None

        if not options:
            options = {"params": {"shots": shots}}

        target = await runtime.get_target()
        qc_ghz = QuantumCircuit(bitlen)
        qc_ghz.h(0)
        qc_ghz.cx(0, range(1, bitlen))
        qc_ghz.measure_active()
        pm = generate_preset_pass_manager(
            optimization_level=3,
            target=target,
            seed_transpiler=123,
        )
        isa = pm.run(qc_ghz)
        results = await runtime.sampler([(isa,)], options=options)
        return results[0].data.meas.get_bitstrings()

    rng = random.Random(123)
    return [format(rng.getrandbits(bitlen), f"0{bitlen}b") for _ in range(shots)]


@task(name="run-bitcount-json")
async def run_bitcount_json_task(
    *,
    bitstrings: list[str],
    bitlen: int,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    work_dir: str,
    script_filename: str,
    metrics_artifact_key: str,
    user_args: list[str] | None = None,
) -> dict[str, int]:
    command_block = await resolve_loaded_block(CommandBlock.load(command_block_name))
    execution_profile_block = await resolve_loaded_block(ExecutionProfileBlock.load(execution_profile_block_name))
    hpc_block = await resolve_loaded_block(HPCProfileBlock.load(hpc_profile_block_name))

    if execution_profile_block.command_name != command_block.command_name:
        raise ValueError(
            f"Execution profile '{execution_profile_block_name}' is bound to "
            f"'{execution_profile_block.command_name}', but command block is '{command_block.command_name}'."
        )

    executable = hpc_block.executable_map.get(command_block.executable_key)
    if not executable:
        raise KeyError(f"Executable key '{command_block.executable_key}' was not found in HPC profile.")

    queue, project = resolve_queue_and_project(hpc_block, execution_profile_block.resource_class)
    if not project:
        raise ValueError("Project is empty. Update project_cpu/project_gpu in HPCProfileBlock.")

    run_dir = Path(work_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input.bin").write_bytes(to_u32_binary(bitstrings))

    exec_profile = build_execution_profile(
        command_block=command_block,
        execution_profile_block=execution_profile_block,
        user_args=user_args,
    )
    req = MiyabiJobRequest(queue_name=queue, project=project, executable=executable)

    result = await run_miyabi_job(
        work_dir=run_dir,
        script_filename=script_filename,
        exec_profile=exec_profile,
        req=req,
        watch_poll_interval=5.0,
        timeout_seconds=1800,
        metrics_artifact_key=metrics_artifact_key,
    )
    if result.exit_status != 0:
        raise RuntimeError(f"Miyabi job failed with exit_status={result.exit_status}")

    output_file = run_dir / "output.json"
    if not output_file.exists():
        raise FileNotFoundError(f"output.json was not generated: {output_file}")

    raw_counts = json.loads(output_file.read_text())
    return from_json_counts(raw_counts, bitlen)


@task(name="run-bitcount-bin")
async def run_bitcount_bin_task(
    *,
    bitstrings: list[str],
    bitlen: int,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    work_dir: str,
    script_filename: str,
    metrics_artifact_key: str,
    user_args: list[str] | None = None,
) -> dict[str, int]:
    command_block = await resolve_loaded_block(CommandBlock.load(command_block_name))
    execution_profile_block = await resolve_loaded_block(ExecutionProfileBlock.load(execution_profile_block_name))
    hpc_block = await resolve_loaded_block(HPCProfileBlock.load(hpc_profile_block_name))

    if execution_profile_block.command_name != command_block.command_name:
        raise ValueError(
            f"Execution profile '{execution_profile_block_name}' is bound to "
            f"'{execution_profile_block.command_name}', but command block is '{command_block.command_name}'."
        )

    executable = hpc_block.executable_map.get(command_block.executable_key)
    if not executable:
        raise KeyError(f"Executable key '{command_block.executable_key}' was not found in HPC profile.")

    queue, project = resolve_queue_and_project(hpc_block, execution_profile_block.resource_class)
    if not project:
        raise ValueError("Project is empty. Update project_cpu/project_gpu in HPCProfileBlock.")

    run_dir = Path(work_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input.bin").write_bytes(to_u32_binary(bitstrings))

    exec_profile = build_execution_profile(
        command_block=command_block,
        execution_profile_block=execution_profile_block,
        user_args=user_args,
    )
    req = MiyabiJobRequest(queue_name=queue, project=project, executable=executable)

    result = await run_miyabi_job(
        work_dir=run_dir,
        script_filename=script_filename,
        exec_profile=exec_profile,
        req=req,
        watch_poll_interval=5.0,
        timeout_seconds=1800,
        metrics_artifact_key=metrics_artifact_key,
    )
    if result.exit_status != 0:
        raise RuntimeError(f"Miyabi job failed with exit_status={result.exit_status}")

    output_file = run_dir / "output.bin"
    if not output_file.exists():
        raise FileNotFoundError(f"output.bin was not generated: {output_file}")

    raw = output_file.read_bytes()
    return from_bin_counts(raw, bitlen)
