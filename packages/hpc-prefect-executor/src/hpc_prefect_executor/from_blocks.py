from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

from hpc_prefect_adapters.fugaku.builder import FugakuJobRequest
from hpc_prefect_adapters.miyabi.builder import MiyabiJobRequest
from hpc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_executor.fugaku.run import run_fugaku_job
from hpc_prefect_executor.miyabi.run import run_miyabi_job


async def _resolve_loaded_block(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _resolve_queue_and_project(hpc_block: HPCProfileBlock, resource_class: str) -> tuple[str, str]:
    if resource_class == "gpu":
        return hpc_block.queue_gpu, hpc_block.project_gpu
    return hpc_block.queue_cpu, hpc_block.project_cpu


def _build_execution_profile(
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


def _default_fugaku_job_name(command_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", command_name).strip("-")
    if not normalized:
        return "prefect-job"
    return normalized[:63]


async def run_job_from_blocks(
    *,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    work_dir: Path,
    script_filename: str,
    user_args: list[str] | None = None,
    watch_poll_interval: float = 10.0,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "hpc-job-metrics",
    fugaku_job_name: str | None = None,
) -> Any:
    """
    Resolve block names into runtime models and execute on the target HPC system.

    Hidden from workflow code:
    - Block loading
    - ExecutionProfile conversion
    - Target-specific request creation
    - Target-specific executor dispatch
    """
    command_block = await _resolve_loaded_block(CommandBlock.load(command_block_name))
    execution_profile_block = await _resolve_loaded_block(
        ExecutionProfileBlock.load(execution_profile_block_name)
    )
    hpc_block = await _resolve_loaded_block(HPCProfileBlock.load(hpc_profile_block_name))

    if execution_profile_block.command_name != command_block.command_name:
        raise ValueError(
            f"ExecutionProfileBlock '{execution_profile_block_name}' is for command "
            f"'{execution_profile_block.command_name}', but command block "
            f"'{command_block_name}' is '{command_block.command_name}'."
        )

    executable = hpc_block.executable_map.get(command_block.executable_key)
    if not executable:
        raise KeyError(
            f"Executable key '{command_block.executable_key}' was not found in "
            f"HPCProfileBlock '{hpc_profile_block_name}'."
        )

    queue, project = _resolve_queue_and_project(hpc_block, execution_profile_block.resource_class)
    if not project:
        raise ValueError("Project is empty. Update HPCProfileBlock project_cpu/project_gpu.")

    exec_profile = _build_execution_profile(
        command_block=command_block,
        execution_profile_block=execution_profile_block,
        user_args=user_args,
    )
    resolved_work_dir = Path(work_dir).expanduser().resolve()

    if hpc_block.hpc_target == "miyabi":
        req = MiyabiJobRequest(
            queue_name=queue,
            project=project,
            executable=executable,
        )
        return await run_miyabi_job(
            work_dir=resolved_work_dir,
            script_filename=script_filename,
            exec_profile=exec_profile,
            req=req,
            watch_poll_interval=watch_poll_interval,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
        )

    if hpc_block.hpc_target == "fugaku":
        req = FugakuJobRequest(
            queue_name=queue,
            project=project,
            executable=executable,
            job_name=fugaku_job_name or _default_fugaku_job_name(command_block.command_name),
            gfscache=hpc_block.gfscache or "/vol0002",
            spack_modules=list(hpc_block.spack_modules) if hpc_block.spack_modules else [],
            mpi_options_for_pjm=list(hpc_block.mpi_options_for_pjm) if hpc_block.mpi_options_for_pjm else [],
            pjm_resources=list(hpc_block.pjm_resources) if hpc_block.pjm_resources else [],
        )
        return await run_fugaku_job(
            work_dir=resolved_work_dir,
            script_filename=script_filename,
            exec_profile=exec_profile,
            req=req,
            watch_poll_interval=watch_poll_interval,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
        )

    raise NotImplementedError(
        f"hpc_target='{hpc_block.hpc_target}' is not supported yet by run_job_from_blocks."
    )
