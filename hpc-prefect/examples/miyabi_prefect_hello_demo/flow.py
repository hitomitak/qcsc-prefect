from __future__ import annotations

import inspect
from pathlib import Path

from prefect import flow

from hpc_prefect_adapters.miyabi.builder import MiyabiJobRequest
from hpc_prefect_blocks.miyabi.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_executor.miyabi.run import run_miyabi_job


async def _resolve_loaded_block(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _resolve_queue_and_project(hpc_block: HPCProfileBlock, resource_class: str) -> tuple[str, str]:
    if resource_class == "gpu":
        return hpc_block.queue_gpu, hpc_block.project_gpu
    return hpc_block.queue_cpu, hpc_block.project_cpu


@flow(name="miyabi-prefect-block-hello-demo")
async def miyabi_prefect_block_hello_flow(
    *,
    command_block_name: str = "cmd-hello-demo",
    execution_profile_block_name: str = "exec-hello-single",
    hpc_profile_block_name: str = "hpc-miyabi",
    work_dir: str = "./work/miyabi_prefect_block_hello",
    script_filename: str = "hello_demo.pbs",
    user_args: list[str] | None = None,
):
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

    arguments = list(cmd.default_args)
    if user_args:
        arguments.extend(user_args)

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
        arguments=arguments,
    )

    req = MiyabiJobRequest(
        queue_name=queue,
        project=project,
        executable=executable,
    )

    result = await run_miyabi_job(
        work_dir=Path(work_dir).expanduser().resolve(),
        script_filename=script_filename,
        exec_profile=exec_profile,
        req=req,
        watch_poll_interval=5.0,
        timeout_seconds=600,
        metrics_artifact_key="miyabi-hello-demo-metrics",
    )
    return {
        "job_id": result.job_id,
        "exit_status": result.exit_status,
        "work_dir": str(Path(work_dir).expanduser().resolve()),
    }
