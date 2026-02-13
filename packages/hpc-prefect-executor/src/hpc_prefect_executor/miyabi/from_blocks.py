from __future__ import annotations

from pathlib import Path

from hpc_prefect_blocks.common.blocks import HPCProfileBlock
from hpc_prefect_executor.from_blocks import run_job_from_blocks
from hpc_prefect_executor.miyabi.run import MiyabiRunResult


async def run_miyabi_job_from_blocks(
    *,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    work_dir: Path,
    script_filename: str,
    user_args: list[str] | None = None,
    watch_poll_interval: float = 10.0,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "miyabi-job-metrics",
) -> MiyabiRunResult:
    """
    Backward-compatible wrapper around `run_job_from_blocks`.
    """
    hpc_block = await HPCProfileBlock.load(hpc_profile_block_name)
    if hpc_block.hpc_target != "miyabi":
        raise ValueError(
            f"run_miyabi_job_from_blocks requires hpc_target='miyabi', "
            f"got '{hpc_block.hpc_target}'."
        )

    result = await run_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=work_dir,
        script_filename=script_filename,
        user_args=user_args,
        watch_poll_interval=watch_poll_interval,
        timeout_seconds=timeout_seconds,
        metrics_artifact_key=metrics_artifact_key,
    )
    return result
