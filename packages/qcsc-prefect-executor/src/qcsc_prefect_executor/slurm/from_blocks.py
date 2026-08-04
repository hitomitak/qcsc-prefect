from __future__ import annotations

from pathlib import Path

from qcsc_prefect_blocks.common.blocks import HPCProfileBlock

from qcsc_prefect_executor.cloud_logs import CloudLogPolicy
from qcsc_prefect_executor.from_blocks import run_job_from_blocks
from qcsc_prefect_executor.slurm.run import SlurmRunResult


async def run_slurm_job_from_blocks(
    *,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    work_dir: Path,
    script_filename: str,
    user_args: list[str] | None = None,
    watch_poll_interval: float = 10.0,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "slurm-job-metrics",
    cloud_log_policy: CloudLogPolicy | None = None,
) -> SlurmRunResult:
    """Backward-compatible wrapper around ``run_job_from_blocks`` for Slurm."""

    hpc_block = await HPCProfileBlock.load(hpc_profile_block_name)
    if hpc_block.hpc_target != "slurm":
        raise ValueError(
            f"run_slurm_job_from_blocks requires hpc_target='slurm', got '{hpc_block.hpc_target}'."
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
        cloud_log_policy=cloud_log_policy,
    )
    return result
