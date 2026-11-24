"""ROQUO GPU Prefect block hello demo: one small GPU job through ``run_job_from_blocks``."""

from __future__ import annotations

from pathlib import Path

from prefect import flow
from qcsc_prefect_executor.from_blocks import run_job_from_blocks


@flow(name="roquo-prefect-gpu-hello-demo")
async def roquo_gpu_hello_flow(
    *,
    command_block_name: str = "cmd-roquo-gpu-hello",
    execution_profile_block_name: str = "exec-roquo-gpu-1",
    hpc_profile_block_name: str = "hpc-roquo-gpu",
    work_dir: str = "./work/roquo_gpu_hello",
    script_filename: str = "roquo_gpu_hello.slurm",
    exit_code: int = 0,
    timeout_seconds: float = 1800.0,
):
    """Submit ``hello_gpu.sh`` on one GPU, wait for it, and return its job record.

    ``exit_code`` is the status the job script exits with, to check that a failing job is
    reported as failed.
    """
    resolved_work_dir = Path(work_dir).expanduser().resolve()
    result = await run_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=resolved_work_dir,
        script_filename=script_filename,
        user_args=[str(exit_code)],
        watch_poll_interval=10.0,
        timeout_seconds=timeout_seconds,
        metrics_artifact_key="roquo-gpu-hello-demo-metrics",
    )
    return {
        "job_id": result.job_id,
        "exit_status": result.exit_status,
        "state": result.state,
        "node_list": result.job_status.get("NodeList"),
        "stdout_file": str(resolved_work_dir / "output.out"),
        "stderr_file": str(resolved_work_dir / "output.err"),
    }
