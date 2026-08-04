from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prefect.artifacts import create_table_artifact
from prefect.logging import get_run_logger
from qcsc_prefect_adapters.slurm.builder import SlurmJobRequest, render_script, write_script_file
from qcsc_prefect_adapters.slurm.runtime import SlurmRuntime
from qcsc_prefect_core.models.execution_profile import ExecutionProfile

from qcsc_prefect_executor.cloud_logs import (
    MAX_CLOUD_LOG_CHARS,
    CloudJobSummary,
    CloudLogPolicy,
    emit_cloud_job_logs,
    read_log_text,
    resolve_cloud_log_policy,
)
from qcsc_prefect_executor.cloud_logs import (
    truncate_log as _truncate_cloud_log,
)

MAX_LOG_SIZE = MAX_CLOUD_LOG_CHARS


def truncate_log(text: str) -> str:
    """Truncate large log text to the configured maximum length."""

    return _truncate_cloud_log(text)


def _read_text_if_exists(path: Path) -> str:
    return read_log_text(path)


def _create_job_artifact(
    *,
    job_id: str,
    job_status: dict[str, Any],
    stdout_file: Path,
    stderr_file: Path,
) -> dict[str, Any]:
    exit_code_text = str(job_status.get("ExitCode", "-1:0"))
    exit_code, _, signal = exit_code_text.partition(":")
    return {
        "job_id": job_id,
        "state": job_status.get("State"),
        "exit_code": exit_code,
        "signal": signal,
        "elapsed_time": job_status.get("Elapsed"),
        "allocated_cpus": job_status.get("AllocCPUS"),
        "node_list": job_status.get("NodeList"),
        "stdout_file": str(stdout_file) if stdout_file.exists() else None,
        "stderr_file": str(stderr_file) if stderr_file.exists() else None,
    }


@dataclass(frozen=True)
class SlurmRunResult:
    """Normalized result returned by `run_slurm_job`.

    Attributes:
        job_id: Slurm job id returned by ``sbatch``.
        exit_status: Integer process exit code parsed from Slurm ``ExitCode``.
        state: Final Slurm job state.
        job_status: Parsed terminal ``sacct`` status dictionary.
    """

    job_id: str
    exit_status: int
    state: str
    job_status: dict[str, Any]


async def run_slurm_job(
    *,
    work_dir: Path,
    script_filename: str,
    exec_profile: ExecutionProfile,
    req: SlurmJobRequest,
    watch_poll_interval: float = 10.0,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "slurm-job-metrics",
    cloud_log_policy: CloudLogPolicy | None = None,
) -> SlurmRunResult:
    """Execute a Slurm job end-to-end from runtime models.

    This high-level executor renders the Slurm script, submits it with
    ``sbatch``, waits for terminal ``sacct`` status, captures stdout/stderr
    files, and publishes a Prefect table artifact with scheduler metrics.

    Args:
        work_dir: Working directory where scripts and job outputs are written.
        script_filename: Job script filename to create in ``work_dir``.
        exec_profile: Scheduler-independent execution profile.
        req: Slurm-specific scheduler request fields.
        watch_poll_interval: Poll interval in seconds for job status checks.
        timeout_seconds: Optional timeout for waiting final status.
        metrics_artifact_key: Prefect artifact key for job metrics table.
        cloud_log_policy: Prefect Cloud output and artifact policy. Omission
            preserves the historical logging and artifact behavior.

    Returns:
        `SlurmRunResult` containing job id, exit status, state, and
        final scheduler status payload.
    """

    logger = get_run_logger()
    policy = resolve_cloud_log_policy(cloud_log_policy)

    script_text = render_script(work_dir=work_dir, exec_profile=exec_profile, req=req)
    script_path = write_script_file(work_dir=work_dir, filename=script_filename, text=script_text)

    runtime = SlurmRuntime()
    submit = await runtime.submit(script_path, cwd=work_dir)
    final_status = await runtime.wait_final_status(
        submit.job_id,
        watch_poll_interval=watch_poll_interval,
        timeout_seconds=timeout_seconds,
    )

    stdout_file = work_dir / "output.out"
    stderr_file = work_dir / "output.err"

    stdout = _read_text_if_exists(stdout_file)
    stderr = _read_text_if_exists(stderr_file)
    exit_code_text = str(final_status.get("ExitCode", "-1:0"))
    emit_cloud_job_logs(
        logger=logger,
        policy=policy,
        summary=CloudJobSummary(
            job_id=submit.job_id,
            state=str(final_status.get("State", "")),
            exit_code=exit_code_text.partition(":")[0],
            elapsed=final_status.get("Elapsed"),
            node=final_status.get("NodeList"),
            stdout_path=stdout_file,
            stderr_path=stderr_file,
        ),
        stdout=stdout,
        stderr=stderr,
    )

    if final_status and policy.should_create_artifact(legacy_default=True):
        artifact = _create_job_artifact(
            job_id=submit.job_id,
            job_status=final_status,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
        )
        await create_table_artifact(
            table=[list(artifact.keys()), list(artifact.values())],
            key=metrics_artifact_key,
        )

    exit_code_text = exit_code_text.split(":", 1)[0]
    exit_status = int(exit_code_text) if exit_code_text.isdigit() else -1

    return SlurmRunResult(
        job_id=submit.job_id,
        exit_status=exit_status,
        state=str(final_status.get("State", "")),
        job_status=final_status,
    )
