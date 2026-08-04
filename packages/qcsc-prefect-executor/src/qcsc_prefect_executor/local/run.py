from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from prefect.artifacts import create_table_artifact
from prefect.logging import get_run_logger
from qcsc_prefect_adapters.local.runtime import (
    LocalJobRequest,
    LocalRuntime,
    build_local_command,
)
from qcsc_prefect_core.models.execution_profile import ExecutionProfile

from qcsc_prefect_executor.cloud_logs import (
    MAX_CLOUD_LOG_CHARS,
    CloudJobSummary,
    CloudLogPolicy,
    emit_cloud_job_logs,
    resolve_cloud_log_policy,
    truncate_log,
)

MAX_LOG_SIZE = MAX_CLOUD_LOG_CHARS


def _truncate_log(text: str) -> str:
    return truncate_log(text)


@dataclass(frozen=True)
class LocalRunResult:
    """Result returned after a command exits on the local Prefect worker."""

    job_id: str
    exit_status: int
    stdout: str
    stderr: str


async def run_local_job(
    *,
    work_dir: Path,
    exec_profile: ExecutionProfile,
    req: LocalJobRequest,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "local-job-metrics",
    cloud_log_policy: CloudLogPolicy | None = None,
) -> LocalRunResult:
    """Execute a command directly without generating or submitting a job script."""

    command = build_local_command(exec_profile=exec_profile, executable=req.executable)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger = get_run_logger()
    policy = resolve_cloud_log_policy(cloud_log_policy)
    process_result = await LocalRuntime().execute(
        command,
        cwd=work_dir,
        environments=exec_profile.environments,
        timeout_seconds=timeout_seconds,
    )

    job_id = f"local-{process_result.pid}"
    emit_cloud_job_logs(
        logger=logger,
        policy=policy,
        summary=CloudJobSummary(
            job_id=job_id,
            state="SUCCEEDED" if process_result.exit_status == 0 else "FAILED",
            exit_code=process_result.exit_status,
            node="local",
        ),
        stdout=process_result.stdout,
        stderr=process_result.stderr,
    )

    artifact = {
        "job_id": job_id,
        "exit_status": process_result.exit_status,
        "command": shlex.join(command),
        "work_dir": str(work_dir),
    }
    if policy.should_create_artifact(legacy_default=True):
        await create_table_artifact(
            table=[list(artifact.keys()), list(artifact.values())],
            key=metrics_artifact_key,
        )

    return LocalRunResult(
        job_id=job_id,
        exit_status=process_result.exit_status,
        stdout=process_result.stdout,
        stderr=process_result.stderr,
    )
