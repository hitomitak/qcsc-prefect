from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prefect.artifacts import create_table_artifact
from prefect.logging import get_run_logger
from qcsc_prefect_adapters.fugaku.builder import FugakuJobRequest, render_script, write_script_file
from qcsc_prefect_adapters.fugaku.runtime import FugakuPJMRuntime
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


def _parse_stats_file(stats_file: Path | None) -> dict[str, str]:
    stats: dict[str, str] = {}
    if stats_file is None or not stats_file.exists():
        return stats

    for line in stats_file.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("Job Statistical Information", "Node Statistical Information")):
            continue
        if " :" not in line:
            continue
        key, _, value = line.partition(" :")
        stats["stats." + key.strip().lower().replace(" ", "_")] = value.strip()
    return stats


def _read_text_if_exists(path: Path) -> str:
    return read_log_text(path)


@dataclass(frozen=True)
class FugakuRunResult:
    """Normalized result returned by `run_fugaku_job`.

    Attributes:
        job_id: PJM job id returned by ``pjsub``.
        exit_status: Integer exit status derived from PJM ``EC`` when present.
        state: Final PJM state, such as ``"EXT"`` or ``"CCL"``.
        job_status: Parsed final PJM status dictionary from ``pjstat``.
    """

    job_id: str
    exit_status: int
    state: str
    job_status: dict[str, Any]


async def run_fugaku_job(
    *,
    work_dir: Path,
    script_filename: str,
    exec_profile: ExecutionProfile,
    req: FugakuJobRequest,
    watch_poll_interval: float = 10.0,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "fugaku-job-metrics",
    cloud_log_policy: CloudLogPolicy | None = None,
) -> FugakuRunResult:
    """Execute a Fugaku job end-to-end from runtime models.

    .. note::
        This function is the high-level executor entrypoint. It internally
        renders a script, submits it, waits for final status, captures logs,
        parses stats, and publishes a metrics artifact.

    Args:
        work_dir: Working directory where scripts and job outputs are written.
        script_filename: Job script filename to create in ``work_dir``.
        exec_profile: Scheduler-independent execution profile.
        req: Fugaku-specific scheduler request fields.
        watch_poll_interval: Poll interval in seconds for job status checks.
        timeout_seconds: Optional timeout for waiting final status.
        metrics_artifact_key: Prefect artifact key for job metrics table.
        cloud_log_policy: Prefect Cloud output and artifact policy. Omission
            preserves the historical logging and artifact behavior.

    Returns:
        `FugakuRunResult` containing job id, exit status, state, and
        final scheduler status payload.
    """

    logger = get_run_logger()
    policy = resolve_cloud_log_policy(cloud_log_policy)

    script_basename = Path(script_filename).name
    script_text = render_script(
        work_dir=work_dir,
        exec_profile=exec_profile,
        req=req,
        script_basename=script_basename,
    )
    script_path = write_script_file(work_dir=work_dir, filename=script_filename, text=script_text)

    runtime = FugakuPJMRuntime()
    submit = await runtime.submit(script_path, cwd=work_dir)
    final_status = await runtime.wait_final_status(
        submit.job_id,
        watch_poll_interval=watch_poll_interval,
        timeout_seconds=timeout_seconds,
    )

    out_file = work_dir / f"{script_basename}.{req.job_name}.out"
    err_file = work_dir / f"{script_basename}.{req.job_name}.err"
    stats_file = work_dir / f"{script_basename}.{req.job_name}.stats"

    stdout = _read_text_if_exists(out_file)
    stderr = _read_text_if_exists(err_file)
    stats = _parse_stats_file(stats_file)
    emit_cloud_job_logs(
        logger=logger,
        policy=policy,
        summary=CloudJobSummary(
            job_id=submit.job_id,
            state=str(final_status.get("ST", "")),
            exit_code=final_status.get("EC"),
            elapsed=final_status.get("ELAPSE") or final_status.get("ELAPSED"),
            node=stats.get("stats.host_name"),
            stdout_path=out_file,
            stderr_path=err_file,
        ),
        stdout=stdout,
        stderr=stderr,
    )

    artifact: dict[str, Any] = {
        "job_id": submit.job_id,
        "state": final_status.get("ST"),
        "exit_code": final_status.get("EC"),
        "stdout_file": str(out_file) if out_file.exists() else None,
        "stderr_file": str(err_file) if err_file.exists() else None,
        "stats_file": str(stats_file) if stats_file.exists() else None,
    }
    artifact.update(stats)

    if policy.should_create_artifact(legacy_default=True):
        await create_table_artifact(
            table=[list(artifact.keys()), list(artifact.values())],
            key=metrics_artifact_key,
        )

    exit_code_text = str(final_status.get("EC", "")).strip()
    if exit_code_text.isdigit():
        exit_status = int(exit_code_text)
    else:
        exit_status = 0 if final_status.get("ST") == "EXT" else -1

    return FugakuRunResult(
        job_id=submit.job_id,
        exit_status=exit_status,
        state=str(final_status.get("ST", "")),
        job_status=final_status,
    )
