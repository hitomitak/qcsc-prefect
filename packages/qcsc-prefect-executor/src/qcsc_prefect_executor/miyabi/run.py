from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from prefect.artifacts import create_table_artifact
from prefect.logging import get_run_logger
from qcsc_prefect_adapters.miyabi.builder import (
    MiyabiJobRequest,
    render_script,
    write_script_file,
)
from qcsc_prefect_adapters.miyabi.runtime import MiyabiPBSRuntime
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
    """
    Mirror existing truncate behavior to avoid 422 Unprocessable Entity.
    """
    return _truncate_cloud_log(text)


def _create_job_artifact(*, job_id: str, job_status: dict[str, str]) -> dict[str, Any]:
    """
    Copy of your existing _create_job_artifact logic (pure function version).
    """

    def _format_time(time_str: str | None) -> str | None:
        if time_str is None:
            return None
        try:
            dt = datetime.strptime(time_str, r"%a %b %d %H:%M:%S %Y")
        except ValueError:
            return time_str
        return dt.isoformat()

    def _to_gb(text: str) -> str:
        if text.endswith("kb"):
            gib = int(text[:-2]) / (1024**2)
            return f"{gib:.5f}gb"
        return text

    report_base = {
        "job_id": job_id,
        "job_name": job_status.get("Job_Name", None),
        "queue": job_status.get("queue", None),
        "resource_list": job_status.get("Resource_List.select", None),
        "token": job_status.get("TOKEN", None),
        "exit_status": job_status.get("Exit_status", None),
        "submit_host": job_status.get("Submit_Host", None),
    }

    if vnodes := job_status.get("exec_vnode", None):
        for vnode in vnodes.split("+"):
            node_name, spec = vnode.strip("()").split(":", 1)
            report_base[f"exec_vnode:{node_name}"] = spec

    report_time = {
        "creation_time": _format_time(job_status.get("ctime", None)),
        "queue_time": _format_time(job_status.get("qtime", None)),
        "eligible_time": _format_time(job_status.get("etime", None)),
        "start_time": _format_time(job_status.get("stime", None)),
        "modify_time": _format_time(job_status.get("mtime", None)),
    }

    report_resource_used: dict[str, Any] = {}
    for key, value in job_status.items():
        if key.startswith("resources_used."):
            k = key.removeprefix("resources_used.").strip()
            report_resource_used[k] = _to_gb(value)

    if mem_per_nodes := report_resource_used.pop("mem_per_nodes", None):
        for node_name, mem_used in json.loads(str(mem_per_nodes).strip("'")).items():
            report_resource_used[f"mem_per_nodes:{node_name}"] = _to_gb(mem_used)

    return {
        **report_base,
        **report_time,
        **report_resource_used,
    }


def _resolve_log_file_path(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = str(path).strip().strip('"').strip("'")
    if ":" in raw:
        _, raw = raw.split(":", 1)
    return raw


def _read_text_if_exists(path: str | Path | None) -> str:
    file_path = _resolve_log_file_path(path)
    if not file_path or not os.path.exists(file_path):
        return ""
    return read_log_text(file_path)


@dataclass(frozen=True)
class MiyabiRunResult:
    """Normalized result returned by `run_miyabi_job`.

    Attributes:
        job_id: PBS job id returned by ``qsub``.
        exit_status: Integer PBS ``Exit_status`` value.
        job_status: Parsed final PBS status dictionary from ``qstat -fH``.
    """

    job_id: str
    exit_status: int
    job_status: dict[str, Any]


async def run_miyabi_job(
    *,
    work_dir: Path,
    script_filename: str,
    exec_profile: ExecutionProfile,
    req: MiyabiJobRequest,
    watch_poll_interval: float = 10.0,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "miyabi-job-metrics",
    cloud_log_policy: CloudLogPolicy | None = None,
) -> MiyabiRunResult:
    """Execute a Miyabi job end-to-end from runtime models.

    .. note::
        This function is the high-level executor entrypoint. It internally
        renders a script, submits it, waits for final status, captures logs,
        and publishes a metrics artifact.

    Args:
        work_dir: Working directory where scripts and job outputs are written.
        script_filename: Job script filename to create in ``work_dir``.
        exec_profile: Scheduler-independent execution profile.
        req: Miyabi-specific scheduler request fields.
        watch_poll_interval: Poll interval in seconds for job status checks.
        timeout_seconds: Optional timeout for waiting final status.
        metrics_artifact_key: Prefect artifact key for job metrics table.
        cloud_log_policy: Prefect Cloud output and artifact policy. Omission
            preserves the historical logging and artifact behavior.

    Returns:
        `MiyabiRunResult` containing job id, exit status, and final
        scheduler status payload.
    """
    logger = get_run_logger()
    policy = resolve_cloud_log_policy(cloud_log_policy)

    # 1) build script
    script_text = render_script(work_dir=work_dir, exec_profile=exec_profile, req=req)
    script_path = write_script_file(work_dir=work_dir, filename=script_filename, text=script_text)

    # 2) submit & wait
    if policy.mode == "legacy":
        logger.info(f"Create Script file in {script_path}")
    rt = MiyabiPBSRuntime()
    submit = await rt.submit(script_path, cwd=work_dir)
    job_id = submit.job_id

    final_status_any = await rt.wait_final_status(
        job_id,
        watch_poll_interval=watch_poll_interval,
        timeout_seconds=timeout_seconds,
    )

    # Type-cast-ish: existing artifact builder expects dict[str, str]
    final_status: dict[str, str] = {str(k): str(v) for k, v in final_status_any.items()}

    # 3) transfer stdout/err into Prefect logs (same behavior)
    out_path = final_status.get("Output_Path")
    err_path = final_status.get("Error_Path")

    stdout_file = _resolve_log_file_path(out_path)
    stderr_file = _resolve_log_file_path(err_path)
    stdout = _read_text_if_exists(out_path)
    stderr = _read_text_if_exists(err_path)
    emit_cloud_job_logs(
        logger=logger,
        policy=policy,
        summary=CloudJobSummary(
            job_id=job_id,
            state=final_status.get("job_state") or final_status.get("state"),
            exit_code=final_status.get("Exit_status"),
            elapsed=final_status.get("resources_used.walltime"),
            node=final_status.get("exec_host") or final_status.get("exec_vnode"),
            stdout_path=stdout_file or None,
            stderr_path=stderr_file or None,
        ),
        stdout=stdout,
        stderr=stderr,
    )

    # 4) metrics artifact (table) (same shape as existing code)
    art_dict = _create_job_artifact(job_id=job_id, job_status=final_status)
    if policy.should_create_artifact(legacy_default=True):
        await create_table_artifact(
            table=[list(art_dict.keys()), list(art_dict.values())],
            key=metrics_artifact_key,
        )

    # existing behavior returns Exit_status as int
    exit_status = int(final_status.get("Exit_status", "0") or "0")
    return MiyabiRunResult(job_id=job_id, exit_status=exit_status, job_status=final_status_any)
