from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prefect.artifacts import create_table_artifact
from prefect.logging import get_run_logger

from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_adapters.fugaku.builder import FugakuJobRequest, render_script, write_script_file
from hpc_prefect_adapters.fugaku.runtime import FugakuPJMRuntime


MAX_LOG_SIZE = 10_000


def truncate_log(text: str) -> str:
    if len(text) > MAX_LOG_SIZE:
        return text[:MAX_LOG_SIZE] + f"... (truncated {len(text) - MAX_LOG_SIZE} chars)"
    return text


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
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


@dataclass(frozen=True)
class FugakuRunResult:
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
) -> FugakuRunResult:
    logger = get_run_logger()

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

    if logs := _read_text_if_exists(out_file):
        logger.info(truncate_log(logs))
    if logs := _read_text_if_exists(err_file):
        logger.error(truncate_log(logs))

    artifact: dict[str, Any] = {
        "job_id": submit.job_id,
        "state": final_status.get("ST"),
        "exit_code": final_status.get("EC"),
        "stdout_file": str(out_file) if out_file.exists() else None,
        "stderr_file": str(err_file) if err_file.exists() else None,
        "stats_file": str(stats_file) if stats_file.exists() else None,
    }
    artifact.update(_parse_stats_file(stats_file))

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

