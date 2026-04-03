"""Helpers for queue-aware GB-SQD submission on Miyabi."""

from __future__ import annotations

import asyncio
import getpass
import importlib
from typing import Any

from ._workspace_support import ensure_workspace_packages

ensure_workspace_packages("qcsc-prefect-adapters")
_runtime = importlib.import_module("qcsc_prefect_adapters.miyabi.runtime")
run_command = _runtime.run_command


TERMINAL_STATES = {"C", "F"}


def parse_qstat_listing(stdout: str) -> list[dict[str, Any]]:
    """Parse ``qstat -f`` output into job dictionaries."""

    jobs: list[dict[str, Any]] = []
    current_job: dict[str, Any] | None = None
    current_key = ""

    for line in stdout.splitlines():
        if line.startswith("Job Id: "):
            if current_job:
                jobs.append(current_job)
            current_job = {"Job_Id": line.split(":", 1)[1].strip()}
            current_key = ""
            continue

        if current_job is None or not line.strip():
            continue

        if line.startswith("\t"):
            if current_key:
                current_job[current_key] = str(current_job[current_key]) + line.strip()
            continue

        stripped = line.strip()
        if "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        current_key = key.strip()
        current_job[current_key] = value.strip()

    if current_job:
        jobs.append(current_job)

    return jobs


def _normalize_queue_name(queue_name: str) -> str:
    return queue_name.split("@", 1)[0].strip()


def _normalize_job_owner(job_owner: str) -> str:
    return job_owner.split("@", 1)[0].strip()


def filter_active_jobs(
    rows: list[dict[str, Any]],
    *,
    user: str,
    queue_name: str,
    job_name_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Filter ``qstat`` rows down to active jobs relevant for queue throttling."""

    normalized_queue_name = _normalize_queue_name(queue_name)
    return [
        row
        for row in rows
        if row.get("job_state") not in TERMINAL_STATES
        and _normalize_job_owner(str(row.get("Job_Owner", ""))) == user
        and _normalize_queue_name(str(row.get("queue", ""))) == normalized_queue_name
        and (job_name_prefix is None or str(row.get("Job_Name", "")).startswith(job_name_prefix))
    ]


async def count_active_jobs(
    *,
    queue_name: str,
    scope: str = "user_queue",
    job_name_prefix: str | None = None,
    user: str | None = None,
) -> int:
    """Count active Miyabi jobs in the target queue according to the requested scope."""

    if scope not in {"user_queue", "flow_jobs_only"}:
        raise ValueError(f"Unsupported queue limit scope: {scope}")
    if scope == "flow_jobs_only" and not job_name_prefix:
        raise ValueError("job_name_prefix is required when scope='flow_jobs_only'")

    stdout = await run_command("qstat", "-f")
    rows = parse_qstat_listing(stdout)
    resolved_user = user or getpass.getuser()
    prefix = job_name_prefix if scope == "flow_jobs_only" else None
    return len(
        filter_active_jobs(
            rows,
            user=resolved_user,
            queue_name=queue_name,
            job_name_prefix=prefix,
        )
    )


async def wait_for_queue_slot(
    *,
    queue_name: str,
    max_jobs_in_queue: int,
    scope: str = "user_queue",
    job_name_prefix: str | None = None,
    poll_interval_seconds: float = 120.0,
    user: str | None = None,
) -> int:
    """Wait until the Miyabi queue has room for another job."""

    if max_jobs_in_queue < 1:
        raise ValueError("max_jobs_in_queue must be >= 1")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be > 0")

    while True:
        active_count = await count_active_jobs(
            queue_name=queue_name,
            scope=scope,
            job_name_prefix=job_name_prefix,
            user=user,
        )
        if active_count < max_jobs_in_queue:
            return active_count
        await asyncio.sleep(poll_interval_seconds)
