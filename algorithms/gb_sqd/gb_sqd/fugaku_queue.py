"""Helpers for queue-aware GB-SQD submission on Fugaku."""

from __future__ import annotations

import asyncio
import getpass
import re
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parents[3]
if (_project_root / "packages").exists():
    sys.path.insert(0, str(_project_root / "packages" / "qcsc-prefect-adapters" / "src"))

from qcsc_prefect_adapters.fugaku.runtime import FugakuPJMRuntime, run_command


TERMINAL_STATES = {"EXT", "CCL"}


def parse_pjstat_listing(stdout: str) -> list[dict[str, Any]]:
    """Parse `pjstat` listing output into row dictionaries."""

    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s or s.startswith("JOB_ID") or s.startswith("===="):
            continue
        cols = re.split(r"\s+", s)
        row = dict(zip(FugakuPJMRuntime.PJSTAT_KEYS, cols))
        if row.get("JOB_ID"):
            rows.append(row)
    return rows


def filter_active_jobs(
    rows: list[dict[str, Any]],
    *,
    user: str,
    resource_group: str,
    job_name_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Filter `pjstat` rows down to active jobs relevant for queue throttling."""

    return [
        row
        for row in rows
        if row.get("ST") not in TERMINAL_STATES
        and row.get("USER") == user
        and row.get("RSC_GRP") == resource_group
        and (job_name_prefix is None or row.get("JOB_NAME", "").startswith(job_name_prefix))
    ]


async def count_active_jobs(
    *,
    resource_group: str,
    scope: str = "user_queue",
    job_name_prefix: str | None = None,
    user: str | None = None,
) -> int:
    """Count active jobs in the target Fugaku queue according to the requested scope."""

    if scope not in {"user_queue", "flow_jobs_only"}:
        raise ValueError(f"Unsupported queue limit scope: {scope}")
    if scope == "flow_jobs_only" and not job_name_prefix:
        raise ValueError("job_name_prefix is required when scope='flow_jobs_only'")

    stdout = await run_command("pjstat")
    rows = parse_pjstat_listing(stdout)
    resolved_user = user or getpass.getuser()
    prefix = job_name_prefix if scope == "flow_jobs_only" else None
    return len(
        filter_active_jobs(
            rows,
            user=resolved_user,
            resource_group=resource_group,
            job_name_prefix=prefix,
        )
    )


async def wait_for_queue_slot(
    *,
    resource_group: str,
    max_jobs_in_queue: int,
    scope: str = "user_queue",
    job_name_prefix: str | None = None,
    poll_interval_seconds: float = 120.0,
    user: str | None = None,
) -> int:
    """Wait until the Fugaku queue has room for another job."""

    if max_jobs_in_queue < 1:
        raise ValueError("max_jobs_in_queue must be >= 1")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be > 0")

    while True:
        active_count = await count_active_jobs(
            resource_group=resource_group,
            scope=scope,
            job_name_prefix=job_name_prefix,
            user=user,
        )
        if active_count < max_jobs_in_queue:
            return active_count
        await asyncio.sleep(poll_interval_seconds)
