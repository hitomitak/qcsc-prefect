from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from qcsc_prefect_adapters.base.subprocess import (
    DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS,
    validate_command_timeout,
)
from qcsc_prefect_core.queue import QueueCapacity

_SQUEUE_CAPACITY_FORMAT = "%i|%T"


@dataclass(frozen=True)
class SlurmQueueRow:
    """One active allocation or expanded array element reported by ``squeue``."""

    job_id: str
    state: str


def parse_squeue_active_rows(stdout: str) -> list[SlurmQueueRow]:
    """Parse the exact two-column output requested by the Slurm queue probe.

    Malformed non-empty rows are rejected so callers can fail closed instead
    of computing capacity from a partial or unexpected scheduler response.
    """

    rows: list[SlurmQueueRow] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 2 or not all(field.strip() for field in fields):
            raise ValueError(f"Malformed squeue capacity row at line {line_number}.")
        rows.append(
            SlurmQueueRow(
                job_id=fields[0].strip(),
                state=fields[1].strip(),
            )
        )
    return rows


def estimate_capacity_from_squeue(
    stdout: str,
    *,
    max_active_jobs: int,
) -> QueueCapacity:
    """Apply an explicit workflow ceiling to active jobs reported by Slurm."""

    configured_ceiling = max(0, int(max_active_jobs))
    current_active_jobs = len(parse_squeue_active_rows(stdout))
    return QueueCapacity(
        max_active_jobs=configured_ceiling,
        current_active_jobs=current_active_jobs,
        available_slots=max(0, configured_ceiling - current_active_jobs),
        raw_output=stdout,
    )


def _run_squeue(
    *,
    user: str,
    account: str | None,
    partition: str | None,
    timeout_seconds: float | None,
) -> str:
    args = [
        "squeue",
        "--noheader",
        "--array",
        "--states=all",
        f"--user={user}",
    ]
    if account:
        args.append(f"--account={account}")
    if partition:
        args.append(f"--partition={partition}")
    args.append(f"--format={_SQUEUE_CAPACITY_FORMAT}")

    completed = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        timeout=validate_command_timeout(timeout_seconds),
    )
    return completed.stdout


def _error_text(exc: BaseException) -> str:
    parts: list[str] = []
    for value in (getattr(exc, "stdout", None), getattr(exc, "stderr", None)):
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        if value is not None and str(value).strip():
            parts.append(str(value).strip())
    return "\n".join(parts) or str(exc)


@dataclass(frozen=True)
class SlurmQueueProbe:
    """Count active Slurm jobs inside one explicit user/account/partition scope.

    ``squeue`` is used only to observe current jobs. ``max_active_jobs`` is a
    caller-configured workflow ceiling, not a quota inferred from Slurm.
    ``QueueAwareSubmitGate`` applies the separate safety margin and per-refill
    limit to the returned capacity.
    """

    max_active_jobs: int
    user: str
    account: str | None = None
    partition: str | None = None
    scheduler_command_timeout_seconds: float | None = DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        normalized_user = str(self.user).strip()
        if not normalized_user:
            raise ValueError("user is required for the Slurm queue probe.")
        object.__setattr__(self, "user", normalized_user)
        object.__setattr__(self, "account", _optional_text(self.account))
        object.__setattr__(self, "partition", _optional_text(self.partition))
        validate_command_timeout(self.scheduler_command_timeout_seconds)

    def get_capacity(self) -> QueueCapacity:
        """Return capacity or a zero-capacity result for any probe failure."""

        try:
            stdout = _run_squeue(
                user=self.user,
                account=self.account,
                partition=self.partition,
                timeout_seconds=self.scheduler_command_timeout_seconds,
            )
            return estimate_capacity_from_squeue(
                stdout,
                max_active_jobs=self.max_active_jobs,
            )
        except Exception as exc:
            return self._zero_capacity(raw_output=_error_text(exc))

    def _zero_capacity(self, *, raw_output: str | None) -> QueueCapacity:
        configured_ceiling = max(0, int(self.max_active_jobs))
        return QueueCapacity(
            max_active_jobs=configured_ceiling,
            current_active_jobs=configured_ceiling,
            available_slots=0,
            raw_output=raw_output,
        )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
