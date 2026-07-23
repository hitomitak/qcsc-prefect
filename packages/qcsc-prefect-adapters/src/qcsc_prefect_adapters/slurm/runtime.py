from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from qcsc_prefect_adapters.base.subprocess import (
    DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS,
    SchedulerCommandError,
    SchedulerCommandTimeout,
    run_scheduler_command,
)


class SubmitError(RuntimeError):
    """Raised when job submission fails."""


class SubmitOutcomeUnknownError(SubmitError):
    """Raised when submission may have succeeded but no job id was received."""

    submit_outcome_unknown = True


class SubmitRejectedError(SubmitError):
    """Raised when ``sbatch`` proves that it did not accept the job."""

    submission_definitely_rejected = True


class WaitTimeout(RuntimeError):
    """Raised when waiting for final job status times out."""


class CancelError(RuntimeError):
    """Raised when job cancellation fails."""


async def run_command(
    *args: str,
    cwd: Path | None = None,
    timeout_seconds: float | None = DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS,
) -> str:
    """Run a Slurm command asynchronously and return decoded stdout.

    Args:
        *args: Command and arguments passed to
            `asyncio.create_subprocess_exec`.
        cwd: Optional working directory for the subprocess.

    Returns:
        Standard output decoded with replacement for invalid bytes.

    Raises:
        RuntimeError: If the command exits with a non-zero return code. The
            error message includes decoded stdout and stderr for diagnostics.
    """

    return await run_scheduler_command(
        *args,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )


@dataclass(frozen=True)
class SubmitResult:
    """Submission result returned after a scheduler accepts a batch script.

    Attributes:
        job_id: Scheduler job id parsed from the submission command output.
        raw_output: Raw, stripped stdout emitted by the submission command.
    """

    job_id: str
    raw_output: str


@dataclass(frozen=True)
class SlurmJobCandidate:
    """One allocation-level Slurm identity-search result."""

    job_id: str
    job_name: str
    comment: str
    user: str
    account: str
    partition: str
    submit_time: datetime | None
    state: str
    source: Literal["squeue", "sacct"]

    @property
    def is_terminal(self) -> bool:
        return _is_terminal_state(self.state)


_PLAIN_JOB_ID_RE = re.compile(r"^[0-9]+$")
_AMBIGUOUS_SBATCH_ERROR_PATTERNS = (
    "connection",
    "socket timed out",
    "timed out",
    "unable to contact",
    "communication",
    "slurmctld",
)
_SQUEUE_IDENTITY_FORMAT = "%i|%j|%k|%u|%a|%P|%V|%T"
_SACCT_IDENTITY_FORMAT = "JobIDRaw,JobName,Comment,User,Account,Partition,Submit,State"


def _parse_slurm_datetime(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized or normalized.lower() in {"unknown", "n/a", "none"}:
        return None
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        local_timezone = datetime.now().astimezone().tzinfo or timezone.utc
        parsed = parsed.replace(tzinfo=local_timezone)
    return parsed


def _parse_identity_rows(
    stdout: str,
    *,
    source: Literal["squeue", "sacct"],
) -> list[SlurmJobCandidate]:
    candidates: list[SlurmJobCandidate] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) < 8:
            continue
        job_id = fields[0].strip()
        # Resumable single-job submission never creates an array or job step.
        # Excluding every non-plain allocation id prevents accidental attach to
        # ``.batch``, ``.extern``, array, or heterogeneous rows.
        if not _PLAIN_JOB_ID_RE.fullmatch(job_id):
            continue
        candidates.append(
            SlurmJobCandidate(
                job_id=job_id,
                job_name=fields[1].strip(),
                comment=fields[2].strip(),
                user=fields[3].strip(),
                account=fields[4].strip(),
                partition=fields[5].strip(),
                submit_time=_parse_slurm_datetime(fields[6]),
                state=fields[7].strip(),
                source=source,
            )
        )
    return candidates


def parse_squeue_identity_rows(stdout: str) -> list[SlurmJobCandidate]:
    """Parse allocation rows emitted by the identity ``squeue`` query."""

    return _parse_identity_rows(stdout, source="squeue")


def parse_sacct_identity_rows(stdout: str) -> list[SlurmJobCandidate]:
    """Parse allocation rows emitted by the identity ``sacct`` query."""

    return _parse_identity_rows(stdout, source="sacct")


def _deduplicate_identity_candidates(
    candidates: list[SlurmJobCandidate],
) -> list[SlurmJobCandidate]:
    deduplicated: dict[tuple[str, datetime | None], SlurmJobCandidate] = {}
    for candidate in candidates:
        key = (candidate.job_id, candidate.submit_time)
        existing = deduplicated.get(key)
        if existing is None or (existing.source == "sacct" and candidate.source == "squeue"):
            deduplicated[key] = candidate
    return list(deduplicated.values())


def _sbatch_error_is_ambiguous(exc: SchedulerCommandError) -> bool:
    message = f"{exc.stdout}\n{exc.stderr}".lower()
    return any(pattern in message for pattern in _AMBIGUOUS_SBATCH_ERROR_PATTERNS)


def _is_terminal_state(state: str) -> bool:
    normalized = state.strip().split()[0].rstrip("+")
    return normalized in {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }


class SlurmRuntime:
    """Async runtime wrapper for Slurm scheduler commands.

    This class is the low-level boundary around ``sbatch``, ``sacct``, and
    ``scancel``. Workflow code usually calls
    `qcsc_prefect_executor.slurm.run.run_slurm_job` or
    `qcsc_prefect_executor.from_blocks.run_job_from_blocks` instead.
    """

    async def submit(
        self,
        script_path: Path,
        *,
        cwd: Path | None = None,
        timeout_seconds: float | None = DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS,
    ) -> SubmitResult:
        """Submit a Slurm batch script with ``sbatch --parsable``.

        Args:
            script_path: Path to the generated Slurm script.
            cwd: Optional working directory for ``sbatch``.

        Returns:
            Parsed submission payload containing the Slurm job id.

        Raises:
            SubmitError: If ``sbatch`` fails or the job id cannot be parsed.
        """

        try:
            stdout = await run_command(
                "sbatch",
                "--parsable",
                str(script_path),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            )
        except SchedulerCommandTimeout as e:
            raise SubmitOutcomeUnknownError(
                "sbatch timed out before a job id was received; submission outcome is "
                f"unknown for {script_path}. Do not retry automatically."
            ) from e
        except SchedulerCommandError as e:
            if _sbatch_error_is_ambiguous(e):
                raise SubmitOutcomeUnknownError(
                    "sbatch lost contact with the scheduler before acceptance could be "
                    f"proven for {script_path}. Do not retry automatically."
                ) from e
            raise SubmitRejectedError(f"sbatch rejected {script_path}") from e
        except Exception as e:
            raise SubmitOutcomeUnknownError(
                f"sbatch outcome is unknown for {script_path}. Do not retry automatically."
            ) from e

        out = stdout.strip()
        if not out:
            raise SubmitOutcomeUnknownError(
                "sbatch returned empty stdout after a zero exit status; submission "
                "outcome is unknown. Do not retry automatically."
            )
        job_id = out.split(";", 1)[0].strip()
        if not _PLAIN_JOB_ID_RE.fullmatch(job_id):
            raise SubmitOutcomeUnknownError(
                "sbatch returned an unrecognized job id after a zero exit status; "
                "submission outcome is unknown. Do not retry automatically."
            )
        return SubmitResult(job_id=job_id, raw_output=out)

    async def find_jobs_by_identity(
        self,
        *,
        job_name: str,
        user: str,
        search_start: datetime,
        timeout_seconds: float | None = DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS,
    ) -> list[SlurmJobCandidate]:
        """Find active and historical allocation rows for one deterministic name.

        The caller must still validate the full comment, account, partition,
        user, and submission window before attaching.
        """

        normalized_name = str(job_name).strip()
        normalized_user = str(user).strip()
        if not normalized_name:
            raise ValueError("job_name is required for Slurm identity search.")
        if not normalized_user:
            raise ValueError("user is required for Slurm identity search.")
        if search_start.tzinfo is None:
            raise ValueError("search_start must be timezone-aware.")

        active_stdout = await run_command(
            "squeue",
            "--noheader",
            f"--user={normalized_user}",
            f"--name={normalized_name}",
            f"--format={_SQUEUE_IDENTITY_FORMAT}",
            timeout_seconds=timeout_seconds,
        )
        local_start = search_start.astimezone()
        history_stdout = await run_command(
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--duplicates",
            f"--user={normalized_user}",
            f"--name={normalized_name}",
            f"--starttime={local_start.strftime('%Y-%m-%dT%H:%M:%S')}",
            f"--format={_SACCT_IDENTITY_FORMAT}",
            timeout_seconds=timeout_seconds,
        )
        return _deduplicate_identity_candidates(
            [
                *parse_squeue_identity_rows(active_stdout),
                *parse_sacct_identity_rows(history_stdout),
            ]
        )

    async def wait_final_status(
        self,
        job_id: str,
        *,
        watch_poll_interval: float = 10.0,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Poll ``sacct`` until the job reaches a terminal state.

        The returned dictionary is normalized to the fields requested from
        ``sacct``: ``JobID``, ``State``, ``ExitCode``, ``Elapsed``,
        ``AllocCPUS``, and ``NodeList``.

        Args:
            job_id: Slurm job id to watch.
            watch_poll_interval: Seconds to wait between ``sacct`` calls.
            timeout_seconds: Optional maximum wait time.

        Returns:
            Parsed terminal ``sacct`` row for the batch job.

        Raises:
            WaitTimeout: If ``timeout_seconds`` elapses before a terminal state.
            RuntimeError: If an underlying ``sacct`` command fails.
        """

        start = asyncio.get_running_loop().time()
        try:
            while True:
                command_timeout = DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS
                wait_deadline_limits_command = False
                if timeout_seconds is not None:
                    now = asyncio.get_running_loop().time()
                    remaining = timeout_seconds - (now - start)
                    if remaining <= 0:
                        raise WaitTimeout(f"timeout waiting for job_id={job_id}")
                    if remaining <= command_timeout:
                        command_timeout = remaining
                        wait_deadline_limits_command = True

                try:
                    stdout = await run_command(
                        "sacct",
                        "-j",
                        job_id,
                        "--format=JobID,State,ExitCode,Elapsed,AllocCPUS,NodeList",
                        "--parsable2",
                        "--noheader",
                        timeout_seconds=command_timeout,
                    )
                except SchedulerCommandTimeout as exc:
                    if wait_deadline_limits_command:
                        raise WaitTimeout(f"timeout waiting for job_id={job_id}") from exc
                    raise

                for line in stdout.splitlines():
                    fields = line.split("|")
                    if len(fields) < 6:
                        continue
                    job_id_field = fields[0].strip()
                    if not job_id_field or job_id_field != job_id:
                        continue
                    state = fields[1].strip()
                    out = {
                        "JobID": job_id_field,
                        "State": state,
                        "ExitCode": fields[2].strip(),
                        "Elapsed": fields[3].strip(),
                        "AllocCPUS": fields[4].strip(),
                        "NodeList": fields[5].strip(),
                    }
                    if _is_terminal_state(state):
                        return out

                await asyncio.sleep(watch_poll_interval)

        except asyncio.CancelledError:
            await run_command("scancel", job_id)
            return {}

    async def cancel(self, job_id: str) -> None:
        """Cancel a Slurm job using ``scancel``.

        Args:
            job_id: Target Slurm job id.

        Raises:
            CancelError: If ``scancel`` exits with an error.
        """

        try:
            await run_command("scancel", job_id)
        except Exception as e:
            raise CancelError(f"scancel failed for job_id={job_id}") from e
