from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SubmitError(RuntimeError):
    """Raised when job submission fails."""


class WaitTimeout(RuntimeError):
    """Raised when waiting for final job status times out."""


class CancelError(RuntimeError):
    """Raised when job cancellation fails."""


async def run_command(*args: str, cwd: Path | None = None) -> str:
    """Run a command asynchronously and return decoded stdout.

    Args:
        *args: Command and arguments to execute.
        cwd: Optional working directory for the command.

    Returns:
        Decoded standard output text.

    Raises:
        RuntimeError: If the command exits with a non-zero return code.
    """

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await proc.communicate()
    out = (out_b or b"").decode(errors="replace")
    err = (err_b or b"").decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)} rc={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}"
        )
    return out


@dataclass(frozen=True)
class SubmitResult:
    """Submission result returned after PJM accepts a batch script.

    Attributes:
        job_id: PJM job id parsed from ``pjsub`` stdout.
        raw_output: Raw, stripped stdout emitted by ``pjsub``.
    """

    job_id: str
    raw_output: str


class FugakuPJMRuntime:
    """Async runtime wrapper for Fugaku PJM scheduler commands.

    The runtime maps to the core PJM commands used on Fugaku: ``pjsub`` for
    submission, ``pjstat`` for status polling, and ``pjdel`` for cancellation.
    Workflow code usually calls
    `qcsc_prefect_executor.fugaku.run.run_fugaku_job` or
    `qcsc_prefect_executor.from_blocks.run_job_from_blocks` instead.
    """

    JOB_ID_RE = re.compile(r"\bJob\s+(\d+)\s+submitted\b", re.IGNORECASE)
    PJSTAT_KEYS = [
        "JOB_ID",
        "JOB_NAME",
        "MD",
        "ST",
        "USER",
        "GROUP",
        "START_DATE",
        "ELAPSE_TIM",
        "ELAPSE_LIM",
        "NODE_REQUIRE",
        "VNODE",
        "CORE",
        "V_MEM",
        "V_POL",
        "E_POL",
        "RANK",
        "LST",
        "EC",
        "PC",
        "SN",
        "PRI",
        "ACCEPT",
        "RSC_GRP",
        "REASON",
    ]

    def __init__(self, *, no_check_directory: bool = False) -> None:
        """Create a Fugaku PJM runtime.

        Args:
            no_check_directory: When true, pass ``--no-check-directory`` to
                ``pjsub`` submissions. This is intentionally opt-in because it
                skips PJM's data-area check for the submit working directory.
        """

        self.no_check_directory = no_check_directory

    def _pjsub_args(self, *args: str) -> tuple[str, ...]:
        command = ["pjsub"]
        if self.no_check_directory:
            command.append("--no-check-directory")
        command.extend(args)
        return tuple(command)

    async def submit(self, script_path: Path, *, cwd: Path | None = None) -> SubmitResult:
        """Submit a PJM script with ``pjsub``.

        Args:
            script_path: Path to the PJM script file.
            cwd: Optional working directory for ``pjsub`` execution.

        Returns:
            Parsed submission result including job id and raw output.

        Raises:
            SubmitError: If submission fails or job id cannot be parsed.
        """

        try:
            stdout = await run_command(*self._pjsub_args(str(script_path)), cwd=cwd)
        except Exception as e:
            raise SubmitError(f"pjsub failed for {script_path}") from e

        out = stdout.strip()
        job_id = self._parse_submit_job_id(out)
        if job_id is None:
            raise SubmitError(f"Failed to parse PJM job id from pjsub output: {out}")
        return SubmitResult(job_id=job_id, raw_output=out)

    async def submit_bulk(
        self,
        script_path: Path,
        bulk_count: int,
        *,
        cwd: Path | None = None,
    ) -> str:
        """Submit a PJM native bulk script and return the parent job id.

        Args:
            script_path: Path to the PJM script file.
            bulk_count: Number of native bulk subjobs. PJM subjob parameters are
                submitted as ``0-{bulk_count - 1}``.
            cwd: Optional working directory for ``pjsub`` execution.

        Returns:
            Parent PJM job id parsed from ``pjsub`` stdout.

        Raises:
            ValueError: If ``bulk_count`` is not positive.
            SubmitError: If submission fails or the parent job id cannot be parsed.
        """

        if int(bulk_count) <= 0:
            raise ValueError("bulk_count must be positive.")

        sparam = f"0-{int(bulk_count) - 1}"
        try:
            stdout = await run_command(
                *self._pjsub_args("--bulk", "--sparam", sparam, str(script_path)),
                cwd=cwd,
            )
        except Exception as e:
            raise SubmitError(
                f"pjsub --bulk failed for {script_path} with --sparam {sparam!r}"
            ) from e

        out = stdout.strip()
        job_id = self._parse_submit_job_id(out)
        if job_id is None:
            raise SubmitError(f"Failed to parse parent PJM job id from pjsub --bulk output: {out}")
        return job_id

    @classmethod
    def _parse_submit_job_id(cls, stdout: str) -> str | None:
        match = cls.JOB_ID_RE.search(stdout)
        if match is None:
            return None
        return match.group(1)

    def _parse_pjstat(self, stdout: str) -> dict[str, Any] | None:
        """Parse a single ``pjstat -v`` row into a dictionary."""

        for row in parse_pjstat_rows(stdout).values():
            return row
        return None

    async def wait_final_status(
        self,
        job_id: str,
        *,
        watch_poll_interval: float = 10.0,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Poll PJM status until a terminal state is reached.

        This method first checks the active job view with ``pjstat -v`` and
        falls back to the historical view with ``pjstat -v -H`` when needed.
        The job is considered terminal when PJM reports ``EXT`` or ``CCL``.

        Args:
            job_id: Target PJM job id.
            watch_poll_interval: Poll interval in seconds.
            timeout_seconds: Optional timeout for waiting terminal status.

        Returns:
            Parsed final ``pjstat`` row.

        Raises:
            WaitTimeout: If timeout is exceeded.
            RuntimeError: If an underlying ``pjstat`` command fails.
        """

        start = asyncio.get_running_loop().time()
        try:
            while True:
                if timeout_seconds is not None:
                    now = asyncio.get_running_loop().time()
                    if now - start > timeout_seconds:
                        raise WaitTimeout(f"timeout waiting for job_id={job_id}")

                stdout = await run_command("pjstat", "-v", job_id)
                if not stdout.strip():
                    stdout = await run_command("pjstat", "-v", "-H", job_id)

                row = self._parse_pjstat(stdout)
                if row and row.get("ST") in {"EXT", "CCL"}:
                    return row

                await asyncio.sleep(watch_poll_interval)

        except asyncio.CancelledError:
            await run_command("pjdel", job_id)
            return {}

    async def cancel(self, job_id: str) -> None:
        """Cancel a PJM job using ``pjdel``.

        Args:
            job_id: Target PJM job id.

        Raises:
            CancelError: If cancellation fails.
        """

        try:
            await run_command("pjdel", job_id)
        except Exception as e:
            raise CancelError(f"pjdel failed for job_id={job_id}") from e


def _parse_fixed_width_pjstat_row(line: str, columns: list[tuple[str, int]]) -> dict[str, str]:
    row: dict[str, str] = {}
    for index, (name, start) in enumerate(columns):
        end = columns[index + 1][1] if index + 1 < len(columns) else None
        row[name] = line[start:end].strip()
    return row


def _parse_split_pjstat_row(line: str) -> dict[str, str]:
    cols = re.split(r"\s+", line.strip())
    return dict(zip(FugakuPJMRuntime.PJSTAT_KEYS, cols))


def parse_pjstat_rows(stdout: str) -> dict[str, dict[str, Any]]:
    """Parse ``pjstat`` table output into rows keyed by PJM job id.

    Fugaku's ``pjstat -v -H`` output is fixed-width. Date fields such as
    ``START_DATE`` and ``ACCEPT`` contain an internal space, so a plain
    whitespace split shifts later columns and can misread ``EC``.
    """

    rows: dict[str, dict[str, Any]] = {}
    header_columns: list[tuple[str, int]] = []

    for line in stdout.splitlines():
        text = line.strip()
        if not text or text.startswith("===="):
            continue
        if text.startswith("JOB_ID"):
            header_columns = [
                (match.group(0), match.start()) for match in re.finditer(r"\S+", line)
            ]
            continue

        row = (
            _parse_fixed_width_pjstat_row(line, header_columns)
            if header_columns
            else _parse_split_pjstat_row(line)
        )
        job_id = str(row.get("JOB_ID", "")).strip()
        if job_id:
            rows[job_id] = row

    return rows
