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
        raise RuntimeError(f"Command failed: {' '.join(args)} rc={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}")
    return out


@dataclass(frozen=True)
class SubmitResult:
    """Submission result payload returned by runtime ``submit`` methods."""

    job_id: str
    raw_output: str


class FugakuPJMRuntime:
    """Minimal async runtime for Fugaku PJM commands."""

    JOB_ID_RE = re.compile(r"Job\s+(\d+)\s+submitted\.?")
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
            stdout = await run_command("pjsub", str(script_path), cwd=cwd)
        except Exception as e:
            raise SubmitError(f"pjsub failed for {script_path}") from e

        out = stdout.strip()
        m = self.JOB_ID_RE.search(out)
        if not m:
            raise SubmitError(f"Failed to parse PJM job id from pjsub output: {out}")
        return SubmitResult(job_id=m.group(1), raw_output=out)

    def _parse_pjstat(self, stdout: str) -> dict[str, Any] | None:
        for line in stdout.splitlines():
            s = line.strip()
            if not s or s.startswith("JOB_ID") or s.startswith("===="):
                continue
            cols = re.split(r"\s+", s)
            row = dict(zip(self.PJSTAT_KEYS, cols))
            if row.get("JOB_ID"):
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

        Args:
            job_id: Target PJM job id.
            watch_poll_interval: Poll interval in seconds.
            timeout_seconds: Optional timeout for waiting terminal status.

        Returns:
            Parsed final ``pjstat`` row.

        Raises:
            WaitTimeout: If timeout is exceeded.
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
                    stdout = await run_command("pjstat", "-H", "-v", job_id)

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
