from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


class SubmitError(RuntimeError):
    """Raised when job submission fails."""


class WaitTimeout(RuntimeError):
    """Raised when waiting for final job status times out."""


class CancelError(RuntimeError):
    """Raised when job cancellation fails."""


async def run_command(*args: str, cwd: Path | None = None) -> str:
    """
    Minimal async command runner (stdout returned; stderr included on error).
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


class MiyabiPBSRuntime:
    """
    Miyabi PBS runtime:
      - submit: qsub
      - wait:   qstat -fH (finished jobs list)
      - cancel: qdel

    This mirrors the behavior of your existing prefect-miyabi executor.py.
    """
    QSTAT_OUT: ClassVar[re.Pattern] = re.compile(r"Job Id: (\d+\.\w+)\n((?:[ \t]+.*(?:\n|$))*)")

    async def submit(self, script_path: Path, *, cwd: Path | None = None) -> SubmitResult:
        """Submit a PBS script with ``qsub``.

        Args:
            script_path: Path to the PBS script file.
            cwd: Optional working directory for ``qsub`` execution.

        Returns:
            Parsed submission result including job id and raw output.

        Raises:
            SubmitError: If submission fails or job id cannot be parsed.
        """

        try:
            stdout = await run_command("qsub", str(script_path), cwd=cwd)
        except Exception as e:
            raise SubmitError(f"qsub failed for {script_path}") from e

        out = stdout.strip()
        if not out:
            raise SubmitError("qsub returned empty stdout; cannot parse job id.")
        job_id = out.split()[0]
        return SubmitResult(job_id=job_id, raw_output=out)

    async def wait_final_status(
        self,
        job_id: str,
        *,
        watch_poll_interval: float = 10.0,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """
        Wait until the job shows up in finished-job list (-fH) and return parsed dict.
        This is a near-copy of the existing _wait() logic.
        """
        start = asyncio.get_running_loop().time()
        try:
            while True:
                if timeout_seconds is not None:
                    now = asyncio.get_running_loop().time()
                    if now - start > timeout_seconds:
                        raise WaitTimeout(f"timeout waiting for job_id={job_id}")

                stdout = await run_command("qstat", "-fH", job_id)
                match = re.search(self.QSTAT_OUT, stdout)
                if match:
                    current_key = ""
                    out: dict[str, Any] = {}

                    for line in match.group(2).splitlines():
                        if len(line) == 0:
                            continue

                        # continuation lines start with tab in this qstat output
                        if line.startswith("\t"):
                            # append continuation to the previous key
                            out[current_key] += line.strip()
                        else:
                            key, val = line.split("=", 1)
                            current_key = key.strip()
                            out[current_key] = val.strip()

                    return out

                await asyncio.sleep(watch_poll_interval)

        except asyncio.CancelledError:
            # keep same behavior: cancel => qdel
            await run_command("qdel", job_id)
            return {}

    async def cancel(self, job_id: str) -> None:
        """Cancel a PBS job using ``qdel``.

        Args:
            job_id: Target PBS job id.

        Raises:
            CancelError: If cancellation fails.
        """

        try:
            await run_command("qdel", job_id)
        except Exception as e:
            raise CancelError(f"qdel failed for job_id={job_id}") from e
