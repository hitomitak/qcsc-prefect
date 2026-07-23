from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS = 300.0


class SchedulerCommandTimeout(TimeoutError):
    """Raised when a scheduler CLI command does not return before its deadline."""


class SchedulerCommandError(RuntimeError):
    """Raised when a scheduler CLI command exits with a non-zero status."""

    def __init__(
        self,
        *,
        args: tuple[str, ...],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.command_args = args
        self.returncode = int(returncode)
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"Command failed: {shlex.join(args)} rc={returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )


def validate_command_timeout(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    timeout = float(timeout_seconds)
    if timeout <= 0:
        raise ValueError("scheduler command timeout_seconds must be greater than 0")
    return timeout


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.communicate()


async def run_scheduler_command(
    *args: str,
    cwd: Path | None = None,
    timeout_seconds: float | None = DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS,
) -> str:
    """Run one scheduler command with a bounded wait and captured output."""

    if not args:
        raise ValueError("scheduler command must not be empty")
    timeout = validate_command_timeout(timeout_seconds)
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except TimeoutError:
        await _stop_process(process)
        command = shlex.join(args)
        raise SchedulerCommandTimeout(
            f"Scheduler command timed out after {timeout:g} seconds: {command}"
        ) from None
    except asyncio.CancelledError:
        await _stop_process(process)
        raise

    stdout = (stdout_bytes or b"").decode(errors="replace")
    stderr = (stderr_bytes or b"").decode(errors="replace")
    if process.returncode != 0:
        raise SchedulerCommandError(
            args=tuple(args),
            returncode=int(process.returncode),
            stdout=stdout,
            stderr=stderr,
        )
    return stdout
