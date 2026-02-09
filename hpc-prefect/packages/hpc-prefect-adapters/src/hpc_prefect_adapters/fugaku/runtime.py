from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SubmitError(RuntimeError): ...
class WaitTimeout(RuntimeError): ...
class CancelError(RuntimeError): ...


async def run_command(*args: str, cwd: Path | None = None) -> str:
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
    job_id: str
    raw_output: str


class FugakuPJMRuntime:
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
        try:
            await run_command("pjdel", job_id)
        except Exception as e:
            raise CancelError(f"pjdel failed for job_id={job_id}") from e

