from __future__ import annotations

import asyncio
from pathlib import Path

from hpc_prefect_adapters.fugaku import runtime as runtime_mod


def test_submit_parses_job_id(tmp_path: Path, monkeypatch):
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append((args, cwd))
        return "Job 43607196 submitted."

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.FugakuPJMRuntime()

    result = asyncio.run(rt.submit(tmp_path / "batch.pjm", cwd=tmp_path))

    assert result.job_id == "43607196"
    assert calls == [(("pjsub", str(tmp_path / "batch.pjm")), tmp_path)]


def test_wait_final_status_with_pjstat_fallback(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append(args)
        if args == ("pjstat", "-v", "43607196"):
            return ""
        return "43607196 test NM EXT user group 2026-02-09T00:00:00 00:00:10 00:05:00 1 1 48 0M N N 1 - 0 - - - - regular-c -"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.FugakuPJMRuntime()

    status = asyncio.run(rt.wait_final_status("43607196", watch_poll_interval=0.01, timeout_seconds=3))

    assert status["JOB_ID"] == "43607196"
    assert status["ST"] == "EXT"
    assert status["EC"] == "0"
    assert calls == [("pjstat", "-v", "43607196"), ("pjstat", "-H", "-v", "43607196")]


def test_cancel_invokes_pjdel(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.FugakuPJMRuntime()

    asyncio.run(rt.cancel("43607196"))

    assert calls == [("pjdel", "43607196")]

