from __future__ import annotations

import asyncio
from pathlib import Path

from hpc_prefect_adapters.miyabi import runtime as runtime_mod


def test_submit_parses_job_id(tmp_path: Path, monkeypatch):
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append((args, cwd))
        return "12345.miyabi\n"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.MiyabiPBSRuntime()

    result = asyncio.run(rt.submit(tmp_path / "job.pbs", cwd=tmp_path))

    assert result.job_id == "12345.miyabi"
    assert result.raw_output == "12345.miyabi"
    assert calls == [(("qsub", str(tmp_path / "job.pbs")), tmp_path)]


def test_wait_final_status_parses_qstat_output(monkeypatch):
    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        return (
            "Job Id: 12345.miyabi\n"
            "    Job_Name = test-job\n"
            "    queue = normal\n"
            "    Exit_status = 0\n"
            "    resources_used.mem = 1048576kb\n"
            "    Variable_List = A=B,\n"
            "\tC=D\n"
        )

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.MiyabiPBSRuntime()

    status = asyncio.run(rt.wait_final_status("12345.miyabi", watch_poll_interval=0.01, timeout_seconds=3))

    assert status["Job_Name"] == "test-job"
    assert status["queue"] == "normal"
    assert status["Exit_status"] == "0"
    assert status["resources_used.mem"] == "1048576kb"
    assert status["Variable_List"] == "A=B,C=D"


def test_cancel_invokes_qdel(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.MiyabiPBSRuntime()

    asyncio.run(rt.cancel("12345.miyabi"))

    assert calls == [("qdel", "12345.miyabi")]
