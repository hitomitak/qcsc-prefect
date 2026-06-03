from __future__ import annotations

import asyncio
from pathlib import Path

from qcsc_prefect_adapters.fugaku import runtime as runtime_mod


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
        return (
            "43607196 test NM EXT user group 2026-02-09T00:00:00 00:00:10 "
            "00:05:00 1 1 48 0M N N 1 - 0 - - - - regular-c -"
        )

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.FugakuPJMRuntime()

    status = asyncio.run(
        rt.wait_final_status("43607196", watch_poll_interval=0.01, timeout_seconds=3)
    )

    assert status["JOB_ID"] == "43607196"
    assert status["ST"] == "EXT"
    assert status["EC"] == "0"
    assert calls == [("pjstat", "-v", "43607196"), ("pjstat", "-v", "-H", "43607196")]


def test_parse_pjstat_rows_handles_history_verbose_dates_and_exit_codes():
    stdout = (
        "JOB_ID     JOB_NAME   MD ST  USER     GROUP    START_DATE      "
        "ELAPSE_TIM ELAPSE_LIM            NODE_REQUIRE    VNODE  CORE "
        "V_MEM        V_POL E_POL RANK      LST EC  PC  SN PRI ACCEPT         "
        "RSC_GRP  REASON\n"
        "49047829   lucj-qpy-b NM EXT u13450   ra010014 06/01 15:03:44  "
        "0000:08:27 0000:15:00            1               -      -    "
        "-            -     -     bychip    RNO 0   0   0  127 "
        "06/01 15:03:14 small    -\n"
        "49047939   lucj-qpy-b NM EXT u13450   ra010014 06/01 15:22:46  "
        "0000:15:02 0000:15:00            1               -      -    "
        "-            -     -     bychip    RNO 0   11  24 127 "
        "06/01 15:22:16 small    ELAPSE LIMIT EXC\n"
    )

    rows = runtime_mod.parse_pjstat_rows(stdout)

    assert rows["49047829"]["ST"] == "EXT"
    assert rows["49047829"]["START_DATE"] == "06/01 15:03:44"
    assert rows["49047829"]["EC"] == "0"
    assert rows["49047829"]["ACCEPT"] == "06/01 15:03:14"
    assert rows["49047829"]["RSC_GRP"] == "small"
    assert rows["49047939"]["EC"] == "0"
    assert rows["49047939"]["PC"] == "11"
    assert rows["49047939"]["REASON"] == "ELAPSE LIMIT EXC"


def test_cancel_invokes_pjdel(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.FugakuPJMRuntime()

    asyncio.run(rt.cancel("43607196"))

    assert calls == [("pjdel", "43607196")]
