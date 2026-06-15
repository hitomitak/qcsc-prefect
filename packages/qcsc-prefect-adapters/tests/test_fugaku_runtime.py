from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
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


def test_submit_with_no_check_directory_adds_pjsub_option(tmp_path: Path, monkeypatch):
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append((args, cwd))
        return "Job 43607196 submitted."

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.FugakuPJMRuntime(no_check_directory=True)
    script_path = tmp_path / "batch.pjm"

    result = asyncio.run(rt.submit(script_path, cwd=tmp_path))

    assert result.job_id == "43607196"
    assert calls == [(("pjsub", "--no-check-directory", str(script_path)), tmp_path)]


def test_submit_bulk_invokes_pjsub_bulk_with_sparam_and_cwd(tmp_path: Path, monkeypatch):
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append((args, cwd))
        return "Job 12345 submitted."

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.FugakuPJMRuntime()
    script_path = tmp_path / "bulk.pjm"

    parent_job_id = asyncio.run(rt.submit_bulk(script_path, bulk_count=5, cwd=tmp_path))

    assert parent_job_id == "12345"
    assert calls == [
        (
            ("pjsub", "--bulk", "--sparam", "0-4", str(script_path)),
            tmp_path,
        )
    ]


def test_submit_bulk_with_no_check_directory_adds_pjsub_option(tmp_path: Path, monkeypatch):
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append((args, cwd))
        return "Job 12345 submitted."

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.FugakuPJMRuntime(no_check_directory=True)
    script_path = tmp_path / "bulk.pjm"

    parent_job_id = asyncio.run(rt.submit_bulk(script_path, bulk_count=5, cwd=tmp_path))

    assert parent_job_id == "12345"
    assert calls == [
        (
            ("pjsub", "--no-check-directory", "--bulk", "--sparam", "0-4", str(script_path)),
            tmp_path,
        )
    ]


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("Job 43607196 submitted.", "43607196"),
        ("[INFO] PJM accepted request\nJob 49047829 submitted.\n", "49047829"),
        ("job 24680 submitted", "24680"),
    ],
)
def test_parse_submit_job_id_handles_pjsub_outputs(stdout: str, expected: str):
    assert runtime_mod.FugakuPJMRuntime._parse_submit_job_id(stdout) == expected


def test_submit_bulk_invalid_output_raises_submit_error(tmp_path: Path, monkeypatch):
    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        return "PJM accepted request but no job id was printed"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.FugakuPJMRuntime()

    with pytest.raises(runtime_mod.SubmitError, match="parent PJM job id"):
        asyncio.run(rt.submit_bulk(tmp_path / "bulk.pjm", bulk_count=2))


def test_submit_bulk_rejects_non_positive_bulk_count(tmp_path: Path, monkeypatch):
    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        raise AssertionError("pjsub must not be called for invalid bulk_count")

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.FugakuPJMRuntime()

    with pytest.raises(ValueError, match="bulk_count must be positive"):
        asyncio.run(rt.submit_bulk(tmp_path / "bulk.pjm", bulk_count=0))


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
