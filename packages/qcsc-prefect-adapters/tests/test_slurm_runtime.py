from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from qcsc_prefect_adapters.slurm import runtime as runtime_mod


def test_submit_parses_job_id(tmp_path: Path, monkeypatch):
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None, **_kwargs: object) -> str:
        calls.append((args, cwd))
        return "12345;cluster-a\n"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.SlurmRuntime()

    result = asyncio.run(rt.submit(tmp_path / "job.slurm", cwd=tmp_path))

    assert result.job_id == "12345"
    assert result.raw_output == "12345;cluster-a"
    assert calls == [(("sbatch", "--parsable", str(tmp_path / "job.slurm")), tmp_path)]


def test_wait_final_status_parses_sacct_output(monkeypatch):
    async def fake_run_command(*args: str, cwd: Path | None = None, **_kwargs: object) -> str:
        return (
            "12345|COMPLETED|0:0|00:00:12|32|node001\n"
            "12345.batch|COMPLETED|0:0|00:00:12|32|node001\n"
        )

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.SlurmRuntime()

    status = asyncio.run(rt.wait_final_status("12345", watch_poll_interval=0.01, timeout_seconds=3))

    assert status["JobID"] == "12345"
    assert status["State"] == "COMPLETED"
    assert status["ExitCode"] == "0:0"
    assert status["Elapsed"] == "00:00:12"
    assert status["AllocCPUS"] == "32"
    assert status["NodeList"] == "node001"


def test_submit_timeout_reports_unknown_outcome(tmp_path: Path, monkeypatch):
    async def fake_run_command(
        *args: str,
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        raise runtime_mod.SchedulerCommandTimeout("sbatch timed out")

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)

    with pytest.raises(runtime_mod.SubmitOutcomeUnknownError, match="outcome is unknown"):
        asyncio.run(runtime_mod.SlurmRuntime().submit(tmp_path / "job.slurm"))


def test_submit_unparseable_success_reports_unknown_outcome(tmp_path: Path, monkeypatch):
    async def fake_run_command(*_args: str, **_kwargs: object) -> str:
        return "accepted-without-a-numeric-id"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)

    with pytest.raises(runtime_mod.SubmitOutcomeUnknownError, match="unrecognized job id"):
        asyncio.run(runtime_mod.SlurmRuntime().submit(tmp_path / "job.slurm"))


@pytest.mark.parametrize(
    ("stderr", "expected_error"),
    [
        ("Unable to contact slurm controller", runtime_mod.SubmitOutcomeUnknownError),
        ("Batch job submission failed: Invalid account", runtime_mod.SubmitRejectedError),
    ],
)
def test_submit_classifies_scheduler_command_errors(
    tmp_path: Path,
    monkeypatch,
    stderr: str,
    expected_error: type[Exception],
):
    async def fake_run_command(*args: str, **_kwargs: object) -> str:
        raise runtime_mod.SchedulerCommandError(
            args=tuple(args),
            returncode=1,
            stdout="",
            stderr=stderr,
        )

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)

    with pytest.raises(expected_error):
        asyncio.run(runtime_mod.SlurmRuntime().submit(tmp_path / "job.slurm"))


def test_find_jobs_by_identity_queries_active_and_history_and_skips_non_allocations(
    monkeypatch,
):
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, **_kwargs: object) -> str:
        calls.append(args)
        if args[0] == "squeue":
            return (
                "123|qcsc-job|identity-comment|alice|acct|compute|"
                "2026-07-23T01:02:03+00:00|RUNNING\n"
            )
        return (
            "123|qcsc-job|identity-comment|alice|acct|compute|"
            "2026-07-23T01:02:03+00:00|PENDING\n"
            "123.batch|batch|identity-comment|alice|acct|compute|"
            "2026-07-23T01:02:03+00:00|COMPLETED\n"
            "124_1|array|identity-comment|alice|acct|compute|"
            "2026-07-23T01:02:04+00:00|RUNNING\n"
            "125|qcsc-job|identity-comment|alice|acct|compute|"
            "2026-07-23T01:02:05+00:00|COMPLETED\n"
        )

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)

    candidates = asyncio.run(
        runtime_mod.SlurmRuntime().find_jobs_by_identity(
            job_name="qcsc-job",
            user="alice",
            search_start=datetime(2026, 7, 23, 1, 0, tzinfo=timezone.utc),
        )
    )

    assert [(candidate.job_id, candidate.source) for candidate in candidates] == [
        ("123", "squeue"),
        ("125", "sacct"),
    ]
    assert calls[0][0] == "squeue"
    assert "--user=alice" in calls[0]
    assert "--name=qcsc-job" in calls[0]
    assert calls[1][0] == "sacct"
    assert "--allocations" in calls[1]
    assert "--duplicates" in calls[1]
    assert any(argument.startswith("--starttime=") for argument in calls[1])


def test_find_candidates_by_identity_normalizes_backend_validation(monkeypatch):
    async def fake_find_jobs_by_identity(**_kwargs: object):
        return [
            runtime_mod.SlurmJobCandidate(
                job_id="123",
                job_name="qcsc-job",
                comment="identity-comment",
                user="alice",
                account="N/A",
                partition="compute",
                submit_time=datetime(2026, 7, 23, 1, 2, tzinfo=timezone.utc),
                state="RUNNING",
                source="squeue",
            ),
            runtime_mod.SlurmJobCandidate(
                job_id="124",
                job_name="qcsc-job",
                comment="different-comment",
                user="alice",
                account="",
                partition="compute",
                submit_time=datetime(2026, 7, 23, 1, 3, tzinfo=timezone.utc),
                state="COMPLETED",
                source="sacct",
            ),
            runtime_mod.SlurmJobCandidate(
                job_id="125",
                job_name="qcsc-job",
                comment="identity-comment",
                user="alice",
                account="",
                partition="other",
                submit_time=datetime(2026, 7, 23, 1, 4, tzinfo=timezone.utc),
                state="PENDING",
                source="squeue",
            ),
            runtime_mod.SlurmJobCandidate(
                job_id="126",
                job_name="qcsc-job",
                comment="identity-comment",
                user="alice",
                account="",
                partition="compute",
                submit_time=datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc),
                state="PENDING",
                source="squeue",
            ),
        ]

    runtime = runtime_mod.SlurmRuntime()
    monkeypatch.setattr(runtime, "find_jobs_by_identity", fake_find_jobs_by_identity)
    identity = runtime_mod.SchedulerJobIdentity(
        search_token="qcsc-job",
        stable_identity="identity-comment",
        owner="alice",
        search_start=datetime(2026, 7, 23, 1, 0, tzinfo=timezone.utc),
        search_end=datetime(2026, 7, 23, 2, 0, tzinfo=timezone.utc),
        metadata={"account": "", "partition": "compute"},
        timeout_seconds=30,
    )

    candidates = asyncio.run(runtime.find_candidates_by_identity(identity))

    assert [(candidate.job_id, candidate.identity_matches) for candidate in candidates] == [
        ("123", True),
        ("124", False),
        ("125", True),
        ("126", True),
    ]
    assert candidates[0].metadata_error is None
    assert candidates[1].metadata_error is None
    assert candidates[2].metadata_error == "job 125 partition does not match"
    assert candidates[3].metadata_error == ("job 126 is beyond the recovery clock-skew window")


def test_wait_timeout_bounds_hung_sacct_command(monkeypatch):
    captured_timeout: float | None = None

    async def fake_run_command(
        *args: str,
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        nonlocal captured_timeout
        captured_timeout = timeout_seconds
        raise runtime_mod.SchedulerCommandTimeout("sacct timed out")

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)

    with pytest.raises(runtime_mod.WaitTimeout, match="timeout waiting"):
        asyncio.run(
            runtime_mod.SlurmRuntime().wait_final_status(
                "12345",
                timeout_seconds=2.0,
            )
        )

    assert captured_timeout is not None
    assert 0 < captured_timeout <= 2.0


def test_wait_cancellation_propagates_without_scancel(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, **_kwargs: object) -> str:
        calls.append(args)
        raise asyncio.CancelledError()

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runtime_mod.SlurmRuntime().wait_final_status("12345"))

    assert calls == [
        (
            "sacct",
            "-j",
            "12345",
            "--format=JobID,State,ExitCode,Elapsed,AllocCPUS,NodeList",
            "--parsable2",
            "--noheader",
        )
    ]


def test_cancel_requires_durable_intent_confirmation(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, **_kwargs: object) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)

    with pytest.raises(runtime_mod.CancelIntentRequiredError, match="durable"):
        asyncio.run(runtime_mod.SlurmRuntime().cancel("12345"))

    assert calls == []


def test_confirmed_cancel_invokes_scancel(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, **_kwargs: object) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.SlurmRuntime()

    asyncio.run(rt.cancel("12345", intent_confirmed=True))

    assert calls == [("scancel", "12345")]


@pytest.mark.parametrize(
    ("stderr", "expected_error"),
    [
        ("scancel: error: Invalid job id specified", runtime_mod.CancelNotFoundError),
        ("Unable to contact slurm controller", runtime_mod.TemporaryCancelError),
        ("Access/permission denied", runtime_mod.CancelRejectedError),
    ],
)
def test_cancel_classifies_scheduler_responses(monkeypatch, stderr, expected_error):
    async def fake_run_command(*args: str, **_kwargs: object) -> str:
        raise runtime_mod.SchedulerCommandError(
            args=tuple(args),
            returncode=1,
            stdout="",
            stderr=stderr,
        )

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)

    with pytest.raises(expected_error):
        asyncio.run(runtime_mod.SlurmRuntime().cancel("12345", intent_confirmed=True))


def test_cancel_timeout_is_temporary(monkeypatch):
    async def fake_run_command(*_args: str, **_kwargs: object) -> str:
        raise runtime_mod.SchedulerCommandTimeout("timeout")

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)

    with pytest.raises(runtime_mod.TemporaryCancelError):
        asyncio.run(runtime_mod.SlurmRuntime().cancel("12345", intent_confirmed=True))
