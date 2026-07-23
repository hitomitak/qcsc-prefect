from __future__ import annotations

import asyncio

import pytest
from qcsc_prefect_adapters.base import subprocess as subprocess_mod


class _HangingProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False
        self.communicate_calls = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        if self.communicate_calls == 1:
            await asyncio.Future()
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class _CompletedProcess:
    returncode = 2

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"partial stdout", b"scheduler rejected request"


def test_run_scheduler_command_times_out_and_kills_process(monkeypatch) -> None:
    process = _HangingProcess()

    async def fake_create_subprocess_exec(*_args: str, **_kwargs: object) -> _HangingProcess:
        return process

    monkeypatch.setattr(
        subprocess_mod.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    with pytest.raises(
        subprocess_mod.SchedulerCommandTimeout,
        match=r"timed out after 0.01 seconds: sbatch batch.slurm",
    ):
        asyncio.run(
            subprocess_mod.run_scheduler_command(
                "sbatch",
                "batch.slurm",
                timeout_seconds=0.01,
            )
        )

    assert process.killed is True
    assert process.communicate_calls == 2


def test_run_scheduler_command_cancellation_kills_process(monkeypatch) -> None:
    process = _HangingProcess()

    async def fake_create_subprocess_exec(*_args: str, **_kwargs: object) -> _HangingProcess:
        return process

    monkeypatch.setattr(
        subprocess_mod.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    async def cancel_command() -> None:
        task = asyncio.create_task(subprocess_mod.run_scheduler_command("sbatch", "batch.slurm"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_command())

    assert process.killed is True
    assert process.communicate_calls == 2


def test_run_scheduler_command_preserves_nonzero_exit_details(monkeypatch) -> None:
    async def fake_create_subprocess_exec(*_args: str, **_kwargs: object) -> _CompletedProcess:
        return _CompletedProcess()

    monkeypatch.setattr(
        subprocess_mod.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    with pytest.raises(subprocess_mod.SchedulerCommandError) as caught:
        asyncio.run(subprocess_mod.run_scheduler_command("sbatch", "batch.slurm"))

    assert caught.value.returncode == 2
    assert caught.value.stdout == "partial stdout"
    assert caught.value.stderr == "scheduler rejected request"
    assert caught.value.command_args == ("sbatch", "batch.slurm")


@pytest.mark.parametrize("timeout_seconds", [0, -1])
def test_run_scheduler_command_rejects_non_positive_timeout(timeout_seconds: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        asyncio.run(
            subprocess_mod.run_scheduler_command(
                "sacct",
                timeout_seconds=timeout_seconds,
            )
        )
