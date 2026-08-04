from __future__ import annotations

import subprocess

import pytest
from qcsc_prefect_adapters.slurm import queue as queue_mod
from qcsc_prefect_core.queue import QueueAwareSubmitGate


def test_estimate_capacity_counts_pending_running_and_expanded_array_rows():
    stdout = "101|PENDING\n102|RUNNING\n103_4|RUNNING\n"

    capacity = queue_mod.estimate_capacity_from_squeue(
        stdout,
        max_active_jobs=8,
    )

    assert capacity.max_active_jobs == 8
    assert capacity.current_active_jobs == 3
    assert capacity.available_slots == 5
    assert capacity.raw_output == stdout


def test_queue_gate_applies_safety_margin_once_to_slurm_capacity(monkeypatch):
    monkeypatch.setattr(
        queue_mod,
        "_run_squeue",
        lambda **_kwargs: "101|PENDING\n102|RUNNING\n103|RUNNING\n",
    )
    probe = queue_mod.SlurmQueueProbe(
        max_active_jobs=10,
        user="alice",
        account="project-a",
        partition="compute",
    )
    gate = QueueAwareSubmitGate(
        queue_probe=probe,
        max_active_jobs=10,
        safety_margin=2,
        max_submit_per_refill=100,
    )

    assert gate.allowed_submit_count() == 5


def test_run_squeue_uses_user_account_partition_and_expands_arrays(monkeypatch):
    calls: list[dict[str, object]] = []

    class _CompletedProcess:
        stdout = "101|PENDING\n"

    def fake_run(args: list[str], **kwargs: object) -> _CompletedProcess:
        calls.append({"args": args, **kwargs})
        return _CompletedProcess()

    monkeypatch.setattr(queue_mod.subprocess, "run", fake_run)

    stdout = queue_mod._run_squeue(
        user="alice",
        account="project-a",
        partition="compute",
        timeout_seconds=12.5,
    )

    assert stdout == "101|PENDING\n"
    assert calls == [
        {
            "args": [
                "squeue",
                "--noheader",
                "--array",
                "--states=all",
                "--user=alice",
                "--account=project-a",
                "--partition=compute",
                "--format=%i|%T",
            ],
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 12.5,
        }
    ]


def test_run_squeue_omits_empty_optional_filters(monkeypatch):
    calls: list[list[str]] = []

    class _CompletedProcess:
        stdout = ""

    def fake_run(args: list[str], **_kwargs: object) -> _CompletedProcess:
        calls.append(args)
        return _CompletedProcess()

    monkeypatch.setattr(queue_mod.subprocess, "run", fake_run)

    queue_mod._run_squeue(
        user="alice",
        account=None,
        partition=None,
        timeout_seconds=30,
    )

    assert calls == [
        [
            "squeue",
            "--noheader",
            "--array",
            "--states=all",
            "--user=alice",
            "--format=%i|%T",
        ]
    ]


def test_slurm_queue_probe_rejects_empty_user():
    with pytest.raises(ValueError, match="user is required"):
        queue_mod.SlurmQueueProbe(max_active_jobs=10, user="  ")


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(
            1,
            ["squeue"],
            output="partial output",
            stderr="controller unavailable",
        ),
        subprocess.TimeoutExpired(["squeue"], 5, output="partial output"),
    ],
)
def test_slurm_queue_probe_command_failure_returns_zero_capacity(monkeypatch, failure):
    def fail(**_kwargs: object) -> str:
        raise failure

    monkeypatch.setattr(queue_mod, "_run_squeue", fail)
    probe = queue_mod.SlurmQueueProbe(max_active_jobs=10, user="alice")

    capacity = probe.get_capacity()

    assert capacity.max_active_jobs == 10
    assert capacity.current_active_jobs == 10
    assert capacity.available_slots == 0
    assert "partial output" in str(capacity.raw_output)


def test_slurm_queue_probe_malformed_output_returns_zero_capacity(monkeypatch):
    monkeypatch.setattr(
        queue_mod,
        "_run_squeue",
        lambda **_kwargs: "unexpected scheduler output\n",
    )
    probe = queue_mod.SlurmQueueProbe(max_active_jobs=4, user="alice")

    capacity = probe.get_capacity()

    assert capacity.current_active_jobs == 4
    assert capacity.available_slots == 0
    assert "Malformed squeue" in str(capacity.raw_output)
