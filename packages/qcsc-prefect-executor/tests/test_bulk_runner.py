from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from qcsc_prefect_core.queue import QueueCapacity
from qcsc_prefect_executor import from_blocks as mod
from qcsc_prefect_executor.bulk.exceptions import QueueFullError, TemporarySubmitError
from qcsc_prefect_executor.bulk.models import BulkJobSpec, BulkJobStatus, SubmittedJob
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry


class _RegistryCapacityProbe:
    def __init__(
        self,
        registry_path: Path,
        *,
        max_active_jobs: int = 1,
        failures: int = 0,
    ) -> None:
        self.registry_path = registry_path
        self.max_active_jobs = max_active_jobs
        self.failures = failures
        self.calls = 0

    def get_capacity(self) -> QueueCapacity:
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("probe failed")

        registry = BulkJobRegistry(self.registry_path)
        current_active_jobs = len(registry.get_active_jobs())
        return QueueCapacity(
            max_active_jobs=self.max_active_jobs,
            current_active_jobs=current_active_jobs,
            available_slots=max(0, self.max_active_jobs - current_active_jobs),
        )


def _spec(
    tmp_path: Path,
    job_key: str,
    *,
    wave_id: str | None = None,
    priority: int = 0,
    expected_outputs: list[Path] | None = None,
) -> BulkJobSpec:
    return BulkJobSpec(
        job_key=job_key,
        work_dir=tmp_path / job_key,
        wave_id=wave_id,
        priority=priority,
        expected_outputs=expected_outputs or [],
    )


def _mark_status(registry: BulkJobRegistry, job_key: str, status: BulkJobStatus) -> None:
    if status == BulkJobStatus.QUEUED:
        registry.mark_queued(job_key)
    elif status == BulkJobStatus.RUNNING:
        registry.mark_running(job_key)
    elif status == BulkJobStatus.SUCCEEDED:
        registry.mark_succeeded(job_key)
    elif status == BulkJobStatus.FAILED:
        registry.mark_failed(job_key, error="failed")
    elif status == BulkJobStatus.CANCELLED:
        registry.mark_cancelled(job_key, error="cancelled")
    elif status == BulkJobStatus.UNKNOWN:
        registry.mark_unknown(job_key, error="unknown")


def _install_fake_submit_and_monitor(
    monkeypatch,
    *,
    submitted: list[str],
    monitor_calls: list[list[str]] | None = None,
    monitor_status_by_job: dict[str, BulkJobStatus] | None = None,
    submit_failures: dict[str, list[Exception]] | None = None,
    active_counts_after_submit: list[int] | None = None,
    assert_deferred_on_retry: set[str] | None = None,
) -> None:
    scheduler_to_job: dict[str, str] = {}
    submit_failures = submit_failures or {}
    monitor_status_by_job = monitor_status_by_job or {}
    assert_deferred_on_retry = assert_deferred_on_retry or set()

    async def fake_submit_job_from_blocks(
        *,
        work_dir: Path,
        job_key: str,
        command_block: str,
        execution_profile_block: str,
        hpc_profile_block: str,
        command_args: dict[str, Any] | None = None,
        registry: BulkJobRegistry | None = None,
    ) -> SubmittedJob:
        failures = submit_failures.get(job_key, [])
        if failures:
            raise failures.pop(0)

        if registry is not None and job_key in assert_deferred_on_retry:
            record = registry.get_job(job_key)
            assert record is not None
            assert record.status == BulkJobStatus.SUBMIT_DEFERRED
            assert_deferred_on_retry.remove(job_key)

        scheduler_job_id = f"sched-{job_key}"
        scheduler_to_job[scheduler_job_id] = job_key
        submitted.append(job_key)
        if registry is not None:
            registry.mark_submitted(job_key, scheduler_job_id)
            if active_counts_after_submit is not None:
                active_counts_after_submit.append(len(registry.get_active_jobs()))
        return SubmittedJob(
            job_key=job_key,
            scheduler_job_id=scheduler_job_id,
            status=BulkJobStatus.SUBMITTED,
            work_dir=work_dir,
        )

    async def fake_monitor_jobs_many(
        *,
        hpc_profile_block: str,
        scheduler_job_ids: list[str],
        registry: BulkJobRegistry | None = None,
    ) -> dict[str, BulkJobStatus]:
        if monitor_calls is not None:
            monitor_calls.append(list(scheduler_job_ids))

        statuses: dict[str, BulkJobStatus] = {}
        for scheduler_job_id in scheduler_job_ids:
            job_key = scheduler_to_job.get(
                scheduler_job_id, scheduler_job_id.removeprefix("sched-")
            )
            status = monitor_status_by_job.get(job_key, BulkJobStatus.SUCCEEDED)
            statuses[scheduler_job_id] = status
            if registry is not None:
                _mark_status(registry, job_key, status)
        return statuses

    monkeypatch.setattr(mod, "submit_job_from_blocks", fake_submit_job_from_blocks)
    monkeypatch.setattr(mod, "monitor_jobs_many", fake_monitor_jobs_many)


def _run_bulk(
    *,
    tmp_path: Path,
    jobs: list[BulkJobSpec],
    queue_probe: _RegistryCapacityProbe,
    max_submit_per_refill: int = 100,
    stop_on_first_failure: bool = False,
):
    return asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=jobs,
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            registry_path=queue_probe.registry_path,
            queue_probe=queue_probe,
            max_active_jobs=queue_probe.max_active_jobs,
            safety_margin=0,
            max_submit_per_refill=max_submit_per_refill,
            poll_interval_seconds=0,
            refill_interval_seconds=0,
            stop_on_first_failure=stop_on_first_failure,
        )
    )


def test_run_jobs_from_blocks_bulk_submits_only_up_to_queue_capacity(
    tmp_path: Path, monkeypatch
):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    active_counts: list[int] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        active_counts_after_submit=active_counts,
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, f"job-{index}") for index in range(3)],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=1),
    )

    assert max(active_counts) == 1
    assert submitted == ["job-0", "job-1", "job-2"]
    assert result.succeeded == 3


def test_run_jobs_from_blocks_bulk_respects_max_submit_per_refill(
    tmp_path: Path, monkeypatch
):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    monitor_calls: list[list[str]] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_calls=monitor_calls,
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, f"job-{index}") for index in range(3)],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=10),
        max_submit_per_refill=2,
    )

    assert len(monitor_calls[0]) == 2
    assert submitted == ["job-0", "job-1", "job-2"]
    assert result.succeeded == 3


def test_queue_full_marks_submit_deferred_and_retries_later(
    tmp_path: Path, monkeypatch
):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        submit_failures={"job-1": [QueueFullError("queue full")]},
        assert_deferred_on_retry={"job-1"},
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=1),
    )

    assert submitted == ["job-1"]
    assert result.succeeded == 1
    assert result.failed == 0


def test_temporary_submit_error_is_retried_later(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        submit_failures={"job-1": [TemporarySubmitError("busy")]},
        assert_deferred_on_retry={"job-1"},
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=1),
    )

    assert submitted == ["job-1"]
    assert result.succeeded == 1


def test_completed_jobs_are_not_resubmitted_after_restart(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    registry = BulkJobRegistry(registry_path)
    registry.upsert_jobs([_spec(tmp_path, "done"), _spec(tmp_path, "new")])
    registry.mark_succeeded("done")
    submitted: list[str] = []
    _install_fake_submit_and_monitor(monkeypatch, submitted=submitted)

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "done"), _spec(tmp_path, "new")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=1),
    )

    assert submitted == ["new"]
    assert result.succeeded == 2


def test_active_jobs_are_monitored_after_restart_not_resubmitted(
    tmp_path: Path, monkeypatch
):
    registry_path = tmp_path / "bulk.sqlite"
    registry = BulkJobRegistry(registry_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])
    registry.mark_submitted("job-1", "sched-job-1")
    submitted: list[str] = []
    monitor_calls: list[list[str]] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_calls=monitor_calls,
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=1),
    )

    assert submitted == []
    assert monitor_calls == [["sched-job-1"]]
    assert result.succeeded == 1


def test_expected_output_existence_skips_submission(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    work_dir = tmp_path / "job-1"
    work_dir.mkdir()
    (work_dir / "done.txt").write_text("ok")
    submitted: list[str] = []
    _install_fake_submit_and_monitor(monkeypatch, submitted=submitted)

    result = asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=[
                BulkJobSpec(
                    job_key="job-1",
                    work_dir=work_dir,
                    expected_outputs=[Path("done.txt")],
                )
            ],
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            registry_path=registry_path,
            poll_interval_seconds=0,
            refill_interval_seconds=0,
        )
    )

    assert submitted == []
    assert result.succeeded == 1


def test_active_jobs_are_passed_to_monitor_many_in_batches(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    registry = BulkJobRegistry(registry_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1"), _spec(tmp_path, "job-2")])
    registry.mark_submitted("job-1", "sched-job-1")
    registry.mark_submitted("job-2", "sched-job-2")
    submitted: list[str] = []
    monitor_calls: list[list[str]] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_calls=monitor_calls,
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1"), _spec(tmp_path, "job-2")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
    )

    assert submitted == []
    assert monitor_calls == [["sched-job-1", "sched-job-2"]]
    assert result.succeeded == 2


def test_all_jobs_eventually_succeeded_returns_bulk_run_result(
    tmp_path: Path, monkeypatch
):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(monkeypatch, submitted=submitted)

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1"), _spec(tmp_path, "job-2")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
    )

    assert result.total_jobs == 2
    assert result.status_counts == {BulkJobStatus.SUCCEEDED.value: 2}
    assert result.succeeded == 2
    assert result.failed == 0
    assert result.cancelled == 0
    assert result.submit_deferred == 0
    assert result.unknown == 0
    assert result.registry_path == registry_path
    assert result.failed_jobs == []


def test_failed_job_returns_bulk_run_result(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_status_by_job={
            "job-ok": BulkJobStatus.SUCCEEDED,
            "job-fail": BulkJobStatus.FAILED,
        },
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-ok"), _spec(tmp_path, "job-fail")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
    )

    assert result.succeeded == 1
    assert result.failed == 1
    assert result.failed_jobs == ["job-fail"]


def test_stop_on_first_failure_stops_loop(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_status_by_job={
            "job-fail": BulkJobStatus.FAILED,
            "job-running": BulkJobStatus.RUNNING,
        },
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-fail"), _spec(tmp_path, "job-running")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
        stop_on_first_failure=True,
    )

    assert result.failed == 1
    assert result.status_counts == {
        BulkJobStatus.FAILED.value: 1,
        BulkJobStatus.RUNNING.value: 1,
    }


def test_wave_readiness_does_not_affect_submit_order(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    monitor_calls: list[list[str]] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_calls=monitor_calls,
    )

    jobs = [
        _spec(tmp_path, "wave-a-1", wave_id="wave-a", priority=3),
        _spec(tmp_path, "wave-b-1", wave_id="wave-b", priority=2),
        _spec(tmp_path, "wave-a-2", wave_id="wave-a", priority=1),
    ]
    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=jobs,
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
        max_submit_per_refill=2,
    )

    registry = BulkJobRegistry(registry_path)
    assert submitted[:2] == ["wave-a-1", "wave-b-1"]
    assert registry.is_wave_ready("wave-a") is True
    assert registry.is_wave_ready("wave-b") is True
    assert result.succeeded == 3


def test_queue_probe_failure_results_in_no_new_submissions_for_that_cycle(
    tmp_path: Path, monkeypatch
):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(monkeypatch, submitted=submitted)
    probe = _RegistryCapacityProbe(registry_path, max_active_jobs=1, failures=1)

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1")],
        queue_probe=probe,
    )

    assert probe.calls >= 2
    assert submitted == ["job-1"]
    assert result.succeeded == 1
