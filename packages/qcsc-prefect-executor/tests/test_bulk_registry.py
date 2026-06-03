from __future__ import annotations

import sqlite3
from pathlib import Path

from qcsc_prefect_executor.bulk.models import BulkJobSpec, BulkJobStatus
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry


def _registry(tmp_path: Path) -> BulkJobRegistry:
    return BulkJobRegistry(tmp_path / "bulk.sqlite")


def _spec(
    tmp_path: Path,
    job_key: str,
    *,
    wave_id: str = "wave-a",
    target_id: str | None = "target-a",
    command_args: dict[str, object] | None = None,
    expected_outputs: list[Path] | None = None,
    priority: int = 0,
    max_submit_attempts: int = 5,
) -> BulkJobSpec:
    return BulkJobSpec(
        job_key=job_key,
        wave_id=wave_id,
        target_id=target_id,
        work_dir=tmp_path / job_key,
        command_args=command_args or {"index": job_key},
        expected_outputs=expected_outputs or [],
        priority=priority,
        max_submit_attempts=max_submit_attempts,
    )


def _only_job(registry: BulkJobRegistry, wave_id: str = "wave-a"):
    jobs = registry.jobs_for_wave(wave_id)
    assert len(jobs) == 1
    return jobs[0]


def test_registry_creates_sqlite_file_and_table(tmp_path: Path):
    db_path = tmp_path / "bulk.sqlite"
    BulkJobRegistry(db_path)

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'bulk_jobs'"
        ).fetchone()
    assert row == ("bulk_jobs",)


def test_upsert_jobs_inserts_new_jobs(tmp_path: Path):
    registry = _registry(tmp_path)

    registry.upsert_jobs([_spec(tmp_path, "job-1"), _spec(tmp_path, "job-2")])

    assert registry.status_counts() == {BulkJobStatus.PENDING.value: 2}
    assert [job.job_key for job in registry.jobs_for_wave("wave-a")] == ["job-1", "job-2"]


def test_upsert_jobs_is_idempotent(tmp_path: Path):
    registry = _registry(tmp_path)
    jobs = [_spec(tmp_path, "job-1")]

    registry.upsert_jobs(jobs)
    registry.upsert_jobs(jobs)

    assert registry.status_counts() == {BulkJobStatus.PENDING.value: 1}
    assert len(registry.jobs_for_wave("wave-a")) == 1


def test_upsert_jobs_does_not_overwrite_succeeded_jobs(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1", wave_id="wave-original")])
    registry.mark_succeeded("job-1")

    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                wave_id="wave-new",
                target_id="target-new",
                command_args={"changed": True},
            )
        ]
    )

    original_jobs = registry.jobs_for_wave("wave-original")
    assert len(original_jobs) == 1
    assert original_jobs[0].status == BulkJobStatus.SUCCEEDED
    assert original_jobs[0].target_id == "target-a"
    assert registry.jobs_for_wave("wave-new") == []


def test_pending_to_submitted_to_succeeded_transition_works(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])

    registry.mark_submitted("job-1", "43607196")
    submitted = _only_job(registry)
    assert submitted.status == BulkJobStatus.SUBMITTED
    assert submitted.scheduler_job_id == "43607196"
    assert submitted.submit_attempts == 1
    assert submitted.submitted_at is not None

    registry.mark_succeeded("job-1")
    succeeded = _only_job(registry)
    assert succeeded.status == BulkJobStatus.SUCCEEDED
    assert succeeded.finished_at is not None
    assert registry.all_terminal() is True


def test_submit_deferred_is_not_terminal(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])

    registry.mark_submit_deferred("job-1", error="queue full")
    record = _only_job(registry)

    assert record.status == BulkJobStatus.SUBMIT_DEFERRED
    assert record.last_error == "queue full"
    assert record.status.is_terminal is False
    assert registry.all_terminal() is False


def test_submit_deferred_jobs_are_submit_candidates(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1"), _spec(tmp_path, "job-2")])
    registry.mark_submit_deferred("job-1", error="queue full")

    candidates = registry.get_submit_candidates(limit=10)

    assert {job.job_key for job in candidates} == {"job-1", "job-2"}
    assert BulkJobStatus.SUBMIT_DEFERRED in {job.status for job in candidates}


def test_submitted_queued_and_running_jobs_are_active(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(tmp_path, "submitted"),
            _spec(tmp_path, "queued"),
            _spec(tmp_path, "running"),
        ]
    )
    registry.mark_submitted("submitted", "1")
    registry.mark_queued("queued")
    registry.mark_running("running")

    active = registry.get_active_jobs()

    assert {job.job_key for job in active} == {"submitted", "queued", "running"}
    assert {job.status for job in active} == {
        BulkJobStatus.SUBMITTED,
        BulkJobStatus.QUEUED,
        BulkJobStatus.RUNNING,
    }


def test_terminal_jobs_are_not_active(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(tmp_path, "succeeded"),
            _spec(tmp_path, "failed"),
            _spec(tmp_path, "cancelled"),
        ]
    )
    registry.mark_succeeded("succeeded")
    registry.mark_failed("failed", error="exit 1")
    registry.mark_cancelled("cancelled", error="cancelled by user")

    assert registry.get_active_jobs() == []
    assert registry.all_terminal() is True


def test_unknown_job_with_scheduler_id_is_monitorable_but_not_active(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])
    registry.mark_submitted("job-1", "43607196")
    registry.mark_unknown("job-1", error="missing from scheduler output")

    assert registry.get_active_jobs() == []
    monitorable = registry.get_monitorable_jobs()
    assert len(monitorable) == 1
    assert monitorable[0].job_key == "job-1"
    assert monitorable[0].status == BulkJobStatus.UNKNOWN
    assert monitorable[0].scheduler_job_id == "43607196"


def test_refresh_completed_jobs_from_outputs_marks_succeeded(tmp_path: Path):
    registry = _registry(tmp_path)
    work_dir = tmp_path / "job-1"
    expected_output = Path("done.txt")
    registry.upsert_jobs(
        [
            BulkJobSpec(
                job_key="job-1",
                wave_id="wave-a",
                work_dir=work_dir,
                expected_outputs=[expected_output],
            )
        ]
    )
    assert _only_job(registry).status == BulkJobStatus.PENDING

    work_dir.mkdir()
    (work_dir / expected_output).write_text("ok")
    registry.refresh_completed_jobs_from_outputs()

    assert _only_job(registry).status == BulkJobStatus.SUCCEEDED


def test_registry_reload_preserves_state(tmp_path: Path):
    db_path = tmp_path / "bulk.sqlite"
    registry = BulkJobRegistry(db_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])
    registry.mark_submitted("job-1", "43607196")

    reloaded = BulkJobRegistry(db_path)
    record = _only_job(reloaded)

    assert record.status == BulkJobStatus.SUBMITTED
    assert record.scheduler_job_id == "43607196"
    assert record.submit_attempts == 1


def test_is_wave_ready_requires_all_jobs_succeeded(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(tmp_path, "job-1", wave_id="wave-a"),
            _spec(tmp_path, "job-2", wave_id="wave-a"),
        ]
    )

    assert registry.is_wave_ready("wave-a") is False
    registry.mark_succeeded("job-1")
    assert registry.is_wave_ready("wave-a") is False
    registry.mark_succeeded("job-2")
    assert registry.is_wave_ready("wave-a") is True


def test_get_ready_waves_returns_only_fully_succeeded_waves(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(tmp_path, "ready-1", wave_id="ready"),
            _spec(tmp_path, "ready-2", wave_id="ready"),
            _spec(tmp_path, "partial-1", wave_id="partial"),
            _spec(tmp_path, "partial-2", wave_id="partial"),
            _spec(tmp_path, "failed-1", wave_id="failed"),
        ]
    )
    registry.mark_succeeded("ready-1")
    registry.mark_succeeded("ready-2")
    registry.mark_succeeded("partial-1")
    registry.mark_failed("failed-1", error="exit 1")

    assert registry.get_ready_waves() == ["ready"]


def test_status_counts_returns_correct_counts(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(tmp_path, "pending"),
            _spec(tmp_path, "submitted"),
            _spec(tmp_path, "succeeded"),
            _spec(tmp_path, "failed"),
        ]
    )
    registry.mark_submitted("submitted", "1")
    registry.mark_succeeded("succeeded")
    registry.mark_failed("failed", error="exit 1")

    assert registry.status_counts() == {
        BulkJobStatus.FAILED.value: 1,
        BulkJobStatus.PENDING.value: 1,
        BulkJobStatus.SUBMITTED.value: 1,
        BulkJobStatus.SUCCEEDED.value: 1,
    }
