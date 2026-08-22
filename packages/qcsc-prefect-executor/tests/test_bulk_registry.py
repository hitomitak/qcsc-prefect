from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from qcsc_prefect_executor.bulk.exceptions import (
    SchedulerIdentityMismatchError,
    SpecHashMismatchError,
)
from qcsc_prefect_executor.bulk.models import (
    BulkCancelOutcome,
    BulkJobDesiredState,
    BulkJobSpec,
    BulkJobStatus,
)
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
    stage_id: str | None = None,
    priority: int = 0,
    max_submit_attempts: int = 5,
    execution_profile_block: str | None = None,
    hpc_profile_block: str | None = None,
    spec_hash: str | None = None,
    input_digest: str | None = None,
    code_digest: str | None = None,
    environment_digest: str | None = None,
    job_name: str | None = None,
    job_comment: str | None = None,
) -> BulkJobSpec:
    return BulkJobSpec(
        job_key=job_key,
        wave_id=wave_id,
        target_id=target_id,
        stage_id=stage_id,
        work_dir=tmp_path / job_key,
        command_args=command_args or {"index": job_key},
        expected_outputs=expected_outputs or [],
        priority=priority,
        max_submit_attempts=max_submit_attempts,
        execution_profile_block=execution_profile_block,
        hpc_profile_block=hpc_profile_block,
        spec_hash=spec_hash,
        input_digest=input_digest,
        code_digest=code_digest,
        environment_digest=environment_digest,
        job_name=job_name,
        job_comment=job_comment,
    )


def _only_job(registry: BulkJobRegistry, wave_id: str = "wave-a"):
    jobs = registry.jobs_for_wave(wave_id)
    assert len(jobs) == 1
    return jobs[0]


def _create_old_schema_registry(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE bulk_jobs (
                job_key TEXT PRIMARY KEY,
                wave_id TEXT,
                target_id TEXT,
                status TEXT NOT NULL,
                work_dir TEXT NOT NULL,
                scheduler_job_id TEXT,
                submit_attempts INTEGER NOT NULL DEFAULT 0,
                monitor_attempts INTEGER NOT NULL DEFAULT 0,
                command_args_json TEXT,
                expected_outputs_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                submitted_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                last_error TEXT,
                priority INTEGER NOT NULL DEFAULT 0,
                max_submit_attempts INTEGER NOT NULL DEFAULT 5
            )
            """
        )
        conn.execute(
            """
            INSERT INTO bulk_jobs (
                job_key,
                wave_id,
                target_id,
                status,
                work_dir,
                scheduler_job_id,
                submit_attempts,
                monitor_attempts,
                command_args_json,
                expected_outputs_json,
                created_at,
                updated_at,
                submitted_at,
                started_at,
                finished_at,
                last_error,
                priority,
                max_submit_attempts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-job",
                "wave-old",
                "target-old",
                BulkJobStatus.SUBMITTED.value,
                str(db_path.parent / "old-job"),
                "43607196",
                1,
                0,
                '{"index": "old-job"}',
                "[]",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                None,
                None,
                None,
                0,
                5,
            ),
        )


def test_registry_creates_sqlite_file_and_table(tmp_path: Path):
    db_path = tmp_path / "bulk.sqlite"
    BulkJobRegistry(db_path)

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'bulk_jobs'"
        ).fetchone()
    assert row == ("bulk_jobs",)


def test_registry_migrates_old_schema_with_resumable_submit_columns(tmp_path: Path):
    db_path = tmp_path / "bulk.sqlite"
    _create_old_schema_registry(db_path)

    registry = BulkJobRegistry(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(bulk_jobs)").fetchall()}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(bulk_jobs)").fetchall()}

    assert {
        "stage_id",
        "submit_mode",
        "bulk_group_key",
        "bulk_parent_job_id",
        "bulk_index",
        "scheduler_subjob_id",
        "execution_profile_block",
        "hpc_profile_block",
        "spec_hash",
        "input_digest",
        "code_digest",
        "environment_digest",
        "prepared_at",
        "job_name",
        "job_comment",
        "desired_state",
        "cancel_requested_at",
        "cancel_requested_by",
        "cancel_reason",
        "cancel_attempts",
        "cancel_dispatch_started_at",
        "cancel_outcome",
        "cancel_outcome_at",
        "cancel_last_error",
    } <= columns
    assert {
        "idx_bulk_jobs_status",
        "idx_bulk_jobs_status_created_at",
        "idx_bulk_jobs_stage_wave_status",
        "idx_bulk_jobs_bulk_parent_job_id",
        "idx_bulk_jobs_scheduler_subjob_id",
    } <= indexes

    record = registry.get_job("old-job")
    assert record is not None
    assert record.stage_id is None
    assert record.submit_mode == "single"
    assert record.bulk_group_key is None
    assert record.bulk_parent_job_id is None
    assert record.bulk_index is None
    assert record.scheduler_subjob_id is None
    assert record.execution_profile_block is None
    assert record.hpc_profile_block is None
    assert record.spec_hash is None
    assert record.input_digest is None
    assert record.code_digest is None
    assert record.environment_digest is None
    assert record.prepared_at is None
    assert record.job_name is None
    assert record.job_comment is None
    assert record.desired_state == BulkJobDesiredState.RUN
    assert record.cancel_requested_at is None
    assert record.cancel_requested_by is None
    assert record.cancel_reason is None
    assert record.cancel_attempts == 0
    assert record.cancel_dispatch_started_at is None
    assert record.cancel_outcome is None
    assert record.cancel_outcome_at is None
    assert record.cancel_last_error is None
    assert record.effective_scheduler_job_id == "43607196"


def test_upsert_jobs_inserts_new_jobs(tmp_path: Path):
    registry = _registry(tmp_path)

    registry.upsert_jobs([_spec(tmp_path, "job-1"), _spec(tmp_path, "job-2")])

    assert registry.status_counts() == {BulkJobStatus.PENDING.value: 2}
    assert [job.job_key for job in registry.jobs_for_wave("wave-a")] == ["job-1", "job-2"]


def test_new_resumable_submit_fields_default_without_changing_existing_api(tmp_path: Path):
    registry = _registry(tmp_path)

    registry.upsert_jobs([_spec(tmp_path, "job-1")])

    record = _only_job(registry)
    assert record.spec_hash is None
    assert record.input_digest is None
    assert record.code_digest is None
    assert record.environment_digest is None
    assert record.prepared_at is None
    assert record.job_name is None
    assert record.job_comment is None
    assert record.desired_state == BulkJobDesiredState.RUN
    assert record.cancel_requested_at is None
    assert record.cancel_requested_by is None
    assert record.cancel_reason is None
    assert record.cancel_attempts == 0
    assert record.cancel_dispatch_started_at is None
    assert record.cancel_outcome is None
    assert record.cancel_outcome_at is None
    assert record.cancel_last_error is None


def test_resumable_submit_fields_round_trip(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                spec_hash="v1:abc123",
                input_digest="input-abc",
                code_digest="code-abc",
                environment_digest="environment-abc",
                job_name="qcsc-job-abc123",
                job_comment="qcsc-spec=v1:abc123",
            )
        ]
    )
    registry.upsert_jobs([_spec(tmp_path, "job-1")])

    legacy_upsert = _only_job(registry)
    assert legacy_upsert.spec_hash == "v1:abc123"
    assert legacy_upsert.input_digest == "input-abc"
    assert legacy_upsert.code_digest == "code-abc"
    assert legacy_upsert.environment_digest == "environment-abc"
    assert legacy_upsert.job_name == "qcsc-job-abc123"
    assert legacy_upsert.job_comment == "qcsc-spec=v1:abc123"

    with sqlite3.connect(registry.path) as conn:
        conn.execute(
            """
            UPDATE bulk_jobs
            SET
                status = ?,
                prepared_at = ?,
                desired_state = ?,
                cancel_requested_at = ?,
                cancel_requested_by = ?,
                cancel_reason = ?
            WHERE job_key = ?
            """,
            (
                BulkJobStatus.PREPARED.value,
                "2026-07-22T01:02:03+00:00",
                BulkJobDesiredState.CANCEL_REQUESTED.value,
                "2026-07-22T01:03:00+00:00",
                "operator@example.invalid",
                "test cancellation intent",
                "job-1",
            ),
        )

    record = _only_job(registry)
    assert record.status == BulkJobStatus.PREPARED
    assert record.spec_hash == "v1:abc123"
    assert record.input_digest == "input-abc"
    assert record.code_digest == "code-abc"
    assert record.environment_digest == "environment-abc"
    assert record.prepared_at == "2026-07-22T01:02:03+00:00"
    assert record.job_name == "qcsc-job-abc123"
    assert record.job_comment == "qcsc-spec=v1:abc123"
    assert record.desired_state == BulkJobDesiredState.CANCEL_REQUESTED
    assert record.cancel_requested_at == "2026-07-22T01:03:00+00:00"
    assert record.cancel_requested_by == "operator@example.invalid"
    assert record.cancel_reason == "test cancellation intent"


def test_request_cancel_is_atomic_idempotent_and_excludes_submit_candidate(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])

    def request(index: int) -> bool:
        return registry.request_cancel(
            "job-1",
            requested_by=f"operator-{index}",
            reason=f"reason-{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(request, range(8)))

    assert sum(results) == 1
    record = registry.get_job("job-1")
    assert record is not None
    assert record.desired_state == BulkJobDesiredState.CANCEL_REQUESTED
    assert record.cancel_requested_at is not None
    assert record.cancel_requested_by is not None
    assert record.cancel_reason is not None
    assert registry.get_submit_candidates(limit=10) == []
    assert [job.job_key for job in registry.get_pending_cancel_requests()] == ["job-1"]


def test_request_cancel_validates_audit_fields_and_job_key(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])

    with pytest.raises(ValueError, match="non-empty"):
        registry.request_cancel("job-1", requested_by="", reason="reason")
    with pytest.raises(ValueError, match="non-empty"):
        registry.request_cancel("job-1", requested_by="operator", reason=" ")
    with pytest.raises(KeyError, match="missing"):
        registry.request_cancel("missing", requested_by="operator", reason="reason")


def test_pending_cancel_completes_without_scheduler_side_effect(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])
    assert registry.request_cancel("job-1", requested_by="operator", reason="no longer needed")

    assert registry.cancel_without_scheduler_submission("job-1")
    assert not registry.cancel_without_scheduler_submission("job-1")

    record = registry.get_job("job-1")
    assert record is not None
    assert record.status == BulkJobStatus.CANCELLED
    assert record.cancel_outcome == BulkCancelOutcome.NOT_SUBMITTED
    assert record.cancel_attempts == 0
    assert record.finished_at is not None


def test_cancel_dispatch_claim_has_one_winner_and_records_outcome(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])
    registry.mark_submitted("job-1", "12345")
    registry.request_cancel("job-1", requested_by="operator", reason="stop")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _index: registry.claim_cancel_dispatch("job-1"), range(8)))

    assert sum(results) == 1
    claimed = registry.get_job("job-1")
    assert claimed is not None
    assert claimed.cancel_attempts == 1
    assert claimed.cancel_dispatch_started_at is not None
    assert claimed.cancel_outcome == BulkCancelOutcome.DISPATCHING

    assert registry.record_cancel_outcome("job-1", BulkCancelOutcome.REQUEST_ACCEPTED)
    assert not registry.record_cancel_outcome("job-1", BulkCancelOutcome.REQUEST_ACCEPTED)
    completed = registry.get_job("job-1")
    assert completed is not None
    assert completed.cancel_outcome == BulkCancelOutcome.REQUEST_ACCEPTED
    assert completed.cancel_outcome_at is not None


def test_same_spec_hash_is_idempotent_but_mismatch_preserves_existing_row(tmp_path: Path):
    registry = _registry(tmp_path)
    original = _spec(
        tmp_path,
        "job-1",
        command_args={"input": "original.dat"},
        spec_hash="qcsc-prefect-bulk-spec-v1:sha256:aaa",
        input_digest="input-original",
    )
    registry.upsert_jobs([original])
    registry.upsert_jobs([original])

    with pytest.raises(SpecHashMismatchError) as exc_info:
        registry.upsert_jobs(
            [
                _spec(
                    tmp_path,
                    "job-1",
                    command_args={"input": "changed.dat"},
                    spec_hash="qcsc-prefect-bulk-spec-v1:sha256:bbb",
                    input_digest="input-changed",
                )
            ]
        )

    record = _only_job(registry)
    assert record.spec_hash == "qcsc-prefect-bulk-spec-v1:sha256:aaa"
    assert record.command_args == {"input": "original.dat"}
    assert record.input_digest == "input-original"
    assert exc_info.value.job_key == "job-1"
    assert "original.dat" not in str(exc_info.value)
    assert "changed.dat" not in str(exc_info.value)


def test_legacy_active_row_without_hash_keeps_execution_spec_immutable(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                command_args={"input": "original.dat"},
            )
        ]
    )
    registry.mark_submitted("job-1", "12345")

    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                command_args={"input": "changed.dat"},
                spec_hash="qcsc-prefect-bulk-spec-v1:sha256:new",
                input_digest="input-changed",
            )
        ]
    )

    record = _only_job(registry)
    assert record.status == BulkJobStatus.SUBMITTED
    assert record.scheduler_job_id == "12345"
    assert record.command_args == {"input": "original.dat"}
    assert record.spec_hash is None
    assert record.input_digest is None


def test_stage_and_native_bulk_fields_round_trip(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                stage_id="stage-a",
                wave_id="wave-a",
            )
        ]
    )

    pending = _only_job(registry)
    assert pending.stage_id == "stage-a"
    assert pending.submit_mode == "single"
    assert pending.bulk_group_key is None
    assert pending.effective_scheduler_job_id is None

    registry.mark_submitted(
        "job-1",
        "12345",
        submit_mode="native_bulk",
        bulk_group_key="bulk-stage-a-0001",
        bulk_parent_job_id="12345",
        bulk_index=7,
        scheduler_subjob_id="12345[7]",
    )

    submitted = _only_job(registry)
    assert submitted.status == BulkJobStatus.SUBMITTED
    assert submitted.scheduler_job_id == "12345"
    assert submitted.stage_id == "stage-a"
    assert submitted.submit_mode == "native_bulk"
    assert submitted.bulk_group_key == "bulk-stage-a-0001"
    assert submitted.bulk_parent_job_id == "12345"
    assert submitted.bulk_index == 7
    assert submitted.scheduler_subjob_id == "12345[7]"
    assert submitted.effective_scheduler_job_id == "12345[7]"


def test_per_job_block_fields_round_trip_and_update_while_pending(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                execution_profile_block="exec-small",
                hpc_profile_block="hpc-a",
            )
        ]
    )

    pending = _only_job(registry)
    assert pending.execution_profile_block == "exec-small"
    assert pending.hpc_profile_block == "hpc-a"

    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                execution_profile_block="exec-large",
                hpc_profile_block="hpc-b",
            )
        ]
    )

    updated = _only_job(registry)
    assert updated.execution_profile_block == "exec-large"
    assert updated.hpc_profile_block == "hpc-b"


def test_per_job_block_fields_are_preserved_after_scheduler_submit(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                execution_profile_block="exec-original",
                hpc_profile_block="hpc-original",
            )
        ]
    )
    registry.mark_submitted("job-1", "43607196")

    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                execution_profile_block="exec-new",
                hpc_profile_block="hpc-new",
            )
        ]
    )

    submitted = _only_job(registry)
    assert submitted.execution_profile_block == "exec-original"
    assert submitted.hpc_profile_block == "hpc-original"


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


def test_all_status_properties_are_classified():
    expected = {
        BulkJobStatus.PENDING: (False, False, True, False),
        BulkJobStatus.SUBMIT_DEFERRED: (False, False, True, False),
        BulkJobStatus.PREPARED: (False, False, False, True),
        BulkJobStatus.SUBMITTED: (False, True, False, False),
        BulkJobStatus.QUEUED: (False, True, False, False),
        BulkJobStatus.RUNNING: (False, True, False, False),
        BulkJobStatus.SUCCEEDED: (True, False, False, False),
        BulkJobStatus.FAILED: (True, False, False, False),
        BulkJobStatus.CANCELLED: (True, False, False, False),
        BulkJobStatus.UNKNOWN: (False, False, False, True),
        BulkJobStatus.AWAITING_OPERATOR: (False, False, False, False),
    }

    assert set(expected) == set(BulkJobStatus)
    for status, classification in expected.items():
        assert (
            status.is_terminal,
            status.is_active,
            status.is_submit_candidate,
            status.is_recovery_candidate,
        ) == classification


def test_submit_deferred_jobs_are_submit_candidates(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1"), _spec(tmp_path, "job-2")])
    registry.mark_submit_deferred("job-1", error="queue full")

    candidates = registry.get_submit_candidates(limit=10)

    assert {job.job_key for job in candidates} == {"job-1", "job-2"}
    assert BulkJobStatus.SUBMIT_DEFERRED in {job.status for job in candidates}


def test_get_submit_candidates_fifo_uses_insert_order_not_priority(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(tmp_path, "first", priority=0),
            _spec(tmp_path, "second", priority=100),
            _spec(tmp_path, "third", priority=50),
            _spec(tmp_path, "active"),
        ]
    )
    registry.mark_submit_deferred("third", error="queue full")
    registry.mark_submitted("active", "scheduler-active")

    candidates = registry.get_submit_candidates_fifo(limit=10)

    assert [job.job_key for job in candidates] == ["first", "second", "third"]
    assert [job.status for job in candidates] == [
        BulkJobStatus.PENDING,
        BulkJobStatus.PENDING,
        BulkJobStatus.SUBMIT_DEFERRED,
    ]
    assert registry.get_submit_candidates_fifo(limit=2)[-1].job_key == "second"


def test_fifo_candidates_exclude_deferred_jobs_at_submit_attempt_limit(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "deferred", max_submit_attempts=1)])
    registry.mark_submit_deferred("deferred", error="queue full")

    assert registry.get_submit_candidates_fifo(limit=10) == []
    assert registry.count_submit_candidates() == 0


def test_count_helpers_and_bootstrap_done(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(tmp_path, "pending"),
            _spec(tmp_path, "deferred"),
            _spec(tmp_path, "submitted"),
            _spec(tmp_path, "queued"),
            _spec(tmp_path, "running"),
            _spec(tmp_path, "succeeded"),
        ]
    )
    assert registry.bootstrap_done() is False

    registry.mark_submit_deferred("deferred", error="queue full")
    registry.mark_submitted("submitted", "1")
    registry.mark_queued("queued")
    registry.mark_running("running")
    registry.mark_succeeded("succeeded")

    assert registry.count_submit_candidates() == 2
    assert registry.count_active_jobs() == 3
    assert registry.bootstrap_done() is True


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


def test_reset_jobs_for_rerun_resets_failed_unknown_and_clears_scheduler_fields(
    tmp_path: Path,
):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "failed",
                stage_id="stage-a",
                target_id="target-failed",
                command_args={"kind": "failed"},
                expected_outputs=[Path("done.txt")],
                spec_hash="qcsc-prefect-bulk-spec-v1:sha256:failed",
                input_digest="input-failed",
                code_digest="code-failed",
                environment_digest="environment-failed",
                job_name="qcsc-failed",
                job_comment="qcsc-spec=failed",
            ),
            _spec(
                tmp_path,
                "unknown",
                stage_id="stage-a",
                target_id="target-unknown",
                command_args={"kind": "unknown"},
                expected_outputs=[Path("result.json")],
                spec_hash="qcsc-prefect-bulk-spec-v1:sha256:unknown",
                input_digest="input-unknown",
                code_digest="code-unknown",
                environment_digest="environment-unknown",
                job_name="qcsc-unknown",
                job_comment="qcsc-spec=unknown",
            ),
            _spec(
                tmp_path,
                "succeeded",
                spec_hash="qcsc-prefect-bulk-spec-v1:sha256:succeeded",
                input_digest="input-succeeded",
                code_digest="code-succeeded",
                environment_digest="environment-succeeded",
                job_name="qcsc-succeeded",
                job_comment="qcsc-spec=succeeded",
            ),
        ]
    )
    for index, job_key in enumerate(["failed", "unknown", "succeeded"]):
        registry.mark_submitted(
            job_key,
            "12345",
            submit_mode="native_bulk",
            bulk_group_key="bulk-group",
            bulk_parent_job_id="12345",
            bulk_index=index,
            scheduler_subjob_id=f"12345[{index}]",
        )
    registry.mark_failed("failed", error="exit 1")
    registry.mark_unknown("unknown", error="missing")
    registry.mark_succeeded("succeeded")

    reset_count = registry.reset_jobs_for_rerun()

    assert reset_count == 2
    expected_errors = {
        "failed": "exit 1",
        "unknown": "missing",
    }
    expected_outputs = {
        "failed": [Path("done.txt")],
        "unknown": [Path("result.json")],
    }
    for job_key in ["failed", "unknown"]:
        record = registry.get_job(job_key)
        assert record is not None
        assert record.status == BulkJobStatus.PENDING
        assert record.scheduler_job_id is None
        assert record.submit_attempts == 0
        assert record.monitor_attempts == 0
        assert record.submitted_at is None
        assert record.started_at is None
        assert record.finished_at is None
        assert record.last_error == expected_errors[job_key]
        assert record.submit_mode == "single"
        assert record.bulk_group_key is None
        assert record.bulk_parent_job_id is None
        assert record.bulk_index is None
        assert record.scheduler_subjob_id is None
        assert record.effective_scheduler_job_id is None
        assert record.spec_hash is None
        assert record.input_digest is None
        assert record.code_digest is None
        assert record.environment_digest is None
        assert record.prepared_at is None
        assert record.job_name is None
        assert record.job_comment is None
        assert record.stage_id == "stage-a"
        assert record.wave_id == "wave-a"
        assert record.target_id == f"target-{job_key}"
        assert record.work_dir == tmp_path / job_key
        assert record.command_args == {"kind": job_key}
        assert record.expected_outputs == expected_outputs[job_key]

    succeeded = registry.get_job("succeeded")
    assert succeeded is not None
    assert succeeded.status == BulkJobStatus.SUCCEEDED
    assert succeeded.scheduler_job_id == "12345"
    assert succeeded.scheduler_subjob_id == "12345[2]"
    assert succeeded.submit_mode == "native_bulk"
    assert succeeded.spec_hash == "qcsc-prefect-bulk-spec-v1:sha256:succeeded"
    assert succeeded.job_name == "qcsc-succeeded"


def test_reset_jobs_for_rerun_accepts_a_changed_spec_for_the_same_job_key(
    tmp_path: Path,
):
    registry = _registry(tmp_path)
    old_spec = _spec(
        tmp_path,
        "failed",
        command_args={"input": "old.dat"},
        spec_hash="qcsc-prefect-bulk-spec-v1:sha256:old",
        input_digest="input-old",
        code_digest="code-old",
        environment_digest="environment-old",
        job_name="qcsc-old",
        job_comment="qcsc-spec=old",
    )
    new_spec = _spec(
        tmp_path,
        "failed",
        command_args={"input": "new.dat"},
        spec_hash="qcsc-prefect-bulk-spec-v1:sha256:new",
        input_digest="input-new",
        code_digest="code-new",
        environment_digest="environment-new",
        job_name="qcsc-new",
        job_comment="qcsc-spec=new",
    )
    registry.upsert_jobs([old_spec])
    registry.mark_failed("failed", error="exit 1")

    with pytest.raises(SpecHashMismatchError):
        registry.upsert_jobs([new_spec])

    assert registry.reset_jobs_for_rerun(job_keys=["failed"]) == 1
    registry.upsert_jobs([new_spec])

    record = registry.get_job("failed")
    assert record is not None
    assert record.status == BulkJobStatus.PENDING
    assert record.command_args == {"input": "new.dat"}
    assert record.spec_hash == "qcsc-prefect-bulk-spec-v1:sha256:new"
    assert record.input_digest == "input-new"
    assert record.code_digest == "code-new"
    assert record.environment_digest == "environment-new"
    assert record.job_name == "qcsc-new"
    assert record.job_comment == "qcsc-spec=new"


def test_reset_jobs_for_rerun_can_clear_error(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "failed")])
    registry.mark_failed("failed", error="exit 1")

    reset_count = registry.reset_jobs_for_rerun(clear_error=True)

    assert reset_count == 1
    record = registry.get_job("failed")
    assert record is not None
    assert record.status == BulkJobStatus.PENDING
    assert record.last_error is None


def test_reset_jobs_for_rerun_does_not_reset_active_or_succeeded_by_default(
    tmp_path: Path,
):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(tmp_path, "submitted"),
            _spec(tmp_path, "queued"),
            _spec(tmp_path, "running"),
            _spec(tmp_path, "succeeded"),
            _spec(tmp_path, "failed"),
            _spec(tmp_path, "unknown"),
        ]
    )
    registry.mark_submitted("submitted", "1")
    registry.mark_submitted("queued", "2")
    registry.mark_queued("queued")
    registry.mark_submitted("running", "3")
    registry.mark_running("running")
    registry.mark_succeeded("succeeded")
    registry.mark_failed("failed", error="exit 1")
    registry.mark_unknown("unknown", error="missing")

    reset_count = registry.reset_jobs_for_rerun()

    assert reset_count == 2
    assert registry.get_job("submitted").status == BulkJobStatus.SUBMITTED
    assert registry.get_job("queued").status == BulkJobStatus.QUEUED
    assert registry.get_job("running").status == BulkJobStatus.RUNNING
    assert registry.get_job("succeeded").status == BulkJobStatus.SUCCEEDED
    assert registry.get_job("failed").status == BulkJobStatus.PENDING
    assert registry.get_job("unknown").status == BulkJobStatus.PENDING


def test_reset_jobs_for_rerun_job_keys_restrict_target(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(tmp_path, "failed-1"),
            _spec(tmp_path, "failed-2"),
            _spec(tmp_path, "unknown-1"),
        ]
    )
    registry.mark_failed("failed-1", error="exit 1")
    registry.mark_failed("failed-2", error="exit 2")
    registry.mark_unknown("unknown-1", error="missing")

    reset_count = registry.reset_jobs_for_rerun(
        job_keys=["failed-2", "unknown-1"],
    )

    assert reset_count == 2
    assert registry.get_job("failed-1").status == BulkJobStatus.FAILED
    assert registry.get_job("failed-2").status == BulkJobStatus.PENDING
    assert registry.get_job("unknown-1").status == BulkJobStatus.PENDING


def test_reset_jobs_for_rerun_respects_status_filter(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "failed"), _spec(tmp_path, "unknown")])
    registry.mark_failed("failed", error="exit 1")
    registry.mark_unknown("unknown", error="missing")

    reset_count = registry.reset_jobs_for_rerun(statuses=[BulkJobStatus.UNKNOWN])

    assert reset_count == 1
    assert registry.get_job("failed").status == BulkJobStatus.FAILED
    assert registry.get_job("unknown").status == BulkJobStatus.PENDING


def test_reset_jobs_for_rerun_can_preserve_scheduler_ids(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "failed",
                spec_hash="qcsc-prefect-bulk-spec-v1:sha256:old",
                input_digest="input-old",
                code_digest="code-old",
                environment_digest="environment-old",
                job_name="qcsc-old",
                job_comment="qcsc-spec=old",
            )
        ]
    )
    registry.mark_submitted(
        "failed",
        "12345",
        submit_mode="native_bulk",
        bulk_group_key="bulk-group",
        bulk_parent_job_id="12345",
        bulk_index=7,
        scheduler_subjob_id="12345[7]",
    )
    registry.mark_failed("failed", error="exit 1")

    reset_count = registry.reset_jobs_for_rerun(clear_scheduler_ids=False)

    assert reset_count == 1
    record = registry.get_job("failed")
    assert record is not None
    assert record.status == BulkJobStatus.PENDING
    assert record.scheduler_job_id == "12345"
    assert record.submit_mode == "native_bulk"
    assert record.bulk_group_key == "bulk-group"
    assert record.bulk_parent_job_id == "12345"
    assert record.bulk_index == 7
    assert record.scheduler_subjob_id == "12345[7]"
    assert record.submitted_at is not None
    assert record.finished_at is not None
    assert record.last_error == "exit 1"
    assert record.spec_hash == "qcsc-prefect-bulk-spec-v1:sha256:old"
    assert record.input_digest == "input-old"
    assert record.code_digest == "code-old"
    assert record.environment_digest == "environment-old"
    assert record.job_name == "qcsc-old"
    assert record.job_comment == "qcsc-spec=old"


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


def test_claim_prepared_is_atomic_across_concurrent_callers(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                spec_hash="a" * 64,
                job_name="qcsc-job-1",
                job_comment="qcsc:v1:" + "a" * 64,
            )
        ]
    )

    def claim() -> bool:
        return registry.claim_prepared(
            job_key="job-1",
            spec_hash="a" * 64,
            job_name="qcsc-job-1",
            job_comment="qcsc:v1:" + "a" * 64,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: claim(), range(8)))

    assert results.count(True) == 1
    record = registry.get_job("job-1")
    assert record is not None
    assert record.status == BulkJobStatus.PREPARED
    assert record.prepared_at is not None
    assert record.submit_attempts == 1


def test_prepared_identity_is_immutable(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                spec_hash="a" * 64,
                job_name="qcsc-job-1",
                job_comment="qcsc:v1:" + "a" * 64,
            )
        ]
    )
    assert registry.claim_prepared(
        job_key="job-1",
        spec_hash="a" * 64,
        job_name="qcsc-job-1",
        job_comment="qcsc:v1:" + "a" * 64,
    )

    with pytest.raises(SchedulerIdentityMismatchError):
        registry.upsert_jobs(
            [
                _spec(
                    tmp_path,
                    "job-1",
                    spec_hash="a" * 64,
                    job_name="qcsc-different",
                    job_comment="qcsc:v1:" + "a" * 64,
                )
            ]
        )


def test_mark_submitted_does_not_double_count_prepared_attempt(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                spec_hash="a" * 64,
                job_name="qcsc-job-1",
                job_comment="qcsc:v1:" + "a" * 64,
            )
        ]
    )
    assert registry.claim_prepared(
        job_key="job-1",
        spec_hash="a" * 64,
        job_name="qcsc-job-1",
        job_comment="qcsc:v1:" + "a" * 64,
    )

    registry.mark_submitted("job-1", "43607196")

    record = registry.get_job("job-1")
    assert record is not None
    assert record.status == BulkJobStatus.SUBMITTED
    assert record.scheduler_job_id == "43607196"
    assert record.submit_attempts == 1


def test_mark_submitted_same_id_does_not_regress_running_state(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])
    assert registry.mark_submitted("job-1", "43607196")
    registry.mark_running("job-1")

    assert registry.mark_submitted("job-1", "43607196")

    record = registry.get_job("job-1")
    assert record is not None
    assert record.status == BulkJobStatus.RUNNING
    assert record.scheduler_job_id == "43607196"
    assert record.submit_attempts == 1


def test_mark_submitted_cannot_overwrite_operator_hold(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])
    registry.mark_awaiting_operator("job-1", "manual review")

    assert registry.mark_submitted("job-1", "43607196") is False

    record = registry.get_job("job-1")
    assert record is not None
    assert record.status == BulkJobStatus.AWAITING_OPERATOR
    assert record.scheduler_job_id is None


def test_mark_submitted_rejects_conflicting_scheduler_id(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1", spec_hash="a" * 64)])
    registry.mark_submitted("job-1", "43607196")

    with pytest.raises(SchedulerIdentityMismatchError):
        registry.mark_submitted("job-1", "99999999")

    record = registry.get_job("job-1")
    assert record is not None
    assert record.scheduler_job_id == "43607196"
    assert record.submit_attempts == 1


def test_operator_can_attach_or_explicitly_reset_held_claim(tmp_path: Path):
    registry = _registry(tmp_path)
    identity = {
        "spec_hash": "a" * 64,
        "job_name": "qcsc-job-1",
        "job_comment": "qcsc:v1:" + "a" * 64,
    }
    registry.upsert_jobs(
        [
            _spec(tmp_path, "attach", **identity),
            _spec(tmp_path, "reset", **identity),
        ]
    )
    for job_key in ("attach", "reset"):
        assert registry.claim_prepared(job_key=job_key, **identity)
        assert registry.mark_awaiting_operator(job_key, "ambiguous scheduler identity")

    assert registry.operator_attach("attach", "43607196")
    attached = registry.get_job("attach")
    assert attached is not None
    assert attached.status == BulkJobStatus.SUBMITTED
    assert attached.scheduler_job_id == "43607196"

    assert registry.confirm_not_submitted_and_reset(
        "reset",
        confirmed_by="operator@example",
        reason="squeue and sacct were checked manually",
    )
    reset = registry.get_job("reset")
    assert reset is not None
    assert reset.status == BulkJobStatus.PENDING
    assert reset.prepared_at is None
    assert reset.submit_attempts == 0
    assert "operator@example" in str(reset.last_error)


def test_operator_attach_rejects_non_allocation_job_id(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])
    registry.mark_awaiting_operator("job-1", "manual review")

    with pytest.raises(ValueError, match="numeric Slurm allocation"):
        registry.operator_attach("job-1", "43607196.batch")

    assert registry.get_job("job-1").status == BulkJobStatus.AWAITING_OPERATOR


def test_output_refresh_does_not_bypass_prepared_or_operator_hold(tmp_path: Path):
    registry = _registry(tmp_path)
    work_dir = tmp_path / "job-1"
    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                expected_outputs=[Path("done.txt")],
                spec_hash="a" * 64,
                job_name="qcsc-job-1",
                job_comment="qcsc:v1:" + "a" * 64,
            )
        ]
    )
    assert registry.claim_prepared(
        job_key="job-1",
        spec_hash="a" * 64,
        job_name="qcsc-job-1",
        job_comment="qcsc:v1:" + "a" * 64,
    )
    work_dir.mkdir()
    (work_dir / "done.txt").write_text("ok")

    registry.upsert_jobs(
        [
            _spec(
                tmp_path,
                "job-1",
                expected_outputs=[Path("done.txt")],
                spec_hash="a" * 64,
                job_name="qcsc-job-1",
                job_comment="qcsc:v1:" + "a" * 64,
            )
        ]
    )
    assert registry.get_job("job-1").status == BulkJobStatus.PREPARED

    registry.refresh_completed_jobs_from_outputs()
    assert registry.get_job("job-1").status == BulkJobStatus.PREPARED

    registry.mark_awaiting_operator("job-1", "operator review required")
    registry.refresh_completed_jobs_from_outputs()
    assert registry.get_job("job-1").status == BulkJobStatus.AWAITING_OPERATOR

    registry.refresh_completed_jobs_from_outputs(include_awaiting_operator=True)
    assert registry.get_job("job-1").status == BulkJobStatus.SUCCEEDED


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
