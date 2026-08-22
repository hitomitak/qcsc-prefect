from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qcsc_prefect_executor.bulk.exceptions import (
    SchedulerIdentityMismatchError,
    SpecHashMismatchError,
)
from qcsc_prefect_executor.bulk.models import (
    ACTIVE_BULK_JOB_STATUSES,
    SUBMIT_CANDIDATE_BULK_JOB_STATUSES,
    TERMINAL_BULK_JOB_STATUSES,
    BulkCancelOutcome,
    BulkJobDesiredState,
    BulkJobRecord,
    BulkJobSpec,
    BulkJobStatus,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_values(statuses: Sequence[BulkJobStatus]) -> list[str]:
    return [status.value for status in statuses]


def _coerce_status(status: BulkJobStatus | str) -> BulkJobStatus:
    return status if isinstance(status, BulkJobStatus) else BulkJobStatus(str(status))


def _coerce_statuses(statuses: Sequence[BulkJobStatus | str]) -> list[BulkJobStatus]:
    return [_coerce_status(status) for status in statuses]


def _coerce_desired_state(value: object) -> BulkJobDesiredState:
    try:
        return BulkJobDesiredState(str(value))
    except ValueError:
        return BulkJobDesiredState.RUN


def _coerce_cancel_outcome(value: object) -> BulkCancelOutcome | None:
    if value is None:
        return None
    try:
        return BulkCancelOutcome(str(value))
    except ValueError:
        return None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _json_loads_dict(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    value = json.loads(text)
    if isinstance(value, dict):
        return value
    return {}


def _json_loads_paths(text: str | None) -> list[Path]:
    if not text:
        return []
    value = json.loads(text)
    if not isinstance(value, list):
        return []
    return [Path(path) for path in value]


def _expected_outputs_json(paths: list[Path]) -> str:
    return _json_dumps([str(path) for path in paths])


def _outputs_are_complete(paths: list[Path], *, work_dir: Path) -> bool:
    if not paths:
        return False
    normalized = [path if path.is_absolute() else work_dir / path for path in paths]
    return all(path.exists() for path in normalized)


class BulkJobRegistry:
    """Persistent SQLite registry for bulk HPC job state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bulk_jobs (
                    job_key TEXT PRIMARY KEY,
                    wave_id TEXT,
                    target_id TEXT,
                    stage_id TEXT,
                    status TEXT NOT NULL,
                    work_dir TEXT NOT NULL,
                    scheduler_job_id TEXT,
                    submit_attempts INTEGER NOT NULL DEFAULT 0,
                    monitor_attempts INTEGER NOT NULL DEFAULT 0,
                    command_args_json TEXT,
                    expected_outputs_json TEXT,
                    execution_profile_block TEXT,
                    hpc_profile_block TEXT,
                    spec_hash TEXT,
                    input_digest TEXT,
                    code_digest TEXT,
                    environment_digest TEXT,
                    prepared_at TEXT,
                    job_name TEXT,
                    job_comment TEXT,
                    desired_state TEXT NOT NULL DEFAULT 'RUN',
                    cancel_requested_at TEXT,
                    cancel_requested_by TEXT,
                    cancel_reason TEXT,
                    cancel_attempts INTEGER NOT NULL DEFAULT 0,
                    cancel_dispatch_started_at TEXT,
                    cancel_outcome TEXT,
                    cancel_outcome_at TEXT,
                    cancel_last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    submitted_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    last_error TEXT,
                    priority INTEGER NOT NULL DEFAULT 0,
                    max_submit_attempts INTEGER NOT NULL DEFAULT 5,
                    submit_mode TEXT NOT NULL DEFAULT 'single',
                    bulk_group_key TEXT,
                    bulk_parent_job_id TEXT,
                    bulk_index INTEGER,
                    scheduler_subjob_id TEXT
                )
                """
            )
            self._migrate_schema(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bulk_jobs_status ON bulk_jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bulk_jobs_wave_id ON bulk_jobs(wave_id)")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bulk_jobs_status_created_at
                ON bulk_jobs(status, created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bulk_jobs_stage_wave_status
                ON bulk_jobs(stage_id, wave_id, status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bulk_jobs_bulk_parent_job_id
                ON bulk_jobs(bulk_parent_job_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bulk_jobs_scheduler_subjob_id
                ON bulk_jobs(scheduler_subjob_id)
                """
            )

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        existing_columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(bulk_jobs)").fetchall()
        }
        column_defs = {
            "stage_id": "stage_id TEXT",
            "submit_mode": "submit_mode TEXT NOT NULL DEFAULT 'single'",
            "bulk_group_key": "bulk_group_key TEXT",
            "bulk_parent_job_id": "bulk_parent_job_id TEXT",
            "bulk_index": "bulk_index INTEGER",
            "scheduler_subjob_id": "scheduler_subjob_id TEXT",
            "execution_profile_block": "execution_profile_block TEXT",
            "hpc_profile_block": "hpc_profile_block TEXT",
            "spec_hash": "spec_hash TEXT",
            "input_digest": "input_digest TEXT",
            "code_digest": "code_digest TEXT",
            "environment_digest": "environment_digest TEXT",
            "prepared_at": "prepared_at TEXT",
            "job_name": "job_name TEXT",
            "job_comment": "job_comment TEXT",
            "desired_state": "desired_state TEXT NOT NULL DEFAULT 'RUN'",
            "cancel_requested_at": "cancel_requested_at TEXT",
            "cancel_requested_by": "cancel_requested_by TEXT",
            "cancel_reason": "cancel_reason TEXT",
            "cancel_attempts": "cancel_attempts INTEGER NOT NULL DEFAULT 0",
            "cancel_dispatch_started_at": "cancel_dispatch_started_at TEXT",
            "cancel_outcome": "cancel_outcome TEXT",
            "cancel_outcome_at": "cancel_outcome_at TEXT",
            "cancel_last_error": "cancel_last_error TEXT",
        }
        for column_name, column_def in column_defs.items():
            if column_name not in existing_columns:
                conn.execute(f"ALTER TABLE bulk_jobs ADD COLUMN {column_def}")

    def upsert_jobs(self, jobs: list[BulkJobSpec]) -> None:
        """Register jobs idempotently without resetting existing progress."""

        if not jobs:
            return

        with self._connect() as conn:
            for job in jobs:
                existing = conn.execute(
                    "SELECT * FROM bulk_jobs WHERE job_key = ?",
                    (job.job_key,),
                ).fetchone()
                if (
                    existing is not None
                    and existing["spec_hash"] is not None
                    and job.spec_hash is not None
                    and existing["spec_hash"] != job.spec_hash
                ):
                    raise SpecHashMismatchError(
                        job_key=job.job_key,
                        stored_spec_hash=str(existing["spec_hash"]),
                        incoming_spec_hash=job.spec_hash,
                    )
                if existing is not None and existing["status"] in {
                    BulkJobStatus.PREPARED.value,
                    BulkJobStatus.AWAITING_OPERATOR.value,
                }:
                    identity_changed = (
                        job.job_name is not None
                        and existing["job_name"] is not None
                        and job.job_name != existing["job_name"]
                    ) or (
                        job.job_comment is not None
                        and existing["job_comment"] is not None
                        and job.job_comment != existing["job_comment"]
                    )
                    if identity_changed:
                        raise SchedulerIdentityMismatchError(
                            job_key=job.job_key,
                            stored_spec_hash=str(existing["spec_hash"] or "<legacy-null>"),
                        )
                if existing is not None and existing["status"] == BulkJobStatus.SUCCEEDED.value:
                    continue

                now = _utcnow_iso()
                command_args_json = _json_dumps(job.command_args)
                expected_outputs_json = _expected_outputs_json(job.expected_outputs)
                outputs_complete = _outputs_are_complete(
                    job.expected_outputs,
                    work_dir=job.work_dir,
                )
                if existing is not None and existing["status"] in {
                    BulkJobStatus.PREPARED.value,
                    BulkJobStatus.AWAITING_OPERATOR.value,
                }:
                    # Output files alone cannot resolve a scheduler-side-effect
                    # ambiguity. Preserve the claim/hold until Slurm identity
                    # reconciliation or an explicit operator action completes.
                    outputs_complete = False

                if existing is None:
                    status = BulkJobStatus.SUCCEEDED if outputs_complete else BulkJobStatus.PENDING
                    conn.execute(
                        """
                        INSERT INTO bulk_jobs (
                            job_key,
                            wave_id,
                            target_id,
                            stage_id,
                            status,
                            work_dir,
                            scheduler_job_id,
                            submit_attempts,
                            monitor_attempts,
                            command_args_json,
                            expected_outputs_json,
                            execution_profile_block,
                            hpc_profile_block,
                            spec_hash,
                            input_digest,
                            code_digest,
                            environment_digest,
                            job_name,
                            job_comment,
                            desired_state,
                            created_at,
                            updated_at,
                            submitted_at,
                            started_at,
                            finished_at,
                            last_error,
                            priority,
                            max_submit_attempts
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, ?, NULL, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, NULL, NULL, NULL, ?, ?
                        )
                        """,
                        (
                            job.job_key,
                            job.wave_id,
                            job.target_id,
                            job.stage_id,
                            status.value,
                            str(job.work_dir),
                            command_args_json,
                            expected_outputs_json,
                            job.execution_profile_block,
                            job.hpc_profile_block,
                            job.spec_hash,
                            job.input_digest,
                            job.code_digest,
                            job.environment_digest,
                            job.job_name,
                            job.job_comment,
                            BulkJobDesiredState.RUN.value,
                            now,
                            now,
                            now if status == BulkJobStatus.SUCCEEDED else None,
                            job.priority,
                            job.max_submit_attempts,
                        ),
                    )
                    continue

                if outputs_complete:
                    conn.execute(
                        """
                        UPDATE bulk_jobs
                        SET
                            wave_id = ?,
                            target_id = ?,
                            stage_id = ?,
                            status = ?,
                            work_dir = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE work_dir
                            END,
                            command_args_json = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE command_args_json
                            END,
                            expected_outputs_json = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE expected_outputs_json
                            END,
                            execution_profile_block = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE execution_profile_block
                            END,
                            hpc_profile_block = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE hpc_profile_block
                            END,
                            spec_hash = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, spec_hash)
                                ELSE spec_hash
                            END,
                            input_digest = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, input_digest)
                                ELSE input_digest
                            END,
                            code_digest = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, code_digest)
                                ELSE code_digest
                            END,
                            environment_digest = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, environment_digest)
                                ELSE environment_digest
                            END,
                            job_name = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, job_name)
                                ELSE job_name
                            END,
                            job_comment = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, job_comment)
                                ELSE job_comment
                            END,
                            updated_at = ?,
                            finished_at = COALESCE(finished_at, ?),
                            last_error = NULL,
                            priority = ?,
                            max_submit_attempts = ?
                        WHERE job_key = ?
                        """,
                        (
                            job.wave_id,
                            job.target_id,
                            job.stage_id,
                            BulkJobStatus.SUCCEEDED.value,
                            str(job.work_dir),
                            command_args_json,
                            expected_outputs_json,
                            job.execution_profile_block,
                            job.hpc_profile_block,
                            job.spec_hash,
                            job.input_digest,
                            job.code_digest,
                            job.environment_digest,
                            job.job_name,
                            job.job_comment,
                            now,
                            now,
                            job.priority,
                            job.max_submit_attempts,
                            job.job_key,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE bulk_jobs
                        SET
                            wave_id = ?,
                            target_id = ?,
                            stage_id = ?,
                            work_dir = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE work_dir
                            END,
                            command_args_json = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE command_args_json
                            END,
                            expected_outputs_json = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE expected_outputs_json
                            END,
                            execution_profile_block = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE execution_profile_block
                            END,
                            hpc_profile_block = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE hpc_profile_block
                            END,
                            spec_hash = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, spec_hash)
                                ELSE spec_hash
                            END,
                            input_digest = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, input_digest)
                                ELSE input_digest
                            END,
                            code_digest = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, code_digest)
                                ELSE code_digest
                            END,
                            environment_digest = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, environment_digest)
                                ELSE environment_digest
                            END,
                            job_name = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, job_name)
                                ELSE job_name
                            END,
                            job_comment = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN COALESCE(?, job_comment)
                                ELSE job_comment
                            END,
                            updated_at = ?,
                            priority = ?,
                            max_submit_attempts = ?
                        WHERE job_key = ?
                        """,
                        (
                            job.wave_id,
                            job.target_id,
                            job.stage_id,
                            str(job.work_dir),
                            command_args_json,
                            expected_outputs_json,
                            job.execution_profile_block,
                            job.hpc_profile_block,
                            job.spec_hash,
                            job.input_digest,
                            job.code_digest,
                            job.environment_digest,
                            job.job_name,
                            job.job_comment,
                            now,
                            job.priority,
                            job.max_submit_attempts,
                            job.job_key,
                        ),
                    )

    def get_active_jobs(self) -> list[BulkJobRecord]:
        statuses = _status_values(tuple(ACTIVE_BULK_JOB_STATUSES))
        return self._fetch_records(
            f"status IN ({self._placeholders(statuses)})",
            statuses,
            "submitted_at IS NULL, submitted_at, created_at, job_key",
        )

    def get_monitorable_jobs(self) -> list[BulkJobRecord]:
        statuses = _status_values(
            (
                BulkJobStatus.SUBMITTED,
                BulkJobStatus.QUEUED,
                BulkJobStatus.RUNNING,
                BulkJobStatus.UNKNOWN,
            )
        )
        return self._fetch_records(
            f"status IN ({self._placeholders(statuses)})",
            statuses,
            "submitted_at IS NULL, submitted_at, created_at, job_key",
        )

    def get_submit_candidates(self, limit: int) -> list[BulkJobRecord]:
        if limit <= 0:
            return []

        statuses = _status_values(tuple(SUBMIT_CANDIDATE_BULK_JOB_STATUSES))
        where = (
            f"status IN ({self._placeholders(statuses)}) "
            "AND submit_attempts < max_submit_attempts AND desired_state = ?"
        )
        return self._fetch_records(
            where,
            [*statuses, BulkJobDesiredState.RUN.value, int(limit)],
            "priority DESC, created_at, job_key LIMIT ?",
        )

    def get_submit_candidates_fifo(self, limit: int) -> list[BulkJobRecord]:
        if limit <= 0:
            return []

        statuses = _status_values(tuple(SUBMIT_CANDIDATE_BULK_JOB_STATUSES))
        return self._fetch_records(
            f"status IN ({self._placeholders(statuses)}) "
            "AND submit_attempts < max_submit_attempts AND desired_state = ?",
            [*statuses, BulkJobDesiredState.RUN.value, int(limit)],
            "created_at ASC, rowid ASC, job_key ASC LIMIT ?",
        )

    def count_submit_candidates(self) -> int:
        statuses = _status_values(tuple(SUBMIT_CANDIDATE_BULK_JOB_STATUSES))
        return self._count_records(
            f"status IN ({self._placeholders(statuses)}) "
            "AND submit_attempts < max_submit_attempts AND desired_state = ?",
            [*statuses, BulkJobDesiredState.RUN.value],
        )

    def count_active_jobs(self) -> int:
        statuses = _status_values(tuple(ACTIVE_BULK_JOB_STATUSES))
        return self._count_records(
            f"status IN ({self._placeholders(statuses)})",
            statuses,
        )

    def bootstrap_done(self) -> bool:
        return (
            self._count_records(
                """
                submit_attempts > 0
                OR submitted_at IS NOT NULL
                OR scheduler_job_id IS NOT NULL
                OR scheduler_subjob_id IS NOT NULL
                """,
                [],
            )
            > 0
        )

    def get_job(self, job_key: str) -> BulkJobRecord | None:
        records = self._fetch_records("job_key = ?", [job_key], "job_key")
        if not records:
            return None
        return records[0]

    def get_all_jobs(self) -> list[BulkJobRecord]:
        return self._fetch_records("1 = 1", [], "created_at, job_key")

    def get_recovery_candidates(self) -> list[BulkJobRecord]:
        return self._fetch_records(
            "status IN (?, ?)",
            [BulkJobStatus.PREPARED.value, BulkJobStatus.UNKNOWN.value],
            "prepared_at IS NULL, prepared_at, created_at, job_key",
        )

    def get_awaiting_operator_jobs(self) -> list[BulkJobRecord]:
        return self._fetch_records(
            "status = ?",
            [BulkJobStatus.AWAITING_OPERATOR.value],
            "updated_at, job_key",
        )

    def get_pending_cancel_requests(self) -> list[BulkJobRecord]:
        """Return explicit cancellation intents not yet claimed by an executor."""

        return self._fetch_records(
            "desired_state = ? AND cancel_outcome IS NULL",
            [BulkJobDesiredState.CANCEL_REQUESTED.value],
            "cancel_requested_at, created_at, job_key",
        )

    def get_ambiguous_cancel_dispatches(self) -> list[BulkJobRecord]:
        """Return cancellation claims whose remote outcome may be unknown."""

        return self._fetch_records(
            "desired_state = ? AND cancel_outcome = ?",
            [
                BulkJobDesiredState.CANCEL_REQUESTED.value,
                BulkCancelOutcome.DISPATCHING.value,
            ],
            "cancel_dispatch_started_at, job_key",
        )

    def request_cancel(self, job_key: str, *, requested_by: str, reason: str) -> bool:
        """Persist one explicit cancellation intent atomically and idempotently.

        The first caller records the operator identity and reason. Later calls
        preserve that original audit record and return ``False``.
        """

        normalized_actor = str(requested_by).strip()
        normalized_reason = str(reason).strip()
        if not normalized_actor or not normalized_reason:
            raise ValueError("requested_by and reason must be non-empty.")

        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    desired_state = ?,
                    cancel_requested_at = ?,
                    cancel_requested_by = ?,
                    cancel_reason = ?,
                    updated_at = ?
                WHERE job_key = ? AND desired_state = ?
                """,
                (
                    BulkJobDesiredState.CANCEL_REQUESTED.value,
                    now,
                    normalized_actor,
                    normalized_reason,
                    now,
                    job_key,
                    BulkJobDesiredState.RUN.value,
                ),
            )
            if int(cursor.rowcount) == 1:
                return True
            existing = conn.execute(
                "SELECT desired_state FROM bulk_jobs WHERE job_key = ?",
                (job_key,),
            ).fetchone()
        if existing is None:
            raise KeyError(f"Unknown bulk job key {job_key!r}.")
        return False

    def cancel_without_scheduler_submission(self, job_key: str) -> bool:
        """Cancel a pending logical job that provably has no scheduler side effect."""

        now = _utcnow_iso()
        allowed_statuses = (
            BulkJobStatus.PENDING.value,
            BulkJobStatus.SUBMIT_DEFERRED.value,
        )
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE bulk_jobs
                SET
                    status = ?,
                    finished_at = COALESCE(finished_at, ?),
                    cancel_outcome = ?,
                    cancel_outcome_at = ?,
                    cancel_last_error = NULL,
                    updated_at = ?,
                    last_error = ?
                WHERE
                    job_key = ?
                    AND desired_state = ?
                    AND scheduler_job_id IS NULL
                    AND scheduler_subjob_id IS NULL
                    AND status IN ({self._placeholders(allowed_statuses)})
                """,
                (
                    BulkJobStatus.CANCELLED.value,
                    now,
                    BulkCancelOutcome.NOT_SUBMITTED.value,
                    now,
                    now,
                    "Cancelled before scheduler submission by explicit durable intent",
                    job_key,
                    BulkJobDesiredState.CANCEL_REQUESTED.value,
                    *allowed_statuses,
                ),
            )
        return int(cursor.rowcount) == 1

    def claim_cancel_dispatch(self, job_key: str) -> bool:
        """Claim the sole allowed scheduler cancellation side effect."""

        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    cancel_attempts = cancel_attempts + 1,
                    cancel_dispatch_started_at = ?,
                    cancel_outcome = ?,
                    cancel_outcome_at = NULL,
                    cancel_last_error = NULL,
                    updated_at = ?
                WHERE
                    job_key = ?
                    AND desired_state = ?
                    AND cancel_outcome IS NULL
                    AND (scheduler_job_id IS NOT NULL OR scheduler_subjob_id IS NOT NULL)
                """,
                (
                    now,
                    BulkCancelOutcome.DISPATCHING.value,
                    now,
                    job_key,
                    BulkJobDesiredState.CANCEL_REQUESTED.value,
                ),
            )
        return int(cursor.rowcount) == 1

    def record_cancel_outcome(
        self,
        job_key: str,
        outcome: BulkCancelOutcome,
        *,
        error: str | None = None,
    ) -> bool:
        """Record the auditable result of a claimed scheduler cancellation."""

        if outcome in {BulkCancelOutcome.DISPATCHING, BulkCancelOutcome.ALREADY_TERMINAL}:
            raise ValueError(f"Cannot record {outcome.value} as a dispatched outcome.")
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    cancel_outcome = ?,
                    cancel_outcome_at = ?,
                    cancel_last_error = ?,
                    updated_at = ?
                WHERE
                    job_key = ?
                    AND desired_state = ?
                    AND cancel_outcome = ?
                """,
                (
                    outcome.value,
                    now,
                    None if error is None else str(error),
                    now,
                    job_key,
                    BulkJobDesiredState.CANCEL_REQUESTED.value,
                    BulkCancelOutcome.DISPATCHING.value,
                ),
            )
        return int(cursor.rowcount) == 1

    def record_terminal_cancel_outcome(self, job_key: str) -> bool:
        """Record scheduler evidence that a requested job is already terminal."""

        now = _utcnow_iso()
        terminal_statuses = _status_values(tuple(TERMINAL_BULK_JOB_STATUSES))
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE bulk_jobs
                SET
                    cancel_outcome = ?,
                    cancel_outcome_at = ?,
                    cancel_last_error = NULL,
                    updated_at = ?
                WHERE
                    job_key = ?
                    AND desired_state = ?
                    AND cancel_outcome IS NULL
                    AND (
                        scheduler_job_id IS NOT NULL
                        OR scheduler_subjob_id IS NOT NULL
                        OR status IN ({self._placeholders(terminal_statuses)})
                    )
                """,
                (
                    BulkCancelOutcome.ALREADY_TERMINAL.value,
                    now,
                    now,
                    job_key,
                    BulkJobDesiredState.CANCEL_REQUESTED.value,
                    *terminal_statuses,
                ),
            )
        return int(cursor.rowcount) == 1

    def claim_prepared(
        self,
        *,
        job_key: str,
        spec_hash: str,
        job_name: str,
        job_comment: str,
        prepared_at: str | None = None,
    ) -> bool:
        """Atomically claim one submit candidate before any scheduler side effect."""

        now = prepared_at or _utcnow_iso()
        submit_statuses = _status_values(tuple(SUBMIT_CANDIDATE_BULK_JOB_STATUSES))
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE bulk_jobs
                SET
                    status = ?,
                    prepared_at = ?,
                    job_name = ?,
                    job_comment = ?,
                    submit_attempts = submit_attempts + 1,
                    updated_at = ?,
                    last_error = NULL
                WHERE
                    job_key = ?
                    AND status IN ({self._placeholders(submit_statuses)})
                    AND desired_state = ?
                    AND spec_hash = ?
                    AND job_name = ?
                    AND job_comment = ?
                """,
                (
                    BulkJobStatus.PREPARED.value,
                    now,
                    job_name,
                    job_comment,
                    now,
                    job_key,
                    *submit_statuses,
                    BulkJobDesiredState.RUN.value,
                    spec_hash,
                    job_name,
                    job_comment,
                ),
            )
        return int(cursor.rowcount) == 1

    def record_prepared_error(self, job_key: str, error: str) -> bool:
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE bulk_jobs
                SET updated_at = ?, last_error = ?
                WHERE job_key = ? AND status = ?
                """,
                (now, str(error), job_key, BulkJobStatus.PREPARED.value),
            )
        return int(cursor.rowcount) == 1

    def release_prepared_for_retry(self, job_key: str, error: str) -> bool:
        """Return a claimed job to retry only after proven scheduler rejection."""

        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    prepared_at = NULL,
                    updated_at = ?,
                    last_error = ?
                WHERE job_key = ? AND status = ?
                """,
                (
                    BulkJobStatus.SUBMIT_DEFERRED.value,
                    now,
                    str(error),
                    job_key,
                    BulkJobStatus.PREPARED.value,
                ),
            )
        return int(cursor.rowcount) == 1

    def mark_awaiting_operator(self, job_key: str, error: str) -> bool:
        now = _utcnow_iso()
        terminal_statuses = _status_values(tuple(TERMINAL_BULK_JOB_STATUSES))
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE bulk_jobs
                SET status = ?, updated_at = ?, last_error = ?
                WHERE job_key = ?
                  AND status NOT IN ({self._placeholders(terminal_statuses)})
                """,
                (
                    BulkJobStatus.AWAITING_OPERATOR.value,
                    now,
                    str(error),
                    job_key,
                    *terminal_statuses,
                ),
            )
        return int(cursor.rowcount) == 1

    def operator_attach(self, job_key: str, scheduler_job_id: str) -> bool:
        """Explicitly attach an operator-verified Slurm job to a held row."""

        normalized_job_id = str(scheduler_job_id).strip()
        if not normalized_job_id:
            raise ValueError("scheduler_job_id must be non-empty.")
        if not normalized_job_id.isascii() or not normalized_job_id.isdigit():
            raise ValueError("scheduler_job_id must be a numeric Slurm allocation id.")
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    scheduler_job_id = ?,
                    submitted_at = COALESCE(submitted_at, ?),
                    updated_at = ?,
                    last_error = NULL
                WHERE job_key = ? AND status IN (?, ?)
                """,
                (
                    BulkJobStatus.SUBMITTED.value,
                    normalized_job_id,
                    now,
                    now,
                    job_key,
                    BulkJobStatus.PREPARED.value,
                    BulkJobStatus.AWAITING_OPERATOR.value,
                ),
            )
        return int(cursor.rowcount) == 1

    def confirm_not_submitted_and_reset(
        self,
        job_key: str,
        *,
        confirmed_by: str,
        reason: str,
    ) -> bool:
        """Explicitly reset a held claim after an operator proves no submit."""

        normalized_actor = str(confirmed_by).strip()
        normalized_reason = str(reason).strip()
        if not normalized_actor or not normalized_reason:
            raise ValueError("confirmed_by and reason must be non-empty.")
        now = _utcnow_iso()
        audit = (
            f"Operator {normalized_actor} confirmed no scheduler submission and reset "
            f"the claim: {normalized_reason}"
        )
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    prepared_at = NULL,
                    scheduler_job_id = NULL,
                    scheduler_subjob_id = NULL,
                    submitted_at = NULL,
                    started_at = NULL,
                    finished_at = NULL,
                    submit_attempts = 0,
                    monitor_attempts = 0,
                    updated_at = ?,
                    last_error = ?
                WHERE
                    job_key = ?
                    AND status IN (?, ?)
                    AND scheduler_job_id IS NULL
                    AND scheduler_subjob_id IS NULL
                """,
                (
                    BulkJobStatus.PENDING.value,
                    now,
                    audit,
                    job_key,
                    BulkJobStatus.PREPARED.value,
                    BulkJobStatus.AWAITING_OPERATOR.value,
                ),
            )
        return int(cursor.rowcount) == 1

    def mark_submitted(
        self,
        job_key: str,
        scheduler_job_id: str,
        *,
        submit_mode: str = "single",
        bulk_group_key: str | None = None,
        bulk_parent_job_id: str | None = None,
        bulk_index: int | None = None,
        scheduler_subjob_id: str | None = None,
    ) -> bool:
        now = _utcnow_iso()
        allowed_statuses = [
            BulkJobStatus.PENDING.value,
            BulkJobStatus.SUBMIT_DEFERRED.value,
            BulkJobStatus.PREPARED.value,
        ]
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE bulk_jobs
                SET
                    status = ?,
                    scheduler_job_id = ?,
                    submit_mode = ?,
                    bulk_group_key = ?,
                    bulk_parent_job_id = ?,
                    bulk_index = ?,
                    scheduler_subjob_id = ?,
                    submit_attempts = submit_attempts
                        + CASE WHEN status = ? THEN 0 ELSE 1 END,
                    submitted_at = COALESCE(submitted_at, ?),
                    updated_at = ?,
                    last_error = NULL
                WHERE
                    job_key = ?
                    AND status != ?
                    AND scheduler_job_id IS NULL
                    AND status IN ({self._placeholders(allowed_statuses)})
                """,
                (
                    BulkJobStatus.SUBMITTED.value,
                    scheduler_job_id,
                    submit_mode,
                    bulk_group_key,
                    bulk_parent_job_id,
                    bulk_index,
                    scheduler_subjob_id,
                    BulkJobStatus.PREPARED.value,
                    now,
                    now,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                    *allowed_statuses,
                ),
            )
            if int(cursor.rowcount) == 1:
                return True

            existing = conn.execute(
                "SELECT scheduler_job_id, spec_hash FROM bulk_jobs WHERE job_key = ?",
                (job_key,),
            ).fetchone()
            if (
                existing is not None
                and existing["scheduler_job_id"] is not None
                and str(existing["scheduler_job_id"]) != str(scheduler_job_id)
            ):
                raise SchedulerIdentityMismatchError(
                    job_key=job_key,
                    stored_spec_hash=str(existing["spec_hash"] or "<legacy-null>"),
                )
            return bool(
                existing is not None
                and existing["scheduler_job_id"] is not None
                and str(existing["scheduler_job_id"]) == str(scheduler_job_id)
            )

    def mark_queued(self, job_key: str) -> None:
        self._mark_status(job_key, BulkJobStatus.QUEUED)

    def mark_running(self, job_key: str) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE job_key = ? AND status != ?
                """,
                (
                    BulkJobStatus.RUNNING.value,
                    now,
                    now,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                ),
            )

    def mark_submit_deferred(self, job_key: str, error: str | None = None) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    submit_attempts = submit_attempts
                        + CASE WHEN status = ? THEN 0 ELSE 1 END,
                    prepared_at = NULL,
                    updated_at = ?,
                    last_error = ?
                WHERE job_key = ? AND status != ?
                """,
                (
                    BulkJobStatus.SUBMIT_DEFERRED.value,
                    BulkJobStatus.PREPARED.value,
                    now,
                    error,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                ),
            )

    def mark_succeeded(self, job_key: str) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    finished_at = COALESCE(finished_at, ?),
                    updated_at = ?,
                    last_error = NULL
                WHERE job_key = ?
                """,
                (BulkJobStatus.SUCCEEDED.value, now, now, job_key),
            )

    def mark_failed(self, job_key: str, error: str | None = None) -> None:
        self._mark_terminal(job_key, BulkJobStatus.FAILED, error=error)

    def mark_cancelled(self, job_key: str, error: str | None = None) -> None:
        self._mark_terminal(job_key, BulkJobStatus.CANCELLED, error=error)

    def mark_unknown(self, job_key: str, error: str | None = None) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    monitor_attempts = monitor_attempts + 1,
                    updated_at = ?,
                    last_error = ?
                WHERE job_key = ? AND status != ?
                """,
                (
                    BulkJobStatus.UNKNOWN.value,
                    now,
                    error,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                ),
            )

    def record_monitor_attempt(self, job_key: str) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET monitor_attempts = monitor_attempts + 1, updated_at = ?
                WHERE job_key = ?
                """,
                (now, job_key),
            )

    def all_terminal(self) -> bool:
        statuses = _status_values(tuple(TERMINAL_BULK_JOB_STATUSES))
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM bulk_jobs
                WHERE status NOT IN ({self._placeholders(statuses)})
                """,
                statuses,
            ).fetchone()
        return int(row["count"]) == 0

    def jobs_for_wave(self, wave_id: str) -> list[BulkJobRecord]:
        return self._fetch_records(
            "wave_id = ?",
            [wave_id],
            "created_at, job_key",
        )

    def is_wave_ready(self, wave_id: str) -> bool:
        jobs = self.jobs_for_wave(wave_id)
        if not jobs:
            return False
        return all(job.status == BulkJobStatus.SUCCEEDED for job in jobs)

    def get_ready_waves(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT wave_id
                FROM bulk_jobs
                WHERE wave_id IS NOT NULL
                ORDER BY wave_id
                """
            ).fetchall()
        return [str(row["wave_id"]) for row in rows if self.is_wave_ready(str(row["wave_id"]))]

    def status_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM bulk_jobs
                GROUP BY status
                ORDER BY status
                """
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def reset_jobs_for_rerun(
        self,
        *,
        job_keys: Sequence[str] | None = None,
        statuses: Sequence[BulkJobStatus | str] | None = None,
        to_status: BulkJobStatus | str = BulkJobStatus.PENDING,
        clear_scheduler_ids: bool = True,
        clear_error: bool = False,
    ) -> int:
        target_status = _coerce_status(to_status)
        source_statuses = _coerce_statuses(
            tuple(statuses)
            if statuses is not None
            else (BulkJobStatus.FAILED, BulkJobStatus.UNKNOWN)
        )
        if not source_statuses:
            return 0

        reset_job_keys = None if job_keys is None else [str(job_key) for job_key in job_keys]
        if reset_job_keys is not None and not reset_job_keys:
            return 0

        now = _utcnow_iso()
        source_status_values = _status_values(tuple(source_statuses))
        set_clauses = [
            "status = ?",
            "submit_attempts = 0",
            "monitor_attempts = 0",
            "updated_at = ?",
        ]
        params: list[Any] = [target_status.value, now]

        if clear_scheduler_ids:
            set_clauses.extend(
                [
                    "scheduler_job_id = NULL",
                    "submitted_at = NULL",
                    "started_at = NULL",
                    "finished_at = NULL",
                    "submit_mode = 'single'",
                    "bulk_group_key = NULL",
                    "bulk_parent_job_id = NULL",
                    "bulk_index = NULL",
                    "scheduler_subjob_id = NULL",
                ]
            )
        if clear_error:
            set_clauses.append("last_error = NULL")

        where_clauses = [f"status IN ({self._placeholders(source_status_values)})"]
        params.extend(source_status_values)
        if reset_job_keys is not None:
            where_clauses.append(f"job_key IN ({self._placeholders(reset_job_keys)})")
            params.extend(reset_job_keys)

        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE bulk_jobs
                SET {", ".join(set_clauses)}
                WHERE {" AND ".join(where_clauses)}
                """,
                params,
            )
        return int(cursor.rowcount)

    def refresh_completed_jobs_from_outputs(
        self,
        *,
        include_awaiting_operator: bool = False,
    ) -> None:
        now = _utcnow_iso()
        excluded_statuses = [
            BulkJobStatus.SUCCEEDED.value,
            BulkJobStatus.PREPARED.value,
        ]
        if not include_awaiting_operator:
            excluded_statuses.append(BulkJobStatus.AWAITING_OPERATOR.value)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM bulk_jobs "
                f"WHERE status NOT IN ({self._placeholders(excluded_statuses)})",
                excluded_statuses,
            ).fetchall()
            for row in rows:
                expected_outputs = _json_loads_paths(row["expected_outputs_json"])
                work_dir = Path(str(row["work_dir"]))
                if not _outputs_are_complete(expected_outputs, work_dir=work_dir):
                    continue

                conn.execute(
                    """
                    UPDATE bulk_jobs
                    SET
                        status = ?,
                        finished_at = COALESCE(finished_at, ?),
                        updated_at = ?,
                        last_error = NULL
                    WHERE job_key = ?
                    """,
                    (
                        BulkJobStatus.SUCCEEDED.value,
                        now,
                        now,
                        row["job_key"],
                    ),
                )

    def _mark_status(self, job_key: str, status: BulkJobStatus) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET status = ?, updated_at = ?
                WHERE job_key = ? AND status != ?
                """,
                (
                    status.value,
                    now,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                ),
            )

    def _mark_terminal(
        self,
        job_key: str,
        status: BulkJobStatus,
        *,
        error: str | None,
    ) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    finished_at = COALESCE(finished_at, ?),
                    updated_at = ?,
                    last_error = ?
                WHERE job_key = ? AND status != ?
                """,
                (
                    status.value,
                    now,
                    now,
                    error,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                ),
            )

    def _fetch_records(
        self,
        where_clause: str,
        params: Sequence[Any],
        order_clause: str,
    ) -> list[BulkJobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM bulk_jobs
                WHERE {where_clause}
                ORDER BY {order_clause}
                """,
                list(params),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def _count_records(self, where_clause: str, params: Sequence[Any]) -> int:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM bulk_jobs
                WHERE {where_clause}
                """,
                list(params),
            ).fetchone()
        return int(row["count"])

    @staticmethod
    def _placeholders(values: Sequence[Any]) -> str:
        return ", ".join("?" for _ in values)


def _record_from_row(row: sqlite3.Row) -> BulkJobRecord:
    try:
        status = BulkJobStatus(str(row["status"]))
    except ValueError:
        status = BulkJobStatus.UNKNOWN

    return BulkJobRecord(
        job_key=str(row["job_key"]),
        wave_id=row["wave_id"],
        target_id=row["target_id"],
        stage_id=row["stage_id"],
        status=status,
        work_dir=Path(str(row["work_dir"])),
        scheduler_job_id=row["scheduler_job_id"],
        submit_attempts=int(row["submit_attempts"]),
        monitor_attempts=int(row["monitor_attempts"]),
        command_args=_json_loads_dict(row["command_args_json"]),
        expected_outputs=_json_loads_paths(row["expected_outputs_json"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        submitted_at=row["submitted_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        last_error=row["last_error"],
        priority=int(row["priority"]),
        max_submit_attempts=int(row["max_submit_attempts"]),
        submit_mode=str(row["submit_mode"] or "single"),
        bulk_group_key=row["bulk_group_key"],
        bulk_parent_job_id=row["bulk_parent_job_id"],
        bulk_index=None if row["bulk_index"] is None else int(row["bulk_index"]),
        scheduler_subjob_id=row["scheduler_subjob_id"],
        execution_profile_block=row["execution_profile_block"],
        hpc_profile_block=row["hpc_profile_block"],
        spec_hash=row["spec_hash"],
        input_digest=row["input_digest"],
        code_digest=row["code_digest"],
        environment_digest=row["environment_digest"],
        prepared_at=row["prepared_at"],
        job_name=row["job_name"],
        job_comment=row["job_comment"],
        desired_state=_coerce_desired_state(row["desired_state"]),
        cancel_requested_at=row["cancel_requested_at"],
        cancel_requested_by=row["cancel_requested_by"],
        cancel_reason=row["cancel_reason"],
        cancel_attempts=int(row["cancel_attempts"]),
        cancel_dispatch_started_at=row["cancel_dispatch_started_at"],
        cancel_outcome=_coerce_cancel_outcome(row["cancel_outcome"]),
        cancel_outcome_at=row["cancel_outcome_at"],
        cancel_last_error=row["cancel_last_error"],
    )
