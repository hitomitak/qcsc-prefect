from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class BulkJobStatus(str, Enum):
    """Normalized lifecycle status for one logical bulk job."""

    PENDING = "PENDING"
    SUBMIT_DEFERRED = "SUBMIT_DEFERRED"
    PREPARED = "PREPARED"
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"
    AWAITING_OPERATOR = "AWAITING_OPERATOR"

    @property
    def is_terminal(self) -> bool:
        """Return whether this status is a terminal state."""

        return self in TERMINAL_BULK_JOB_STATUSES

    @property
    def is_active(self) -> bool:
        """Return whether this status represents an accepted scheduler job."""

        return self in ACTIVE_BULK_JOB_STATUSES

    @property
    def is_submit_candidate(self) -> bool:
        """Return whether this status may be submitted in a future refill."""

        return self in SUBMIT_CANDIDATE_BULK_JOB_STATUSES

    @property
    def is_recovery_candidate(self) -> bool:
        """Return whether this status may be reconciled automatically."""

        return self in RECOVERY_CANDIDATE_BULK_JOB_STATUSES


class BulkJobDesiredState(str, Enum):
    """Operator-requested target state for one logical bulk job."""

    RUN = "RUN"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"


class BulkCancelOutcome(str, Enum):
    """Durable outcome of processing one explicit cancellation request."""

    NOT_SUBMITTED = "NOT_SUBMITTED"
    DISPATCHING = "DISPATCHING"
    REQUEST_ACCEPTED = "REQUEST_ACCEPTED"
    ALREADY_TERMINAL = "ALREADY_TERMINAL"
    NOT_FOUND = "NOT_FOUND"
    TEMPORARY_FAILURE = "TEMPORARY_FAILURE"
    REJECTED = "REJECTED"


TERMINAL_BULK_JOB_STATUSES = frozenset(
    {
        BulkJobStatus.SUCCEEDED,
        BulkJobStatus.FAILED,
        BulkJobStatus.CANCELLED,
    }
)
ACTIVE_BULK_JOB_STATUSES = frozenset(
    {
        BulkJobStatus.SUBMITTED,
        BulkJobStatus.QUEUED,
        BulkJobStatus.RUNNING,
    }
)
SUBMIT_CANDIDATE_BULK_JOB_STATUSES = frozenset(
    {
        BulkJobStatus.PENDING,
        BulkJobStatus.SUBMIT_DEFERRED,
    }
)
RECOVERY_CANDIDATE_BULK_JOB_STATUSES = frozenset(
    {
        BulkJobStatus.PREPARED,
        BulkJobStatus.UNKNOWN,
    }
)


@dataclass(frozen=True)
class BulkJobSpec:
    """Desired bulk job registration payload.

    ``job_key`` is the stable idempotency key and must be unique within one
    registry.
    """

    job_key: str
    work_dir: Path
    command_args: dict[str, Any] = field(default_factory=dict)
    wave_id: str | None = None
    target_id: str | None = None
    stage_id: str | None = None
    priority: int = 0
    expected_outputs: list[Path] = field(default_factory=list)
    max_submit_attempts: int = 5
    execution_profile_block: str | None = None
    hpc_profile_block: str | None = None
    spec_hash: str | None = None
    input_digest: str | None = None
    code_digest: str | None = None
    environment_digest: str | None = None
    job_name: str | None = None
    job_comment: str | None = None

    def __post_init__(self) -> None:
        if not self.job_key.strip():
            raise ValueError("BulkJobSpec.job_key must be non-empty.")
        for field_name in (
            "execution_profile_block",
            "hpc_profile_block",
            "spec_hash",
            "input_digest",
            "code_digest",
            "environment_digest",
            "job_name",
            "job_comment",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = str(value)
            if not normalized.strip():
                raise ValueError(f"BulkJobSpec.{field_name} must be non-empty when set.")
            object.__setattr__(self, field_name, normalized)
        object.__setattr__(self, "work_dir", Path(self.work_dir))
        object.__setattr__(
            self,
            "stage_id",
            None if self.stage_id is None else str(self.stage_id),
        )
        object.__setattr__(
            self,
            "expected_outputs",
            [Path(path) for path in self.expected_outputs],
        )
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "max_submit_attempts", int(self.max_submit_attempts))


@dataclass(frozen=True)
class BulkJobRecord:
    """Persisted state for one logical bulk job."""

    job_key: str
    wave_id: str | None
    target_id: str | None
    status: BulkJobStatus
    work_dir: Path
    scheduler_job_id: str | None
    submit_attempts: int
    monitor_attempts: int
    command_args: dict[str, Any]
    expected_outputs: list[Path]
    created_at: str
    updated_at: str
    submitted_at: str | None
    started_at: str | None
    finished_at: str | None
    last_error: str | None
    priority: int = 0
    max_submit_attempts: int = 5
    stage_id: str | None = None
    submit_mode: str = "single"
    bulk_group_key: str | None = None
    bulk_parent_job_id: str | None = None
    bulk_index: int | None = None
    scheduler_subjob_id: str | None = None
    execution_profile_block: str | None = None
    hpc_profile_block: str | None = None
    spec_hash: str | None = None
    input_digest: str | None = None
    code_digest: str | None = None
    environment_digest: str | None = None
    prepared_at: str | None = None
    job_name: str | None = None
    job_comment: str | None = None
    desired_state: BulkJobDesiredState = BulkJobDesiredState.RUN
    cancel_requested_at: str | None = None
    cancel_requested_by: str | None = None
    cancel_reason: str | None = None
    cancel_attempts: int = 0
    cancel_dispatch_started_at: str | None = None
    cancel_outcome: BulkCancelOutcome | None = None
    cancel_outcome_at: str | None = None
    cancel_last_error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def is_active(self) -> bool:
        return self.status.is_active

    @property
    def is_submit_candidate(self) -> bool:
        return self.status.is_submit_candidate

    @property
    def is_recovery_candidate(self) -> bool:
        return self.status.is_recovery_candidate

    @property
    def effective_scheduler_job_id(self) -> str | None:
        return self.scheduler_subjob_id or self.scheduler_job_id


def effective_execution_profile_block(
    job: BulkJobSpec | BulkJobRecord,
    default_block: str,
) -> str:
    """Return the per-job execution profile block or the runner/API default."""

    return job.execution_profile_block if job.execution_profile_block is not None else default_block


def effective_hpc_profile_block(
    job: BulkJobSpec | BulkJobRecord,
    default_block: str,
) -> str:
    """Return the per-job HPC profile block or the runner/API default."""

    return job.hpc_profile_block if job.hpc_profile_block is not None else default_block


@dataclass(frozen=True)
class SubmittedJob:
    """Scheduler identity returned after a bulk job is accepted."""

    job_key: str
    scheduler_job_id: str
    status: BulkJobStatus
    work_dir: Path


@dataclass(frozen=True)
class BulkRunResult:
    """Summary returned by a future bulk run orchestration API."""

    total_jobs: int
    status_counts: dict[str, int]
    succeeded: int
    failed: int
    cancelled: int
    submit_deferred: int
    unknown: int
    registry_path: Path
    failed_jobs: list[str]
    prepared: int = 0
    awaiting_operator: int = 0
    operator_action_required_jobs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BulkTickResult:
    """Summary returned by one non-blocking bulk runner tick."""

    submitted: list[SubmittedJob]
    monitored: dict[str, BulkJobStatus]
    status_counts: dict[str, int]
    registry_path: Path
