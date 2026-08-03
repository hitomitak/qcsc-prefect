"""Durable journal and identity helpers for Qiskit Runtime submissions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

QISKIT_SUBMISSION_TAG_PREFIX = "qcsc-prefect:"
QISKIT_SUBMISSION_TAG_DIGEST_LENGTH = 24
QISKIT_MAX_JOB_TAGS = 8
QISKIT_MAX_JOB_TAG_LENGTH = 86


class QiskitDurableSubmissionError(RuntimeError):
    """Base error for durable Qiskit Runtime submission."""


class QiskitSpecHashMismatchError(QiskitDurableSubmissionError):
    """Raised when one submission key is reused with a different spec hash."""


class QiskitJournalConflictError(QiskitDurableSubmissionError):
    """Raised when a journal update conflicts with already-persisted evidence."""


class QiskitRecoveryPending(QiskitDurableSubmissionError):
    """Raised while a PREPARED submission remains inside its visibility grace."""

    def __init__(self, *, submission_key: str, retry_after_seconds: float) -> None:
        self.submission_key = submission_key
        self.retry_after_seconds = max(0.0, float(retry_after_seconds))
        super().__init__(
            f"Qiskit submission {submission_key!r} is awaiting Runtime visibility; "
            f"retry after {self.retry_after_seconds:g} seconds."
        )


class QiskitOperatorActionRequired(QiskitDurableSubmissionError):
    """Raised when automatic Qiskit submit-or-attach cannot proceed safely."""

    def __init__(self, *, submission_key: str, reason: str) -> None:
        self.submission_key = submission_key
        self.reason = reason
        super().__init__(f"Qiskit submission {submission_key!r} requires operator action: {reason}")


class QiskitSubmissionStatus(str, Enum):
    """Durable lifecycle states for one Qiskit Runtime submission."""

    PREPARED = "PREPARED"
    SUBMITTED = "SUBMITTED"
    AWAITING_OPERATOR = "AWAITING_OPERATOR"


@dataclass(frozen=True)
class QiskitSubmissionRecord:
    """One durable Qiskit Runtime journal record."""

    submission_key: str
    spec_hash: str
    stable_tag: str
    status: QiskitSubmissionStatus
    prepared_at: datetime
    updated_at: datetime
    job_id: str | None = None
    job_reference: dict[str, Any] | None = None
    last_error: str | None = None
    held_at: datetime | None = None


@dataclass(frozen=True)
class QiskitJobIdentity:
    """Inputs for one Qiskit Runtime tag search."""

    stable_tag: str
    backend_name: str
    search_start: datetime
    search_end: datetime


@dataclass(frozen=True)
class QiskitJobCandidate:
    """Normalized Qiskit Runtime candidate returned to shared recovery logic."""

    job_id: str
    job: Any
    identity_matches: bool = True
    metadata_error: str | None = None


def build_qiskit_submission_tag(*, submission_key: str, spec_hash: str) -> str:
    """Build a short stable tag without exposing the key or spec hash."""

    normalized_key = _required_text(submission_key, field_name="submission_key")
    normalized_hash = _required_text(spec_hash, field_name="spec_hash")
    digest = hashlib.sha256(
        f"qiskit-submission-v1\0{normalized_key}\0{normalized_hash}".encode()
    ).hexdigest()[:QISKIT_SUBMISSION_TAG_DIGEST_LENGTH]
    return f"{QISKIT_SUBMISSION_TAG_PREFIX}{digest}"


def with_qiskit_submission_tags(
    options: Mapping[str, Any] | None,
    *,
    stable_tag: str,
    submission_tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return primitive options with validated pre-submit Runtime job tags."""

    if isinstance(submission_tags, str | bytes):
        raise ValueError("submission_tags must be an iterable of strings.")
    merged_options = dict(options or {})
    raw_environment = merged_options.get("environment")
    if raw_environment is None:
        environment: dict[str, Any] = {}
    elif isinstance(raw_environment, Mapping):
        environment = dict(raw_environment)
    else:
        raise ValueError("options.environment must be a mapping in durable submit mode.")

    raw_existing_tags = environment.get("job_tags")
    if raw_existing_tags is None:
        existing_tags: list[str] = []
    elif isinstance(raw_existing_tags, str | bytes):
        raise ValueError("options.environment.job_tags must be an iterable of strings.")
    else:
        existing_tags = [str(tag) for tag in raw_existing_tags]

    combined = [*existing_tags, *(submission_tags or ()), stable_tag]
    tags = list(dict.fromkeys(_validated_job_tag(tag) for tag in combined))
    if len(tags) > QISKIT_MAX_JOB_TAGS:
        raise ValueError(
            f"Qiskit Runtime supports at most {QISKIT_MAX_JOB_TAGS} job tags; "
            f"durable submission would use {len(tags)}."
        )
    environment["job_tags"] = tags
    merged_options["environment"] = environment
    return merged_options


class QiskitSubmissionJournal:
    """SQLite journal for resumable Qiskit Runtime submission."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS qiskit_submissions (
                    submission_key TEXT PRIMARY KEY,
                    spec_hash TEXT NOT NULL,
                    stable_tag TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prepared_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    job_id TEXT,
                    job_reference_json TEXT,
                    last_error TEXT,
                    held_at TEXT
                )
                """
            )

    def get(self, submission_key: str) -> QiskitSubmissionRecord | None:
        key = _required_text(submission_key, field_name="submission_key")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM qiskit_submissions WHERE submission_key = ?",
                (key,),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def prepare(
        self,
        *,
        submission_key: str,
        spec_hash: str,
        stable_tag: str,
    ) -> tuple[QiskitSubmissionRecord, bool]:
        """Atomically create PREPARED, returning whether this caller claimed submit."""

        key = _required_text(submission_key, field_name="submission_key")
        incoming_hash = _required_text(spec_hash, field_name="spec_hash")
        incoming_tag = _validated_job_tag(stable_tag)
        now = _utc_now()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM qiskit_submissions WHERE submission_key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                record = _record_from_row(row)
                _require_matching_spec(record, incoming_hash=incoming_hash)
                if record.stable_tag != incoming_tag:
                    raise QiskitJournalConflictError(
                        f"Qiskit submission {key!r} has a different stored stable tag."
                    )
                return record, False

            connection.execute(
                """
                INSERT INTO qiskit_submissions (
                    submission_key, spec_hash, stable_tag, status,
                    prepared_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    incoming_hash,
                    incoming_tag,
                    QiskitSubmissionStatus.PREPARED.value,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM qiskit_submissions WHERE submission_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                raise QiskitJournalConflictError(
                    f"Qiskit submission {key!r} disappeared after PREPARED claim."
                )
            return _record_from_row(row), True

    def mark_submitted(
        self,
        *,
        submission_key: str,
        spec_hash: str,
        job_id: str,
        job_reference: Mapping[str, Any],
    ) -> QiskitSubmissionRecord:
        """Persist a job ID/reference without overwriting conflicting evidence."""

        key = _required_text(submission_key, field_name="submission_key")
        incoming_hash = _required_text(spec_hash, field_name="spec_hash")
        incoming_job_id = _required_text(job_id, field_name="job_id")
        serialized_reference = json.dumps(
            dict(job_reference),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        now = _utc_now()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM qiskit_submissions WHERE submission_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                raise QiskitJournalConflictError(
                    f"Qiskit submission {key!r} has no PREPARED journal row."
                )
            record = _record_from_row(row)
            _require_matching_spec(record, incoming_hash=incoming_hash)
            if record.job_id is not None and record.job_id != incoming_job_id:
                raise QiskitJournalConflictError(
                    f"Qiskit submission {key!r} already stores a different job ID."
                )
            if record.status == QiskitSubmissionStatus.AWAITING_OPERATOR:
                raise QiskitJournalConflictError(
                    f"Qiskit submission {key!r} is in durable operator hold."
                )

            connection.execute(
                """
                UPDATE qiskit_submissions
                SET status = ?, job_id = ?, job_reference_json = ?,
                    last_error = NULL, held_at = NULL, updated_at = ?
                WHERE submission_key = ?
                """,
                (
                    QiskitSubmissionStatus.SUBMITTED.value,
                    incoming_job_id,
                    serialized_reference,
                    now.isoformat(),
                    key,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM qiskit_submissions WHERE submission_key = ?",
                (key,),
            ).fetchone()
        if updated is None:
            raise QiskitJournalConflictError(
                f"Qiskit submission {key!r} disappeared after job ID persistence."
            )
        return _record_from_row(updated)

    def record_prepared_error(self, submission_key: str, error: str) -> None:
        key = _required_text(submission_key, field_name="submission_key")
        now = _utc_now().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE qiskit_submissions
                SET last_error = ?, updated_at = ?
                WHERE submission_key = ? AND status = ?
                """,
                (str(error), now, key, QiskitSubmissionStatus.PREPARED.value),
            )

    def mark_awaiting_operator(
        self,
        submission_key: str,
        reason: str,
    ) -> QiskitSubmissionRecord:
        key = _required_text(submission_key, field_name="submission_key")
        normalized_reason = _required_text(reason, field_name="reason")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM qiskit_submissions WHERE submission_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                raise QiskitJournalConflictError(f"Qiskit submission {key!r} does not exist.")
            record = _record_from_row(row)
            if record.status == QiskitSubmissionStatus.SUBMITTED:
                raise QiskitJournalConflictError(
                    f"Qiskit submission {key!r} already has a durable job ID."
                )
            connection.execute(
                """
                UPDATE qiskit_submissions
                SET status = ?, last_error = ?, held_at = ?, updated_at = ?
                WHERE submission_key = ?
                """,
                (
                    QiskitSubmissionStatus.AWAITING_OPERATOR.value,
                    normalized_reason,
                    now.isoformat(),
                    now.isoformat(),
                    key,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM qiskit_submissions WHERE submission_key = ?",
                (key,),
            ).fetchone()
        if updated is None:
            raise QiskitJournalConflictError(
                f"Qiskit submission {key!r} disappeared while entering operator hold."
            )
        return _record_from_row(updated)


class QiskitIdentityRecoveryRuntime:
    """Per-provider identity lookup for Qiskit Runtime jobs."""

    def __init__(self, service: Any) -> None:
        self.service = service

    def find_candidates_by_identity(
        self,
        identity: QiskitJobIdentity,
    ) -> list[QiskitJobCandidate]:
        """Return every tag-filtered candidate without selecting one."""

        jobs = self.service.jobs(
            limit=None,
            backend_name=identity.backend_name,
            job_tags=[identity.stable_tag],
            created_after=identity.search_start.astimezone(),
            created_before=identity.search_end.astimezone(),
            descending=False,
        )
        candidates: dict[str, QiskitJobCandidate] = {}
        for job in jobs:
            job_id = _job_value(job, "job_id")
            if job_id is None:
                raise QiskitJournalConflictError(
                    "Qiskit Runtime identity search returned a job without a job ID."
                )
            normalized_job_id = str(job_id)
            tags = _job_tags(job)
            creation_date = _job_datetime(job, "creation_date")
            backend_name = _job_backend_name(job)
            metadata_error: str | None = None
            if backend_name != identity.backend_name:
                metadata_error = f"job {normalized_job_id} backend does not match"
            elif creation_date is None:
                metadata_error = f"job {normalized_job_id} has no creation date"
            elif creation_date.astimezone(timezone.utc) < identity.search_start.astimezone(
                timezone.utc
            ):
                metadata_error = f"job {normalized_job_id} predates the recovery window"
            elif creation_date.astimezone(timezone.utc) > identity.search_end.astimezone(
                timezone.utc
            ):
                metadata_error = f"job {normalized_job_id} is beyond the recovery clock-skew window"
            candidates.setdefault(
                normalized_job_id,
                QiskitJobCandidate(
                    job_id=normalized_job_id,
                    job=job,
                    identity_matches=identity.stable_tag in tags,
                    metadata_error=metadata_error,
                ),
            )
        return list(candidates.values())


def _required_text(value: Any, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _validated_job_tag(value: Any) -> str:
    tag = _required_text(value, field_name="Qiskit job tag")
    if len(tag) > QISKIT_MAX_JOB_TAG_LENGTH:
        raise ValueError(f"Qiskit job tags must be at most {QISKIT_MAX_JOB_TAG_LENGTH} characters.")
    return tag


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _record_from_row(row: sqlite3.Row) -> QiskitSubmissionRecord:
    reference_json = row["job_reference_json"]
    reference = json.loads(reference_json) if reference_json else None
    return QiskitSubmissionRecord(
        submission_key=str(row["submission_key"]),
        spec_hash=str(row["spec_hash"]),
        stable_tag=str(row["stable_tag"]),
        status=QiskitSubmissionStatus(str(row["status"])),
        prepared_at=_parse_datetime(str(row["prepared_at"])) or _utc_now(),
        updated_at=_parse_datetime(str(row["updated_at"])) or _utc_now(),
        job_id=str(row["job_id"]) if row["job_id"] is not None else None,
        job_reference=reference,
        last_error=str(row["last_error"]) if row["last_error"] is not None else None,
        held_at=_parse_datetime(row["held_at"]),
    )


def _require_matching_spec(
    record: QiskitSubmissionRecord,
    *,
    incoming_hash: str,
) -> None:
    if record.spec_hash != incoming_hash:
        raise QiskitSpecHashMismatchError(
            f"Qiskit submission {record.submission_key!r} already has spec hash "
            f"{record.spec_hash!r}; incoming hash is {incoming_hash!r}. Use a new "
            "submission_key for an intentional spec change."
        )


def _job_value(job: Any, name: str) -> Any | None:
    value = getattr(job, name, None)
    if callable(value):
        value = value()
    return value


def _job_tags(job: Any) -> set[str]:
    value = _job_value(job, "tags")
    if value is None:
        return set()
    return {str(tag) for tag in value}


def _job_datetime(job: Any, name: str) -> datetime | None:
    value = _job_value(job, name)
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.now().astimezone().tzinfo or timezone.utc)
    return value


def _job_backend_name(job: Any) -> str | None:
    backend = _job_value(job, "backend")
    if backend is None:
        return None
    for attr in ("name", "backend_name"):
        value = _job_value(backend, attr)
        if value is not None:
            return str(value)
    if isinstance(backend, str):
        return backend
    return None
