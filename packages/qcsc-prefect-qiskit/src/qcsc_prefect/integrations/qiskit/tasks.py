"""Prefect task utilities for native Qiskit execution."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict

from anyio.to_thread import run_sync
from prefect import get_run_logger, task
from qcsc_prefect.integrations.qiskit.artifacts import (
    create_qiskit_estimator_metadata_artifact,
    create_qiskit_estimator_result_artifact,
    create_qiskit_execution_markdown_artifact,
    create_qiskit_sampler_metadata_artifact,
    create_qiskit_sampler_result_artifact,
)
from qcsc_prefect.integrations.qiskit.blocks import QiskitRuntimeConfig
from qcsc_prefect.integrations.qiskit.cache import (
    build_qiskit_cache_payload,
    qiskit_result_fetch_cache_key,
)
from qcsc_prefect.integrations.qiskit.durable import (
    QiskitIdentityRecoveryRuntime,
    QiskitJobIdentity,
    QiskitOperatorActionRequired,
    QiskitRecoveryPending,
    QiskitSubmissionJournal,
    QiskitSubmissionRecord,
    QiskitSubmissionStatus,
    build_qiskit_submission_tag,
    with_qiskit_submission_tags,
)
from qcsc_prefect.integrations.qiskit.metadata import collect_qiskit_execution_metadata


def _sampler_class():
    from qiskit_ibm_runtime import SamplerV2

    return SamplerV2


def _estimator_class():
    from qiskit_ibm_runtime import EstimatorV2

    return EstimatorV2


class QiskitSamplerTaskError(RuntimeError):
    """Raised when the native Qiskit Sampler task cannot complete."""


class QiskitEstimatorTaskError(RuntimeError):
    """Raised when the native Qiskit Estimator task cannot complete."""


class QiskitJobFetchTaskError(RuntimeError):
    """Raised when an existing native Qiskit Runtime job cannot be fetched."""


class QiskitJobReference(TypedDict, total=False):
    """Reference to an already-submitted native Qiskit Runtime job."""

    program_type: str
    primitive: str
    backend_name: str
    job_id: str
    runtime_block_name: str | None
    shots: int
    precision: float
    input_digest: str
    options: dict[str, Any]
    submission_key: str
    spec_hash: str
    job_tags: list[str]
    journal_path: str


async def _resolve_loaded_block(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _load_runtime_config(
    runtime_block_name: str,
    *,
    error_cls: type[RuntimeError] = QiskitSamplerTaskError,
) -> Any:
    try:
        return await _resolve_loaded_block(QiskitRuntimeConfig.load(runtime_block_name))
    except Exception as exc:
        raise error_cls(
            "Failed to load QiskitRuntimeConfig block "
            f"{runtime_block_name!r} ({type(exc).__name__})."
        ) from None


async def _resolve_runtime_config(
    runtime_block_name: str | None,
    *,
    runtime_config: Any | None = None,
    error_cls: type[RuntimeError] = QiskitSamplerTaskError,
) -> Any:
    if runtime_config is not None:
        if runtime_block_name is not None:
            raise error_cls("Pass either runtime_block_name or runtime_config, not both.")
        return runtime_config
    if runtime_block_name is None:
        raise error_cls("Either runtime_block_name or runtime_config is required.")
    return await _load_runtime_config(runtime_block_name, error_cls=error_cls)


async def _load_runtime_backend(
    runtime_block_name: str | None,
    *,
    runtime_config: Any | None = None,
    error_cls: type[RuntimeError] = QiskitSamplerTaskError,
) -> tuple[Any, Any, str]:
    runtime_config = await _resolve_runtime_config(
        runtime_block_name,
        runtime_config=runtime_config,
        error_cls=error_cls,
    )
    backend = runtime_config.get_backend()
    backend_name = _backend_name(backend) or runtime_config.backend_name
    return runtime_config, backend, backend_name


def _logger() -> logging.Logger:
    try:
        return get_run_logger()
    except Exception:
        return logging.getLogger(__name__)


def _job_id(job: Any) -> str | None:
    value = getattr(job, "job_id", None)
    if callable(value):
        value = value()
    if value is None:
        return None
    return str(value)


def _require_job_id(job: Any, *, primitive_name: str, error_cls: type[RuntimeError]) -> str:
    """Extract and validate a Qiskit job ID."""

    job_id = _job_id(job)
    if job_id is None:
        raise error_cls(f"Submitted native Qiskit {primitive_name} job did not expose a job ID.")
    return job_id


def _backend_name(backend: Any) -> str | None:
    for attr in ("name", "backend_name"):
        value = getattr(backend, attr, None)
        if callable(value):
            value = value()
        if value is not None:
            return str(value)
    if isinstance(backend, str):
        return backend
    return None


def _metadata_options(
    options: dict[str, Any] | None,
    *,
    shots: int | None = None,
    precision: float | None = None,
) -> dict[str, Any]:
    metadata_options: dict[str, Any] = dict(options or {})
    if shots is not None or precision is not None:
        params = metadata_options.get("params")
        if isinstance(params, dict):
            params = dict(params)
        else:
            params = {}
        if shots is not None:
            params.setdefault("shots", shots)
        if precision is not None:
            params.setdefault("precision", precision)
        metadata_options["params"] = params
    return metadata_options


def _job_reference(
    *,
    program_type: str,
    backend_name: str,
    job_id: str,
    runtime_block_name: str | None = None,
    shots: int | None = None,
    precision: float | None = None,
    input_digest: str | None = None,
    options: Mapping[str, Any] | None = None,
    submission_key: str | None = None,
    spec_hash: str | None = None,
    job_tags: Iterable[str] | None = None,
    journal_path: str | Path | None = None,
) -> QiskitJobReference:
    """Create a serializable job reference."""

    reference: QiskitJobReference = {
        "program_type": program_type,
        "primitive": program_type,
        "backend_name": backend_name,
        "job_id": job_id,
        "runtime_block_name": runtime_block_name,
    }
    if shots is not None:
        reference["shots"] = shots
    if precision is not None:
        reference["precision"] = precision
    if input_digest is not None:
        reference["input_digest"] = input_digest
    if submission_key is not None:
        reference["submission_key"] = submission_key
    if spec_hash is not None:
        reference["spec_hash"] = spec_hash
    if job_tags is not None:
        reference["job_tags"] = list(job_tags)
    if journal_path is not None:
        reference["journal_path"] = str(journal_path)
    options_summary = _options_summary(options)
    if options_summary is not None:
        reference["options"] = options_summary
    return reference


def _options_summary(options: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if options is None:
        return None
    try:
        payload = build_qiskit_cache_payload(program_type="options", options=options)
    except (TypeError, ValueError):
        return None
    summary = payload.get("options")
    if isinstance(summary, dict):
        return summary
    return None


def _reference_value(
    job_reference: Mapping[str, Any] | None,
    key: str,
    explicit: Any | None = None,
) -> Any | None:
    """Return explicit value or the corresponding job-reference value."""

    if explicit is not None:
        return explicit
    if job_reference is None:
        return None
    return job_reference.get(key)


def _resolve_job_reference(
    *,
    job_reference: Mapping[str, Any] | None,
    job_id: str | None,
    primitive: str | None,
    backend_name: str | None,
    shots: int | None,
    precision: float | None,
) -> QiskitJobReference:
    """Resolve explicit fetch parameters and an optional job reference."""

    resolved_job_id = _reference_value(job_reference, "job_id", job_id)
    if resolved_job_id is None:
        raise QiskitJobFetchTaskError("A Qiskit Runtime job ID is required to fetch a result.")

    resolved: QiskitJobReference = {"job_id": str(resolved_job_id)}
    resolved_primitive = _reference_value(job_reference, "primitive", primitive)
    if resolved_primitive is None:
        resolved_primitive = _reference_value(job_reference, "program_type")
    if resolved_primitive is not None:
        resolved["primitive"] = str(resolved_primitive)
        resolved["program_type"] = str(resolved_primitive)
    resolved_backend_name = _reference_value(job_reference, "backend_name", backend_name)
    if resolved_backend_name is not None:
        resolved["backend_name"] = str(resolved_backend_name)
    resolved_shots = _reference_value(job_reference, "shots", shots)
    if resolved_shots is not None:
        resolved["shots"] = resolved_shots
    resolved_precision = _reference_value(job_reference, "precision", precision)
    if resolved_precision is not None:
        resolved["precision"] = resolved_precision
    resolved_input_digest = _reference_value(job_reference, "input_digest")
    if resolved_input_digest is not None:
        resolved["input_digest"] = str(resolved_input_digest)
    return resolved


def _result_response(
    *,
    primitive: str | None,
    backend_name: str | None,
    job_id: str | None,
    result: Any,
    metadata: Any,
    shots: int | None = None,
    precision: float | None = None,
    include_shots: bool = False,
    include_precision: bool = False,
    input_digest: str | None = None,
) -> dict[str, Any]:
    """Build the structured task response."""

    response: dict[str, Any] = {
        "primitive": primitive,
        "backend_name": backend_name,
        "job_id": job_id,
        "result": result,
        "metadata": metadata,
    }
    if include_shots or shots is not None:
        response["shots"] = shots
    if include_precision or precision is not None:
        response["precision"] = precision
    if input_digest is not None:
        response["input_digest"] = input_digest
    return response


def _create_sampler(*, backend: Any, backend_name: str, options: dict[str, Any] | None) -> Any:
    sampler_kwargs: dict[str, Any] = {"mode": backend}
    if options is not None:
        sampler_kwargs["options"] = options

    try:
        return _sampler_class()(**sampler_kwargs)
    except Exception as exc:
        raise QiskitSamplerTaskError(
            "Failed to create native SamplerV2 "
            f"for backend {backend_name!r} ({type(exc).__name__})."
        ) from None


def _create_estimator(*, backend: Any, backend_name: str, options: dict[str, Any] | None) -> Any:
    estimator_kwargs: dict[str, Any] = {"mode": backend}
    if options is not None:
        estimator_kwargs["options"] = options

    try:
        return _estimator_class()(**estimator_kwargs)
    except Exception as exc:
        raise QiskitEstimatorTaskError(
            "Failed to create native EstimatorV2 "
            f"for backend {backend_name!r} ({type(exc).__name__})."
        ) from None


def _run_primitive_job(
    *,
    primitive: Any,
    pubs: list[Any],
    primitive_name: str,
    error_cls: type[RuntimeError],
    backend_name: str,
    shots: int | None = None,
    precision: float | None = None,
    include_shots: bool = False,
    include_precision: bool = False,
) -> Any:
    run_kwargs: dict[str, Any] = {}
    if include_shots or shots is not None:
        run_kwargs["shots"] = shots
    if include_precision or precision is not None:
        run_kwargs["precision"] = precision
    try:
        return primitive.run(pubs, **run_kwargs)
    except Exception as exc:
        raise error_cls(
            f"Failed to submit native Qiskit {primitive_name} job "
            f"to backend {backend_name!r} ({type(exc).__name__})."
        ) from None


def _submit_sampler_job(
    *,
    sampler: Any,
    pubs: list[Any],
    shots: int | None,
    backend_name: str,
) -> Any:
    return _run_primitive_job(
        primitive=sampler,
        pubs=pubs,
        primitive_name="SamplerV2",
        error_cls=QiskitSamplerTaskError,
        backend_name=backend_name,
        shots=shots,
        include_shots=True,
    )


def _submit_estimator_job(
    *,
    estimator: Any,
    pubs: list[Any],
    precision: float | None,
    backend_name: str,
) -> Any:
    return _run_primitive_job(
        primitive=estimator,
        pubs=pubs,
        primitive_name="EstimatorV2",
        error_cls=QiskitEstimatorTaskError,
        backend_name=backend_name,
        precision=precision,
        include_precision=True,
    )


async def _wait_for_primitive_result(
    *,
    job: Any,
    job_id: str | None,
    backend_name: str,
    primitive_name: str,
    error_cls: type[RuntimeError],
) -> Any:
    try:
        return await run_sync(job.result)
    except Exception as exc:
        raise error_cls(
            f"Failed while waiting for native Qiskit {primitive_name} job result "
            f"(job_id={job_id!r}, backend={backend_name!r}, {type(exc).__name__})."
        ) from None


def _runtime_config_source(runtime_block_name: str | None) -> str:
    if runtime_block_name is None:
        return "runtime_config argument"
    return f"block {runtime_block_name!r}"


def _get_service(runtime_config: Any, *, runtime_block_name: str | None) -> Any:
    """Create a native QiskitRuntimeService from runtime configuration."""

    try:
        return runtime_config.get_service()
    except Exception as exc:
        raise QiskitJobFetchTaskError(
            "Failed to create native QiskitRuntimeService "
            f"from {_runtime_config_source(runtime_block_name)} ({type(exc).__name__})."
        ) from None


def _fetch_existing_job(*, service: Any, job_id: str) -> Any:
    """Fetch an existing Qiskit Runtime job by ID."""

    try:
        return service.job(job_id)
    except Exception as exc:
        raise QiskitJobFetchTaskError(
            f"Failed to fetch native Qiskit Runtime job {job_id!r} ({type(exc).__name__})."
        ) from None


def _primitive_name(primitive: str | None) -> str:
    """Convert a primitive type string to a display name."""

    if primitive == "sampler":
        return "SamplerV2"
    if primitive == "estimator":
        return "EstimatorV2"
    return "Runtime"


async def _create_result_artifacts(
    *,
    primitive: str | None,
    metadata: Any,
    result: Any,
    artifact_key: str | None,
) -> None:
    """Create Prefect artifacts for a Qiskit job result."""

    if primitive == "sampler":
        key = artifact_key or "qiskit-sampler-summary"
        await create_qiskit_sampler_metadata_artifact(metadata, key=key)
        await create_qiskit_sampler_result_artifact(result, key=f"{key}-result")
        return
    if primitive == "estimator":
        key = artifact_key or "qiskit-estimator-summary"
        await create_qiskit_estimator_metadata_artifact(metadata, result=result, key=key)
        await create_qiskit_estimator_result_artifact(result, key=f"{key}-result")
        return

    key = artifact_key or "qiskit-runtime-summary"
    await create_qiskit_execution_markdown_artifact(metadata, key=key)


def _durable_submission_enabled(
    *,
    submission_key: str | None,
    spec_hash: str | None,
    journal_path: str | Path | None,
    submission_tags: Iterable[str] | None,
) -> bool:
    values = (submission_key, spec_hash, journal_path)
    if all(value is None for value in values):
        if submission_tags is not None:
            raise ValueError(
                "submission_tags requires submission_key, spec_hash, and journal_path."
            )
        return False
    if any(value is None for value in values):
        raise ValueError(
            "Durable Qiskit submission requires submission_key, spec_hash, and journal_path."
        )
    return True


def _durable_job_tags(options: Mapping[str, Any]) -> list[str]:
    environment = options.get("environment")
    if not isinstance(environment, Mapping):
        return []
    tags = environment.get("job_tags")
    if tags is None or isinstance(tags, str | bytes):
        return []
    return [str(tag) for tag in tags]


def _reference_from_journal_record(
    record: QiskitSubmissionRecord,
) -> QiskitJobReference:
    if record.job_id is None or record.job_reference is None:
        raise QiskitOperatorActionRequired(
            submission_key=record.submission_key,
            reason="SUBMITTED journal row is missing its durable job reference",
        )
    reference: QiskitJobReference = dict(record.job_reference)
    if str(reference.get("job_id", "")) != record.job_id:
        raise QiskitOperatorActionRequired(
            submission_key=record.submission_key,
            reason="journal job ID and durable job reference do not match",
        )
    return reference


def _raise_qiskit_operator_hold(
    *,
    journal: QiskitSubmissionJournal,
    submission_key: str,
    reason: str,
) -> None:
    journal.mark_awaiting_operator(submission_key, reason)
    raise QiskitOperatorActionRequired(
        submission_key=submission_key,
        reason=reason,
    )


def _recover_qiskit_submission(
    *,
    journal: QiskitSubmissionJournal,
    record: QiskitSubmissionRecord,
    service: Any,
    backend_name: str,
    recovery_grace_seconds: float,
    recovery_clock_skew_margin_seconds: float,
    reference_factory: Any,
    now: datetime | None = None,
) -> QiskitJobReference:
    if record.status == QiskitSubmissionStatus.SUBMITTED:
        return _reference_from_journal_record(record)
    if record.status == QiskitSubmissionStatus.AWAITING_OPERATOR:
        raise QiskitOperatorActionRequired(
            submission_key=record.submission_key,
            reason=record.last_error or "journal row is in durable operator hold",
        )

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("Qiskit recovery current time must be timezone-aware.")
    skew = timedelta(seconds=recovery_clock_skew_margin_seconds)
    identity = QiskitJobIdentity(
        stable_tag=record.stable_tag,
        backend_name=backend_name,
        search_start=record.prepared_at - skew,
        search_end=current_time + skew,
    )
    elapsed = max(0.0, (current_time - record.prepared_at).total_seconds())

    try:
        candidates = QiskitIdentityRecoveryRuntime(service).find_candidates_by_identity(identity)
    except Exception as exc:
        reason = (
            f"Qiskit Runtime identity search failed; PREPARED is preserved: {type(exc).__name__}"
        )
        journal.record_prepared_error(record.submission_key, reason)
        if elapsed >= recovery_grace_seconds:
            _raise_qiskit_operator_hold(
                journal=journal,
                submission_key=record.submission_key,
                reason=reason,
            )
        raise QiskitRecoveryPending(
            submission_key=record.submission_key,
            retry_after_seconds=recovery_grace_seconds - elapsed,
        ) from exc

    if any(not candidate.identity_matches for candidate in candidates):
        _raise_qiskit_operator_hold(
            journal=journal,
            submission_key=record.submission_key,
            reason=("Qiskit Runtime candidate does not carry the immutable spec-derived tag"),
        )

    metadata_errors = [
        candidate.metadata_error for candidate in candidates if candidate.metadata_error
    ]
    if metadata_errors:
        _raise_qiskit_operator_hold(
            journal=journal,
            submission_key=record.submission_key,
            reason="; ".join(metadata_errors),
        )

    if len(candidates) > 1:
        _raise_qiskit_operator_hold(
            journal=journal,
            submission_key=record.submission_key,
            reason=(
                f"Found {len(candidates)} matching Qiskit Runtime jobs; automatic "
                "selection is forbidden"
            ),
        )
    if len(candidates) == 1:
        candidate = candidates[0]
        reference = reference_factory(candidate.job_id)
        journal.mark_submitted(
            submission_key=record.submission_key,
            spec_hash=record.spec_hash,
            job_id=candidate.job_id,
            job_reference=reference,
        )
        return reference

    if elapsed < recovery_grace_seconds:
        remaining = max(0.0, recovery_grace_seconds - elapsed)
        journal.record_prepared_error(
            record.submission_key,
            "No Qiskit Runtime identity candidate is visible yet; PREPARED is preserved "
            f"for {remaining:g} more grace seconds.",
        )
        raise QiskitRecoveryPending(
            submission_key=record.submission_key,
            retry_after_seconds=remaining,
        )

    _raise_qiskit_operator_hold(
        journal=journal,
        submission_key=record.submission_key,
        reason=("No matching Qiskit Runtime job was found after the visibility grace period"),
    )


async def _submit_primitive_job_reference(
    *,
    runtime_block_name: str | None,
    runtime_config: Any | None,
    pubs: Iterable[Any],
    primitive_type: str,
    primitive_name: str,
    error_cls: type[RuntimeError],
    create_primitive_fn: Any,
    options: dict[str, Any] | None = None,
    shots: int | None = None,
    precision: float | None = None,
    input_digest: str | None = None,
    submission_key: str | None = None,
    spec_hash: str | None = None,
    journal_path: str | Path | None = None,
    submission_tags: Iterable[str] | None = None,
    recovery_grace_seconds: float = 120.0,
    recovery_clock_skew_margin_seconds: float = 60.0,
) -> QiskitJobReference:
    """Submit a primitive job and return a serializable job reference."""

    logger = _logger()
    durable_enabled = _durable_submission_enabled(
        submission_key=submission_key,
        spec_hash=spec_hash,
        journal_path=journal_path,
        submission_tags=submission_tags,
    )
    if recovery_grace_seconds < 0:
        raise ValueError("recovery_grace_seconds must be non-negative.")
    if recovery_clock_skew_margin_seconds < 0:
        raise ValueError("recovery_clock_skew_margin_seconds must be non-negative.")

    effective_options = options
    journal: QiskitSubmissionJournal | None = None
    record: QiskitSubmissionRecord | None = None
    claimed = False
    stable_tag: str | None = None
    if durable_enabled:
        assert submission_key is not None
        assert spec_hash is not None
        assert journal_path is not None
        stable_tag = build_qiskit_submission_tag(
            submission_key=submission_key,
            spec_hash=spec_hash,
        )
        effective_options = with_qiskit_submission_tags(
            options,
            stable_tag=stable_tag,
            submission_tags=submission_tags,
        )
        journal = QiskitSubmissionJournal(journal_path)
        record, claimed = journal.prepare(
            submission_key=submission_key,
            spec_hash=spec_hash,
            stable_tag=stable_tag,
        )
        if record.status == QiskitSubmissionStatus.SUBMITTED:
            return _reference_from_journal_record(record)
        if record.status == QiskitSubmissionStatus.AWAITING_OPERATOR:
            raise QiskitOperatorActionRequired(
                submission_key=submission_key,
                reason=record.last_error or "journal row is in durable operator hold",
            )

    resolved_runtime_config, backend, backend_name = await _load_runtime_backend(
        runtime_block_name,
        runtime_config=runtime_config,
        error_cls=error_cls,
    )

    def reference_factory(job_id: str) -> QiskitJobReference:
        reference_kwargs: dict[str, Any] = {
            "program_type": primitive_type,
            "backend_name": backend_name,
            "job_id": job_id,
            "runtime_block_name": runtime_block_name,
            "input_digest": input_digest,
            "options": effective_options,
        }
        if shots is not None:
            reference_kwargs["shots"] = shots
        if precision is not None:
            reference_kwargs["precision"] = precision
        if durable_enabled:
            reference_kwargs.update(
                {
                    "submission_key": submission_key,
                    "spec_hash": spec_hash,
                    "job_tags": _durable_job_tags(effective_options or {}),
                    "journal_path": journal_path,
                }
            )
        return _job_reference(**reference_kwargs)

    if durable_enabled and not claimed:
        assert journal is not None
        assert record is not None
        service = _get_service(
            resolved_runtime_config,
            runtime_block_name=runtime_block_name,
        )
        return _recover_qiskit_submission(
            journal=journal,
            record=record,
            service=service,
            backend_name=backend_name,
            recovery_grace_seconds=recovery_grace_seconds,
            recovery_clock_skew_margin_seconds=recovery_clock_skew_margin_seconds,
            reference_factory=reference_factory,
        )

    primitive = create_primitive_fn(
        backend=backend,
        backend_name=backend_name,
        options=effective_options,
    )
    pub_list = list(pubs)
    try:
        job = _run_primitive_job(
            primitive=primitive,
            pubs=pub_list,
            primitive_name=primitive_name,
            error_cls=error_cls,
            backend_name=backend_name,
            shots=shots,
            precision=precision,
            include_shots=primitive_type == "sampler",
            include_precision=primitive_type == "estimator",
        )
    except Exception as exc:
        if not durable_enabled:
            raise
        assert journal is not None
        assert record is not None
        journal.record_prepared_error(
            record.submission_key,
            "Qiskit Runtime submission outcome is unknown; automatic resubmit is "
            f"forbidden: {type(exc).__name__}",
        )
        service = _get_service(
            resolved_runtime_config,
            runtime_block_name=runtime_block_name,
        )
        return _recover_qiskit_submission(
            journal=journal,
            record=record,
            service=service,
            backend_name=backend_name,
            recovery_grace_seconds=recovery_grace_seconds,
            recovery_clock_skew_margin_seconds=recovery_clock_skew_margin_seconds,
            reference_factory=reference_factory,
        )

    job_id = _require_job_id(job, primitive_name=primitive_name, error_cls=error_cls)
    logger.info("Submitted Qiskit %s job %s to backend %s.", primitive_name, job_id, backend_name)
    reference = reference_factory(job_id)
    if not durable_enabled:
        return reference

    assert journal is not None
    assert record is not None
    try:
        journal.mark_submitted(
            submission_key=record.submission_key,
            spec_hash=record.spec_hash,
            job_id=job_id,
            job_reference=reference,
        )
    except Exception as exc:
        journal.record_prepared_error(
            record.submission_key,
            "Qiskit Runtime returned a job ID, but journal persistence failed; "
            f"automatic resubmit is forbidden: {type(exc).__name__}",
        )
        service = _get_service(
            resolved_runtime_config,
            runtime_block_name=runtime_block_name,
        )
        return _recover_qiskit_submission(
            journal=journal,
            record=record,
            service=service,
            backend_name=backend_name,
            recovery_grace_seconds=recovery_grace_seconds,
            recovery_clock_skew_margin_seconds=recovery_clock_skew_margin_seconds,
            reference_factory=reference_factory,
        )
    return reference


@task(name="submit-qiskit-sampler-job")
async def submit_sampler_job_task(
    pubs: Iterable[Any],
    runtime_block_name: str | None = None,
    shots: int | None = None,
    options: dict[str, Any] | None = None,
    runtime_config: QiskitRuntimeConfig | None = None,
    *,
    input_digest: str | None = None,
    submission_key: str | None = None,
    spec_hash: str | None = None,
    journal_path: str | Path | None = None,
    submission_tags: Iterable[str] | None = None,
    recovery_grace_seconds: float = 120.0,
    recovery_clock_skew_margin_seconds: float = 60.0,
) -> QiskitJobReference:
    """Submit a native Qiskit Runtime ``SamplerV2`` job without waiting.

    Args:
        pubs: Native Qiskit Sampler pubs.
        runtime_block_name: Name of a saved ``QiskitRuntimeConfig`` block.
        shots: Optional number of shots passed to ``SamplerV2.run``.
        options: Optional native Qiskit Sampler options.
        runtime_config: Inline runtime configuration. Mutually exclusive with
            ``runtime_block_name``.
        input_digest: Optional stable digest used by submit cache helpers.
        submission_key: Stable caller-defined key that enables durable mode.
        spec_hash: Immutable resolved-spec hash for ``submission_key``.
        journal_path: Shared-filesystem SQLite journal path.
        submission_tags: Optional additional non-secret Runtime job tags.
        recovery_grace_seconds: Visibility grace before zero matches enter hold.
        recovery_clock_skew_margin_seconds: Search-window clock-skew margin.

    Returns:
        A serializable job reference containing primitive, backend, job ID and
        optional execution settings.
    """
    return await _submit_primitive_job_reference(
        runtime_block_name=runtime_block_name,
        runtime_config=runtime_config,
        pubs=pubs,
        primitive_type="sampler",
        primitive_name="SamplerV2",
        error_cls=QiskitSamplerTaskError,
        create_primitive_fn=_create_sampler,
        options=options,
        shots=shots,
        input_digest=input_digest,
        submission_key=submission_key,
        spec_hash=spec_hash,
        journal_path=journal_path,
        submission_tags=submission_tags,
        recovery_grace_seconds=recovery_grace_seconds,
        recovery_clock_skew_margin_seconds=recovery_clock_skew_margin_seconds,
    )


@task(name="submit-qiskit-estimator-job")
async def submit_estimator_job_task(
    pubs: Iterable[Any],
    runtime_block_name: str | None = None,
    precision: float | None = None,
    options: dict[str, Any] | None = None,
    runtime_config: QiskitRuntimeConfig | None = None,
    *,
    input_digest: str | None = None,
    submission_key: str | None = None,
    spec_hash: str | None = None,
    journal_path: str | Path | None = None,
    submission_tags: Iterable[str] | None = None,
    recovery_grace_seconds: float = 120.0,
    recovery_clock_skew_margin_seconds: float = 60.0,
) -> QiskitJobReference:
    """Submit a native Qiskit Runtime ``EstimatorV2`` job without waiting.

    Args:
        pubs: Native Qiskit Estimator pubs.
        runtime_block_name: Name of a saved ``QiskitRuntimeConfig`` block.
        precision: Optional target precision passed to ``EstimatorV2.run``.
        options: Optional native Qiskit Estimator options.
        runtime_config: Inline runtime configuration. Mutually exclusive with
            ``runtime_block_name``.
        input_digest: Optional stable digest used by submit cache helpers.
        submission_key: Stable caller-defined key that enables durable mode.
        spec_hash: Immutable resolved-spec hash for ``submission_key``.
        journal_path: Shared-filesystem SQLite journal path.
        submission_tags: Optional additional non-secret Runtime job tags.
        recovery_grace_seconds: Visibility grace before zero matches enter hold.
        recovery_clock_skew_margin_seconds: Search-window clock-skew margin.

    Returns:
        A serializable job reference containing primitive, backend, job ID and
        optional execution settings.
    """
    return await _submit_primitive_job_reference(
        runtime_block_name=runtime_block_name,
        runtime_config=runtime_config,
        pubs=pubs,
        primitive_type="estimator",
        primitive_name="EstimatorV2",
        error_cls=QiskitEstimatorTaskError,
        create_primitive_fn=_create_estimator,
        options=options,
        precision=precision,
        input_digest=input_digest,
        submission_key=submission_key,
        spec_hash=spec_hash,
        journal_path=journal_path,
        submission_tags=submission_tags,
        recovery_grace_seconds=recovery_grace_seconds,
        recovery_clock_skew_margin_seconds=recovery_clock_skew_margin_seconds,
    )


@task(name="fetch-qiskit-job-result")
async def fetch_qiskit_job_result_task(
    runtime_block_name: str | None = None,
    job_id: str | None = None,
    job_reference: Mapping[str, Any] | None = None,
    pubs: Iterable[Any] | None = None,
    primitive: str | None = None,
    backend_name: str | None = None,
    shots: int | None = None,
    precision: float | None = None,
    artifact_key: str | None = None,
    options: dict[str, Any] | None = None,
    runtime_config: QiskitRuntimeConfig | None = None,
) -> dict[str, Any]:
    """Fetch an existing native Qiskit Runtime job and collect its result.

    Args:
        runtime_block_name: Name of a saved ``QiskitRuntimeConfig`` block.
        job_id: Qiskit Runtime job ID. Optional when ``job_reference`` is
            provided.
        job_reference: Reference returned by ``submit_sampler_job_task`` or
            ``submit_estimator_job_task``.
        pubs: Original native Qiskit pubs used for best-effort metadata.
        primitive: Optional primitive type when not present in
            ``job_reference``.
        backend_name: Optional backend name when not present in
            ``job_reference``.
        shots: Optional shots value for metadata.
        precision: Optional precision value for metadata.
        artifact_key: Optional Prefect artifact key prefix.
        options: Optional native Qiskit primitive options for metadata.
        runtime_config: Inline runtime configuration. Mutually exclusive with
            ``runtime_block_name``.

    Returns:
        A structured dictionary containing primitive, backend name, job ID,
        native Qiskit result and collected metadata.
    """

    reference = _resolve_job_reference(
        job_reference=job_reference,
        job_id=job_id,
        primitive=primitive,
        backend_name=backend_name,
        shots=shots,
        precision=precision,
    )
    resolved_job_id = reference["job_id"]
    resolved_primitive = reference.get("primitive")
    resolved_backend_name = reference.get("backend_name")
    resolved_shots = reference.get("shots")
    resolved_precision = reference.get("precision")
    resolved_input_digest = reference.get("input_digest")

    runtime_config = await _resolve_runtime_config(
        runtime_block_name,
        runtime_config=runtime_config,
        error_cls=QiskitJobFetchTaskError,
    )
    service = _get_service(runtime_config, runtime_block_name=runtime_block_name)
    job = _fetch_existing_job(service=service, job_id=resolved_job_id)
    logger = _logger()
    logger.info("Fetching Qiskit Runtime job %s.", resolved_job_id)

    result = await _wait_for_primitive_result(
        job=job,
        job_id=resolved_job_id,
        backend_name=str(resolved_backend_name or "<unknown>"),
        primitive_name=_primitive_name(resolved_primitive),
        error_cls=QiskitJobFetchTaskError,
    )
    pub_list = list(pubs) if pubs is not None else None
    metadata = collect_qiskit_execution_metadata(
        job=job,
        pubs=pub_list,
        result=result,
        options=_metadata_options(
            options,
            shots=resolved_shots,
            precision=resolved_precision,
        ),
        resource=resolved_backend_name,
        program_type=resolved_primitive,
        input_digest=resolved_input_digest,
    )
    if metadata.job_id is None:
        metadata.job_id = resolved_job_id

    await _create_result_artifacts(
        primitive=resolved_primitive,
        metadata=metadata,
        result=result,
        artifact_key=artifact_key,
    )

    return _result_response(
        primitive=resolved_primitive,
        backend_name=resolved_backend_name,
        job_id=metadata.job_id,
        shots=resolved_shots,
        precision=resolved_precision,
        input_digest=resolved_input_digest,
        result=result,
        metadata=metadata,
    )


def build_cached_fetch_qiskit_job_result_task(**task_options: Any) -> Any:
    """Build a fetch task that persists native Qiskit results with Prefect.

    This mirrors prefect-qiskit's execution cache pattern: the task result is
    persisted with Prefect's compressed pickle serializer and keyed by Qiskit
    Runtime job ID. On cache hits, Prefect restores the stored result without
    calling the Qiskit Runtime service again.
    """

    options: dict[str, Any] = {
        "cache_key_fn": qiskit_result_fetch_cache_key,
        "persist_result": True,
        "result_serializer": "compressed/pickle",
    }
    options.update(task_options)
    return fetch_qiskit_job_result_task.with_options(**options)


cached_fetch_qiskit_job_result_task = build_cached_fetch_qiskit_job_result_task()


@task(name="run-qiskit-sampler")
async def run_sampler_task(
    pubs: Iterable[Any],
    runtime_block_name: str | None = None,
    shots: int | None = None,
    artifact_key: str | None = None,
    options: dict[str, Any] | None = None,
    runtime_config: QiskitRuntimeConfig | None = None,
    *,
    input_digest: str | None = None,
) -> dict[str, Any]:
    """Run native Qiskit Runtime ``SamplerV2`` inside a Prefect task.

    This is the simple one-task execution path. It submits a Qiskit Runtime job,
    waits for the result, records artifacts and returns the result dictionary.

    Args:
        pubs: Native Qiskit Sampler pubs.
        runtime_block_name: Name of a saved ``QiskitRuntimeConfig`` block.
        shots: Optional number of shots passed to ``SamplerV2.run``.
        artifact_key: Optional Prefect artifact key prefix.
        options: Optional native Qiskit Sampler options.
        runtime_config: Inline runtime configuration. Mutually exclusive with
            ``runtime_block_name``.
        input_digest: Optional input digest included in collected metadata.

    Returns:
        A structured dictionary containing primitive, backend name, job ID,
        shots, native Qiskit result and collected metadata.
    """

    logger = _logger()
    _, backend, backend_name = await _load_runtime_backend(
        runtime_block_name,
        runtime_config=runtime_config,
    )

    sampler = _create_sampler(backend=backend, backend_name=backend_name, options=options)
    pub_list = list(pubs)
    job = _submit_sampler_job(
        sampler=sampler,
        pubs=pub_list,
        shots=shots,
        backend_name=backend_name,
    )
    job_id = _job_id(job)
    logger.info(
        "Submitted Qiskit SamplerV2 job %s to backend %s.",
        job_id or "<unknown>",
        backend_name,
    )

    result = await _wait_for_primitive_result(
        job=job,
        job_id=job_id,
        backend_name=backend_name,
        primitive_name="SamplerV2",
        error_cls=QiskitSamplerTaskError,
    )
    metadata = collect_qiskit_execution_metadata(
        job=job,
        pubs=pub_list,
        result=result,
        options=_metadata_options(options, shots=shots),
        resource=backend_name,
        program_type="sampler",
        input_digest=input_digest,
    )
    if metadata.job_id is None:
        metadata.job_id = job_id

    await _create_result_artifacts(
        primitive="sampler",
        metadata=metadata,
        result=result,
        artifact_key=artifact_key,
    )

    return _result_response(
        primitive="sampler",
        backend_name=backend_name,
        job_id=metadata.job_id,
        shots=shots,
        result=result,
        metadata=metadata,
        include_shots=True,
        input_digest=input_digest,
    )


@task(name="run-qiskit-estimator")
async def run_estimator_task(
    pubs: Iterable[Any],
    runtime_block_name: str | None = None,
    precision: float | None = None,
    artifact_key: str | None = None,
    options: dict[str, Any] | None = None,
    runtime_config: QiskitRuntimeConfig | None = None,
    *,
    input_digest: str | None = None,
) -> dict[str, Any]:
    """Run native Qiskit Runtime ``EstimatorV2`` inside a Prefect task.

    This is the simple one-task execution path. It submits a Qiskit Runtime job,
    waits for the result, records artifacts and returns the result dictionary.

    Args:
        pubs: Native Qiskit Estimator pubs.
        runtime_block_name: Name of a saved ``QiskitRuntimeConfig`` block.
        precision: Optional target precision passed to ``EstimatorV2.run``.
        artifact_key: Optional Prefect artifact key prefix.
        options: Optional native Qiskit Estimator options.
        runtime_config: Inline runtime configuration. Mutually exclusive with
            ``runtime_block_name``.
        input_digest: Optional input digest included in collected metadata.

    Returns:
        A structured dictionary containing primitive, backend name, job ID,
        precision, native Qiskit result and collected metadata.
    """

    logger = _logger()
    _, backend, backend_name = await _load_runtime_backend(
        runtime_block_name,
        runtime_config=runtime_config,
        error_cls=QiskitEstimatorTaskError,
    )

    estimator = _create_estimator(backend=backend, backend_name=backend_name, options=options)
    pub_list = list(pubs)
    job = _submit_estimator_job(
        estimator=estimator,
        pubs=pub_list,
        precision=precision,
        backend_name=backend_name,
    )
    job_id = _job_id(job)
    logger.info(
        "Submitted Qiskit EstimatorV2 job %s to backend %s.",
        job_id or "<unknown>",
        backend_name,
    )

    result = await _wait_for_primitive_result(
        job=job,
        job_id=job_id,
        backend_name=backend_name,
        primitive_name="EstimatorV2",
        error_cls=QiskitEstimatorTaskError,
    )
    metadata = collect_qiskit_execution_metadata(
        job=job,
        pubs=pub_list,
        result=result,
        options=_metadata_options(options, precision=precision),
        resource=backend_name,
        program_type="estimator",
        input_digest=input_digest,
    )
    if metadata.job_id is None:
        metadata.job_id = job_id

    await _create_result_artifacts(
        primitive="estimator",
        metadata=metadata,
        result=result,
        artifact_key=artifact_key,
    )

    return _result_response(
        primitive="estimator",
        backend_name=backend_name,
        job_id=metadata.job_id,
        precision=precision,
        result=result,
        metadata=metadata,
        include_precision=True,
        input_digest=input_digest,
    )
