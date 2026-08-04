"""Bulk HPC execution helpers."""

from __future__ import annotations

from typing import Any

from qcsc_prefect_core.queue import QueueAwareSubmitGate, QueueCapacity, QueueProbe

from qcsc_prefect_executor.bulk.exceptions import (
    CancellationRequestedError,
    DuplicateJobKeyError,
    OperatorActionRequired,
    QueueFullError,
    RecoveryPending,
    SchedulerIdentityMismatchError,
    SpecHashMismatchError,
    SubmitError,
    SubmitOutcomeUnknownError,
    TemporarySubmitError,
)
from qcsc_prefect_executor.bulk.global_fugaku_runner import GlobalFugakuBulkRunner
from qcsc_prefect_executor.bulk.models import (
    BulkCancelOutcome,
    BulkJobDesiredState,
    BulkJobRecord,
    BulkJobSpec,
    BulkJobStatus,
    BulkRunResult,
    BulkTickResult,
    SubmittedJob,
)
from qcsc_prefect_executor.bulk.native_manifest import (
    NativeBulkManifestGroup,
    create_native_bulk_group_manifests,
)
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry
from qcsc_prefect_executor.bulk.spec_hash import (
    BULK_SPEC_HASH_SCHEMA_VERSION,
    build_bulk_spec_hash,
    canonical_bulk_spec_json,
)
from qcsc_prefect_executor.cloud_logs import CloudLogPolicy


async def monitor_jobs_many(*args: Any, **kwargs: Any):
    # Lazy import keeps this package importable while from_blocks imports bulk modules.
    from qcsc_prefect_executor.from_blocks import monitor_jobs_many as _monitor_jobs_many

    return await _monitor_jobs_many(*args, **kwargs)


async def run_jobs_from_blocks_bulk(*args: Any, **kwargs: Any):
    # Lazy import keeps this package importable while from_blocks imports bulk modules.
    from qcsc_prefect_executor.from_blocks import (
        run_jobs_from_blocks_bulk as _run_jobs_from_blocks_bulk,
    )

    return await _run_jobs_from_blocks_bulk(*args, **kwargs)


async def execute_cancel_requests(*args: Any, **kwargs: Any):
    from qcsc_prefect_executor.from_blocks import (
        execute_cancel_requests as _execute_cancel_requests,
    )

    return await _execute_cancel_requests(*args, **kwargs)


async def execute_cancel_request(*args: Any, **kwargs: Any):
    from qcsc_prefect_executor.from_blocks import execute_cancel_request as _execute_cancel_request

    return await _execute_cancel_request(*args, **kwargs)


async def submit_job_from_blocks(*args: Any, **kwargs: Any):
    # Lazy import keeps this package importable while from_blocks imports bulk modules.
    from qcsc_prefect_executor.from_blocks import submit_job_from_blocks as _submit_job_from_blocks

    return await _submit_job_from_blocks(*args, **kwargs)


__all__ = [
    "BULK_SPEC_HASH_SCHEMA_VERSION",
    "BulkCancelOutcome",
    "BulkJobDesiredState",
    "BulkJobRecord",
    "BulkJobRegistry",
    "BulkJobSpec",
    "BulkJobStatus",
    "BulkRunResult",
    "BulkTickResult",
    "CancellationRequestedError",
    "CloudLogPolicy",
    "DuplicateJobKeyError",
    "GlobalFugakuBulkRunner",
    "NativeBulkManifestGroup",
    "OperatorActionRequired",
    "QueueFullError",
    "QueueAwareSubmitGate",
    "QueueCapacity",
    "QueueProbe",
    "RecoveryPending",
    "SchedulerIdentityMismatchError",
    "SpecHashMismatchError",
    "SubmittedJob",
    "SubmitError",
    "SubmitOutcomeUnknownError",
    "TemporarySubmitError",
    "build_bulk_spec_hash",
    "canonical_bulk_spec_json",
    "create_native_bulk_group_manifests",
    "execute_cancel_request",
    "execute_cancel_requests",
    "monitor_jobs_many",
    "run_jobs_from_blocks_bulk",
    "submit_job_from_blocks",
]
