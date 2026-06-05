"""Bulk HPC execution helpers."""

from __future__ import annotations

from typing import Any

from qcsc_prefect_core.queue import QueueAwareSubmitGate, QueueCapacity, QueueProbe

from qcsc_prefect_executor.bulk.exceptions import (
    DuplicateJobKeyError,
    QueueFullError,
    SubmitError,
    TemporarySubmitError,
)
from qcsc_prefect_executor.bulk.global_fugaku_runner import GlobalFugakuBulkRunner
from qcsc_prefect_executor.bulk.models import (
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


async def submit_job_from_blocks(*args: Any, **kwargs: Any):
    # Lazy import keeps this package importable while from_blocks imports bulk modules.
    from qcsc_prefect_executor.from_blocks import submit_job_from_blocks as _submit_job_from_blocks

    return await _submit_job_from_blocks(*args, **kwargs)


__all__ = [
    "BulkJobRecord",
    "BulkJobRegistry",
    "BulkJobSpec",
    "BulkJobStatus",
    "BulkRunResult",
    "BulkTickResult",
    "DuplicateJobKeyError",
    "GlobalFugakuBulkRunner",
    "NativeBulkManifestGroup",
    "QueueFullError",
    "QueueAwareSubmitGate",
    "QueueCapacity",
    "QueueProbe",
    "SubmittedJob",
    "SubmitError",
    "TemporarySubmitError",
    "create_native_bulk_group_manifests",
    "monitor_jobs_many",
    "run_jobs_from_blocks_bulk",
    "submit_job_from_blocks",
]
