"""Bulk HPC execution helpers."""

from qcsc_prefect_core.queue import QueueAwareSubmitGate, QueueCapacity, QueueProbe

from qcsc_prefect_executor.bulk.exceptions import (
    DuplicateJobKeyError,
    QueueFullError,
    SubmitError,
    TemporarySubmitError,
)
from qcsc_prefect_executor.bulk.models import (
    BulkJobRecord,
    BulkJobSpec,
    BulkJobStatus,
    BulkRunResult,
    SubmittedJob,
)
from qcsc_prefect_executor.bulk.native_manifest import (
    NativeBulkManifestGroup,
    create_native_bulk_group_manifests,
)
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry

__all__ = [
    "BulkJobRecord",
    "BulkJobRegistry",
    "BulkJobSpec",
    "BulkJobStatus",
    "BulkRunResult",
    "DuplicateJobKeyError",
    "NativeBulkManifestGroup",
    "QueueFullError",
    "QueueAwareSubmitGate",
    "QueueCapacity",
    "QueueProbe",
    "SubmittedJob",
    "SubmitError",
    "TemporarySubmitError",
    "create_native_bulk_group_manifests",
]
