"""Shared Prefect enablement helpers for GB-SQD workflows."""

from __future__ import annotations

from ._workspace_support import ensure_workspace_packages

ensure_workspace_packages(
    "qcsc-prefect-core",
    "qcsc-prefect-adapters",
    "qcsc-prefect-blocks",
    "qcsc-prefect-executor",
)

from qcsc_prefect_executor.from_blocks import (  # noqa: E402
    build_scheduler_script_filename,
    resolve_hpc_target,
    resolve_submission_target,
    run_job_from_blocks,
)

__all__ = [
    "build_scheduler_script_filename",
    "resolve_hpc_target",
    "resolve_submission_target",
    "run_job_from_blocks",
]
