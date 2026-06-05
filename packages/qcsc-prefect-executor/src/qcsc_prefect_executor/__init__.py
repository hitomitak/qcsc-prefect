"""HPC executor package.

Keep top-level imports lightweight to avoid circular-import side effects when
submodules (e.g. ``qcsc_prefect_executor.miyabi.run``) are imported directly.
"""

from __future__ import annotations

from typing import Any


async def run_job_from_blocks(*args: Any, **kwargs: Any):
    # Lazy import prevents circular initialization across package __init__.py files.
    from .from_blocks import run_job_from_blocks as _run_job_from_blocks

    return await _run_job_from_blocks(*args, **kwargs)


async def run_jobs_from_blocks_bulk(*args: Any, **kwargs: Any):
    # Lazy import prevents circular initialization across package __init__.py files.
    from .from_blocks import run_jobs_from_blocks_bulk as _run_jobs_from_blocks_bulk

    return await _run_jobs_from_blocks_bulk(*args, **kwargs)


__all__ = ["run_job_from_blocks", "run_jobs_from_blocks_bulk"]
