from __future__ import annotations

from typing import Any

from .run import SlurmRunResult, run_slurm_job


async def run_slurm_job_from_blocks(*args: Any, **kwargs: Any):
    # Lazy import avoids cycles with qcsc_prefect_executor.from_blocks.
    from .from_blocks import run_slurm_job_from_blocks as _run_slurm_job_from_blocks

    return await _run_slurm_job_from_blocks(*args, **kwargs)


__all__ = ["SlurmRunResult", "run_slurm_job", "run_slurm_job_from_blocks"]
