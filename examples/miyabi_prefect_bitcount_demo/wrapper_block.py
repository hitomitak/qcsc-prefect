from __future__ import annotations

import json
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from prefect import task
from prefect.blocks.core import Block
from pydantic import Field

from hpc_prefect_adapters.miyabi.builder import MiyabiJobRequest
from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_executor.miyabi.run import run_miyabi_job

BITLEN = 10


class BitCounterWrapperBlock(Block):
    """
    Legacy-style wrapper block.

    This keeps the old tutorial ergonomics: `counts = await counter.get(bitstrings)`.
    """

    _block_type_name = "Bit Counter Wrapper"
    _block_type_slug = "bit-counter-wrapper"

    root_dir: str = Field(title="Root Directory")
    executable: str = Field(title="Executable")
    queue_name: str = Field(default="regular-c", title="Queue Name")
    project: str = Field(title="Project")
    num_nodes: int = Field(default=2, gt=0, title="Num Nodes")
    num_mpi_processes: int = Field(default=5, gt=0, title="Num MPI Processes")
    ompthreads: int = Field(default=1, gt=0, title="OMP Threads")
    walltime: str = Field(default="00:10:00", title="Walltime")
    launcher: Literal["single", "mpirun", "mpiexec", "mpiexec.hydra"] = Field(default="mpiexec.hydra", title="Launcher")
    load_modules: list[str] = Field(
        default_factory=lambda: ["intel/2023.2.0", "impi/2021.10.0"],
        title="Load Modules",
    )
    mpi_options: list[str] = Field(default_factory=list, title="MPI Options")
    environments: dict[str, str] = Field(default_factory=dict, title="Environment Variables")
    script_filename: str = Field(default="bitcount_wrapper.pbs", title="Script Filename")

    async def get(self, bitstrings: list[str]) -> dict[str, int]:
        return await _run_bitcount_wrapper(self, bitstrings)


@task(name="get_counts_mpi_wrapper")
async def _run_bitcount_wrapper(
    job: BitCounterWrapperBlock,
    bitstrings: list[str],
) -> dict[str, int]:
    base_work_dir = Path(job.root_dir).expanduser().resolve()
    base_work_dir.mkdir(parents=True, exist_ok=True)
    job_work_dir = _make_job_work_dir(base_work_dir)

    # Keep data format identical to the original tutorial: raw uint32 sequence in input.bin.
    input_values = array("I", (int(bits, 2) for bits in bitstrings))
    with (job_work_dir / "input.bin").open("wb") as f:
        input_values.tofile(f)

    exec_profile = ExecutionProfile(
        command_key="bit-count-wrapper",
        num_nodes=job.num_nodes,
        mpiprocs=job.num_mpi_processes,
        ompthreads=job.ompthreads,
        walltime=job.walltime,
        launcher=job.launcher,
        mpi_options=list(job.mpi_options),
        modules=list(job.load_modules),
        environments=dict(job.environments),
    )

    req = MiyabiJobRequest(
        queue_name=job.queue_name,
        project=job.project,
        executable=job.executable,
    )

    result = await run_miyabi_job(
        work_dir=job_work_dir,
        script_filename=job.script_filename,
        exec_profile=exec_profile,
        req=req,
        watch_poll_interval=5.0,
        timeout_seconds=1800,
        metrics_artifact_key="miyabi-bitcount-wrapper-metrics",
    )
    if result.exit_status != 0:
        raise RuntimeError(f"BitCount wrapper job failed: exit_status={result.exit_status}")

    with (job_work_dir / "output.json").open("r", encoding="utf-8") as f:
        int_counts = json.load(f)

    return {format(int(k), f"0{BITLEN}b"): int(v) for k, v in int_counts.items()}


def _make_job_work_dir(base_work_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_dir = base_work_dir / f"job_{timestamp}_{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir
