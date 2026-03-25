"""DICE SHCI solver block backed by qcsc-prefect block execution."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from prefect import get_run_logger, task
from prefect.blocks.core import Block
from pydantic import Field
from qiskit_addon_sqd.fermion import SCIResult

from qcsc_prefect_executor.from_blocks import run_job_from_blocks

from .io_utils import make_job_work_dir, prep_dice_input_files, read_dice_output_files


def _logger():
    try:
        return get_run_logger()
    except Exception:
        return logging.getLogger(__name__)


class DiceSHCISolverJob(Block):
    """Prefect block facade for DICE execution through qcsc-prefect blocks."""

    _block_type_name = "DICE SHCI Solver Job"
    _block_type_slug = "dice_shci_solver_job"

    root_dir: str = Field(
        title="Root Directory",
        description="Root directory where per-job work directories are created.",
    )
    command_block_name: str = Field(
        default="cmd-dice-solver",
        title="Command Block Name",
        description="Prefect CommandBlock document name.",
    )
    execution_profile_block_name: str = Field(
        default="exec-dice-mpi",
        title="Execution Profile Block Name",
        description="Prefect ExecutionProfileBlock document name.",
    )
    hpc_profile_block_name: str = Field(
        default="hpc-miyabi-dice",
        title="HPC Profile Block Name",
        description="Prefect HPCProfileBlock document name.",
    )
    script_filename: str = Field(
        default="dice_solver.pbs",
        title="Script Filename",
        description="Scheduler script filename (.pbs for Miyabi, .pjm for Fugaku).",
    )
    metrics_artifact_key: str = Field(default="dice-metrics", title="Metrics Artifact Key")
    timeout_seconds: float = Field(default=7200.0, title="Timeout Seconds")

    select_cutoff: float = Field(
        default=5e-4,
        title="Select Cutoff",
        description="Cutoff threshold for retaining state vector coefficients.",
    )
    davidson_tol: float = Field(
        default=1e-5,
        title="Davidson Tolerance",
        description="Floating point tolerance for Davidson solver.",
    )
    energy_tol: float = Field(
        default=1e-10,
        title="Energy Tolerance",
        description="Floating point tolerance for SCI energy.",
    )
    max_iter: int = Field(
        default=10,
        title="Maximum Iteration",
        description="The maximum number of HCI iterations to perform.",
    )
    return_sci_state: bool = Field(
        default=True,
        title="SCI State",
        description=(
            "Construct SCIState object from dets.bin. "
            "Disable this to skip determinant reconstruction."
        ),
    )

    async def run(
        self,
        ci_strings: tuple[np.ndarray, np.ndarray],
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
        spin_sq: float | None = None,
    ) -> SCIResult:
        """Run DICE and return parsed SHCI outputs."""

        return await _run_dice_inner(
            solver=self,
            ci_strings=ci_strings,
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            norb=norb,
            nelec=nelec,
            spin_sq=spin_sq,
        )


@task(name="solve_eigenstate")
async def _run_dice_inner(
    *,
    solver: DiceSHCISolverJob,
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    spin_sq: float | None = None,
) -> SCIResult:
    logger = _logger()
    base_work_dir = Path(solver.root_dir).expanduser().resolve()
    job_work_dir = make_job_work_dir(base_work_dir)

    prep_dice_input_files(
        work_dir=job_work_dir,
        ci_strings=ci_strings,
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        norb=norb,
        nelec=nelec,
        spin_sq=spin_sq,
        select_cutoff=solver.select_cutoff,
        davidson_tol=solver.davidson_tol,
        energy_tol=solver.energy_tol,
        max_iter=solver.max_iter,
    )

    result = await run_job_from_blocks(
        command_block_name=solver.command_block_name,
        execution_profile_block_name=solver.execution_profile_block_name,
        hpc_profile_block_name=solver.hpc_profile_block_name,
        work_dir=job_work_dir,
        script_filename=solver.script_filename,
        user_args=[],
        watch_poll_interval=5.0,
        timeout_seconds=solver.timeout_seconds,
        metrics_artifact_key=solver.metrics_artifact_key,
    )

    if result.exit_status != 0:
        logger.warning(
            "Dice solver returned nonzero exit status %s. "
            "Dice may still emit valid outputs; inspecting result files.",
            result.exit_status,
        )

    try:
        return read_dice_output_files(
            work_dir=job_work_dir,
            norb=norb,
            nelec=nelec,
            return_sci_state=solver.return_sci_state,
        )
    except FileNotFoundError as exc:
        expected_outputs = ["spin1RDM.0.0.txt", "shci.e"]
        if solver.return_sci_state:
            expected_outputs.append("dets.bin")
        raise RuntimeError(
            f"DICE output file not found: {exc.filename!r}. "
            f"exit_status={result.exit_status}, "
            f"work_dir={str(job_work_dir)!r}, "
            f"expected_outputs={expected_outputs}"
        ) from exc
