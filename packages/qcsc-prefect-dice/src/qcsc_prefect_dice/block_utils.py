"""Helpers for registering and creating DICE-related Prefect blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock

from .solver_job import DiceSHCISolverJob


def register_dice_block_types() -> None:
    """Register common block schemas used by DICE integrations."""

    for block_cls in (
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
        DiceSHCISolverJob,
    ):
        register = getattr(block_cls, "register_type_and_schema", None)
        if callable(register):
            register()


def create_dice_blocks(
    *,
    hpc_target: Literal["miyabi", "fugaku"],
    project: str,
    queue: str,
    root_dir: str,
    dice_executable: str,
    command_block_name: str = "cmd-dice-solver",
    execution_profile_block_name: str | None = None,
    hpc_profile_block_name: str | None = None,
    solver_block_name: str = "dice-solver",
    command_name: str = "dice",
    executable_key: str = "dice_solver",
    profile_name: str = "dice-mpi",
    launcher: Literal["single", "mpirun", "mpiexec", "mpiexec.hydra"] | None = None,
    num_nodes: int = 1,
    mpiprocs: int = 4,
    ompthreads: int | None = None,
    walltime: str = "01:00:00",
    resource_class: Literal["cpu", "gpu"] = "cpu",
    modules: list[str] | None = None,
    mpi_options: list[str] | None = None,
    pre_commands: list[str] | None = None,
    environments: dict[str, str] | None = None,
    script_filename: str | None = None,
    metrics_artifact_key: str | None = None,
    select_cutoff: float = 5e-4,
    davidson_tol: float = 1e-5,
    energy_tol: float = 1e-10,
    max_iter: int = 10,
    return_sci_state: bool = True,
    gfscache: str | None = None,
    spack_modules: list[str] | None = None,
    mpi_options_for_pjm: list[str] | None = None,
    pjm_resources: list[str] | None = None,
) -> dict[str, str]:
    """Create the standard block set required to run DICE on qcsc-prefect.

    This helper keeps a typed, explicit API on purpose. Algorithm-specific
    wrappers can translate TOML or CLI config dictionaries into these arguments
    without leaking config-shape coupling into the shared package.
    """

    register_dice_block_types()

    resolved_launcher = launcher or ("mpiexec.hydra" if hpc_target == "miyabi" else "mpiexec")
    resolved_exec_block_name = execution_profile_block_name or (
        "exec-dice-mpi" if hpc_target == "miyabi" else "exec-dice-fugaku"
    )
    resolved_hpc_block_name = hpc_profile_block_name or f"hpc-{hpc_target}-dice"
    resolved_script_filename = script_filename or (
        "dice_solver.pbs" if hpc_target == "miyabi" else "dice_solver.pjm"
    )
    resolved_metrics_key = metrics_artifact_key or (
        "miyabi-dice-metrics" if hpc_target == "miyabi" else "fugaku-dice-metrics"
    )

    CommandBlock(
        command_name=command_name,
        executable_key=executable_key,
        description="DICE SHCI solver executable",
        default_args=[],
    ).save(command_block_name, overwrite=True)

    ExecutionProfileBlock(
        profile_name=profile_name,
        command_name=command_name,
        resource_class=resource_class,
        num_nodes=num_nodes,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        walltime=walltime,
        launcher=resolved_launcher,
        mpi_options=list(mpi_options or []),
        modules=list(modules or []),
        pre_commands=list(pre_commands or []),
        environments=dict(environments or {}),
    ).save(resolved_exec_block_name, overwrite=True)

    HPCProfileBlock(
        hpc_target=hpc_target,
        queue_cpu=queue,
        queue_gpu=queue,
        project_cpu=project,
        project_gpu=project,
        executable_map={executable_key: str(Path(dice_executable).expanduser().resolve())},
        gfscache=gfscache,
        spack_modules=list(spack_modules or []),
        mpi_options_for_pjm=list(mpi_options_for_pjm or []),
        pjm_resources=list(pjm_resources or []),
    ).save(resolved_hpc_block_name, overwrite=True)

    DiceSHCISolverJob(
        root_dir=str(Path(root_dir).expanduser().resolve()),
        command_block_name=command_block_name,
        execution_profile_block_name=resolved_exec_block_name,
        hpc_profile_block_name=resolved_hpc_block_name,
        script_filename=resolved_script_filename,
        metrics_artifact_key=resolved_metrics_key,
        select_cutoff=select_cutoff,
        davidson_tol=davidson_tol,
        energy_tol=energy_tol,
        max_iter=max_iter,
        return_sci_state=return_sci_state,
    ).save(solver_block_name, overwrite=True)

    return {
        "command_block_name": command_block_name,
        "execution_profile_block_name": resolved_exec_block_name,
        "hpc_profile_block_name": resolved_hpc_block_name,
        "solver_block_name": solver_block_name,
    }
