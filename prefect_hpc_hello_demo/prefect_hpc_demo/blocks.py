"""
Prefect Blocks for an HPC-agnostic execution demo on Miyabi (PBS) using an MPI hello program.
"""

from __future__ import annotations

from typing import Literal

from prefect.blocks.core import Block
from pydantic import Field


class CommandBlock(Block):
    """HPC-agnostic command definition (WHAT to run)."""

    _block_type_name = "HPC Command"
    _block_type_slug = "hpc_command"

    command_name: str = Field(title="Command Name", description="Logical command name, e.g., 'mpi-hello'.")
    executable_key: str = Field(
        title="Executable Key",
        description="Logical key for the executable. Resolved to a path by the HPC profile.",
    )
    description: str | None = Field(default=None, title="Description")
    default_args: list[str] | None = Field(default=None, title="Default Args")


class ExecutionProfileBlock(Block):
    """Baseline execution profile (HOW to run) for a specific command."""

    _block_type_name = "Execution Profile"
    _block_type_slug = "execution_profile"

    profile_name: str = Field(title="Profile Name", description="Human-friendly profile name, e.g., 'hello-n2'.")
    command_name: str = Field(title="Command Name", description="Which command this profile is intended for.")
    resource_class: Literal["cpu", "gpu"] = Field(default="cpu", title="Resource Class")

    nodes: int = Field(default=1, gt=0, title="Nodes (Baseline)")
    walltime: str = Field(default="00:05:00", title="Walltime (Baseline)", description="Format HH:MM:SS")
    ranks_per_node: int = Field(default=1, gt=0, title="Ranks per Node (Baseline)")
    threads_per_rank: int = Field(default=1, gt=0, title="Threads per Rank (Baseline)")

    launcher: Literal["mpirun", "mpiexec.hydra", "single"] = Field(default="mpiexec.hydra", title="Launcher")

    # Optional: command-specific environment setup (kept here, not in Command)
    modules: list[str] | None = Field(default=None, title="Modules (optional)")
    spack_setup: str | None = Field(default=None, title="Spack setup script (optional)")
    spack_load: list[str] | None = Field(default=None, title="Spack loads (optional)")
    env: dict[str, str] | None = Field(default=None, title="Extra env (optional)")


class MiyabiHPCProfileBlock(Block):
    """Miyabi-specific profile (WHERE/HOW to submit)."""

    _block_type_name = "Miyabi HPC Profile"
    _block_type_slug = "miyabi_hpc_profile"

    queue_cpu: str = Field(default="cpu", title="CPU Queue")
    queue_gpu: str = Field(default="gpu", title="GPU Queue")

    project_cpu: str | None = Field(default=None, title="CPU Project/Account")
    project_gpu: str | None = Field(default=None, title="GPU Project/Account")

    # Optional global init steps (system mechanics). Keep these minimal.
    module_init: list[str] | None = Field(default=None, title="Module init commands (optional)")
    spack_setup: str | None = Field(default=None, title="Default spack setup (optional)")

    executable_map: dict[str, str] = Field(
        default_factory=dict,
        title="Executable Map",
        description="Map from executable_key -> path. For this demo you can use a relative path like './hello_mpi'.",
    )
