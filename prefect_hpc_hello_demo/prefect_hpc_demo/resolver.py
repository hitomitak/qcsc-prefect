"""
Resolve (Command + ExecutionProfile + Tuning + MiyabiHPCProfile) into concrete run config.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from .blocks import CommandBlock, ExecutionProfileBlock, MiyabiHPCProfileBlock
from .models import Tuning


@dataclass(frozen=True)
class ResolvedRun:
    command_name: str
    profile_name: str
    executable: str
    launcher: Literal["mpirun", "mpiexec", "single"]
    queue: str
    project: str | None
    nodes: int
    walltime: str
    ranks_per_node: int
    threads_per_rank: int
    total_ranks: int
    argv: list[str]

    module_init: list[str]
    modules: list[str]
    spack_setup: str | None
    spack_load: list[str]
    env: dict[str, str]


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def resolve_run(
    *,
    cmd: CommandBlock,
    profile: ExecutionProfileBlock,
    hpc: MiyabiHPCProfileBlock,
    tuning: Tuning | None,
    user_args: list[str] | None,
) -> ResolvedRun:
    if profile.command_name != cmd.command_name:
        raise ValueError(
            f"ExecutionProfile '{profile.profile_name}' is for command '{profile.command_name}', "
            f"but you selected command '{cmd.command_name}'."
        )

    nodes = profile.nodes
    walltime = profile.walltime
    ranks_per_node = profile.ranks_per_node
    threads_per_rank = profile.threads_per_rank

    if tuning is not None:
        if tuning.nodes is not None:
            nodes = tuning.nodes
        if tuning.walltime is not None:
            walltime = tuning.walltime
        if tuning.ranks_per_node is not None:
            ranks_per_node = tuning.ranks_per_node
        if tuning.threads_per_rank is not None:
            threads_per_rank = tuning.threads_per_rank

    if profile.resource_class == "gpu":
        queue = hpc.queue_gpu
        project = hpc.project_gpu
    else:
        queue = hpc.queue_cpu
        project = hpc.project_cpu

    if cmd.executable_key not in hpc.executable_map:
        raise KeyError(f"Executable key '{cmd.executable_key}' not found in MiyabiHPCProfile.executable_map")
    executable = hpc.executable_map[cmd.executable_key]

    total_ranks = nodes * ranks_per_node

    argv = [executable]
    if cmd.default_args:
        argv += cmd.default_args
    if user_args:
        argv += user_args

    module_init = hpc.module_init or []
    modules = _dedup_keep_order(profile.modules or [])
    spack_setup = profile.spack_setup or hpc.spack_setup
    spack_load = _dedup_keep_order(profile.spack_load or [])
    env = dict(profile.env or {})

    return ResolvedRun(
        command_name=cmd.command_name,
        profile_name=profile.profile_name,
        executable=executable,
        launcher=profile.launcher,
        queue=queue,
        project=project,
        nodes=nodes,
        walltime=walltime,
        ranks_per_node=ranks_per_node,
        threads_per_rank=threads_per_rank,
        total_ranks=total_ranks,
        argv=argv,
        module_init=module_init,
        modules=modules,
        spack_setup=spack_setup,
        spack_load=spack_load,
        env=env,
    )
