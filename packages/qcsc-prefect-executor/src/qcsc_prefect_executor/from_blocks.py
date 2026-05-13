from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qcsc_prefect_adapters.fugaku.builder import FugakuJobRequest
from qcsc_prefect_adapters.miyabi.builder import MiyabiJobRequest
from qcsc_prefect_adapters.slurm.builder import SlurmJobRequest
from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
from qcsc_prefect_core.models.execution_profile import ExecutionProfile

from qcsc_prefect_executor.fugaku.run import run_fugaku_job
from qcsc_prefect_executor.miyabi.run import run_miyabi_job
from qcsc_prefect_executor.slurm.run import run_slurm_job

_EXECUTION_PROFILE_OVERRIDE_KEYS = {
    "num_nodes",
    "mpiprocs",
    "ompthreads",
    "walltime",
    "launcher",
    "mpi_options",
    "modules",
    "pre_commands",
    "environments",
}
_SCRIPT_SUFFIX_BY_TARGET = {
    "miyabi": ".pbs",
    "fugaku": ".pjm",
    "slurm": ".slurm",
}
_KNOWN_SCRIPT_SUFFIXES = frozenset(_SCRIPT_SUFFIX_BY_TARGET.values())


@dataclass(frozen=True)
class SubmissionTarget:
    """Scheduler routing information resolved from Prefect blocks.

    Attributes:
        hpc_target: Scheduler backend name, such as ``"miyabi"``,
            ``"fugaku"``, or ``"slurm"``.
        queue_name: Queue, partition, or resource-group name selected for the
            execution profile's resource class.
        project: Project, group, or account name selected for the resource
            class. This can be empty for scheduler targets that do not require
            an account.
    """

    hpc_target: str
    queue_name: str
    project: str


async def _resolve_loaded_block(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def _load_block(block_cls, block_name: str):
    return await _resolve_loaded_block(block_cls.load(block_name))


def _resolve_submission_target_from_loaded_blocks(
    hpc_block: HPCProfileBlock, resource_class: str
) -> SubmissionTarget:
    if resource_class == "gpu":
        return SubmissionTarget(
            hpc_target=hpc_block.hpc_target,
            queue_name=hpc_block.queue_gpu,
            project=hpc_block.project_gpu,
        )
    return SubmissionTarget(
        hpc_target=hpc_block.hpc_target,
        queue_name=hpc_block.queue_cpu,
        project=hpc_block.project_cpu,
    )


async def resolve_hpc_target(*, hpc_profile_block_name: str) -> str:
    """Load an ``HPCProfileBlock`` and return its scheduler target name.

    Args:
        hpc_profile_block_name: Prefect block document name for
            `qcsc_prefect_blocks.common.blocks.HPCProfileBlock`.

    Returns:
        The configured ``hpc_target`` value, for example ``"miyabi"``,
        ``"fugaku"``, or ``"slurm"``.
    """

    hpc_block = await _load_block(HPCProfileBlock, hpc_profile_block_name)
    return str(hpc_block.hpc_target)


async def resolve_submission_target(
    *,
    hpc_profile_block_name: str,
    execution_profile_block_name: str,
) -> SubmissionTarget:
    """Resolve scheduler routing from block names without submitting a job.

    This helper is useful when a flow needs to inspect the target queue or
    project before it creates scheduler-specific filenames or logs. It loads
    the ``HPCProfileBlock`` and ``ExecutionProfileBlock`` and chooses CPU or
    GPU queue/project fields from the execution profile's ``resource_class``.

    Args:
        hpc_profile_block_name: Prefect block document name for target-specific
            scheduler settings.
        execution_profile_block_name: Prefect block document name for
            scheduler-independent execution settings.

    Returns:
        Resolved scheduler target, queue/partition/resource group, and
        project/account values.
    """

    hpc_block = await _load_block(HPCProfileBlock, hpc_profile_block_name)
    execution_profile_block = await _load_block(ExecutionProfileBlock, execution_profile_block_name)
    return _resolve_submission_target_from_loaded_blocks(
        hpc_block, execution_profile_block.resource_class
    )


def build_scheduler_script_filename(script_stem: str, hpc_target: str) -> str:
    """Build a scheduler-specific script filename from a logical stem.

    Existing scheduler suffixes are replaced, while names without a known
    scheduler suffix receive the target suffix appended. For example,
    ``"batch"`` becomes ``"batch.pbs"`` for Miyabi and ``"batch.slurm"`` for
    Slurm; ``"batch.pbs"`` becomes ``"batch.pjm"`` for Fugaku.

    Args:
        script_stem: Logical script name or existing scheduler script filename.
        hpc_target: Scheduler target name.

    Returns:
        Script filename with the suffix required by the scheduler target.

    Raises:
        NotImplementedError: If ``hpc_target`` is not supported.
    """

    suffix = _SCRIPT_SUFFIX_BY_TARGET.get(hpc_target)
    if suffix is None:
        raise NotImplementedError(f"Unsupported hpc_target for script naming: {hpc_target}")

    script_path = Path(script_stem)
    if script_path.suffix in _KNOWN_SCRIPT_SUFFIXES:
        script_path = script_path.with_suffix(suffix)
    else:
        script_path = script_path.with_name(script_path.name + suffix)
    return str(script_path)


async def resolve_scheduler_script_filename(
    *,
    script_stem: str,
    hpc_profile_block_name: str,
) -> str:
    """Resolve scheduler target from blocks and return a matching filename.

    Args:
        script_stem: Logical script name or existing scheduler script filename.
        hpc_profile_block_name: Prefect block document name used to determine
            the scheduler target.

    Returns:
        Scheduler-specific script filename.
    """

    hpc_target = await resolve_hpc_target(hpc_profile_block_name=hpc_profile_block_name)
    return build_scheduler_script_filename(script_stem, hpc_target)


def _build_execution_profile(
    *,
    command_block: CommandBlock,
    execution_profile_block: ExecutionProfileBlock,
    user_args: list[str] | None,
    execution_profile_overrides: dict[str, Any] | None,
) -> ExecutionProfile:
    arguments = list(command_block.default_args)
    if user_args:
        arguments.extend(user_args)

    profile_kwargs: dict[str, Any] = {
        "command_key": command_block.command_name,
        "num_nodes": execution_profile_block.num_nodes,
        "mpiprocs": execution_profile_block.mpiprocs,
        "ompthreads": execution_profile_block.ompthreads,
        "walltime": execution_profile_block.walltime,
        "launcher": execution_profile_block.launcher,
        "mpi_options": list(execution_profile_block.mpi_options),
        "modules": list(execution_profile_block.modules),
        "pre_commands": list(getattr(execution_profile_block, "pre_commands", [])),
        "environments": dict(execution_profile_block.environments),
        "arguments": arguments,
    }
    if execution_profile_overrides:
        invalid_keys = sorted(set(execution_profile_overrides) - _EXECUTION_PROFILE_OVERRIDE_KEYS)
        if invalid_keys:
            raise ValueError(
                "Unsupported execution_profile_overrides keys: " + ", ".join(invalid_keys)
            )
        for key, value in execution_profile_overrides.items():
            if key in {"mpi_options", "modules", "pre_commands"} and value is not None:
                profile_kwargs[key] = list(value)
            elif key == "environments" and value is not None:
                profile_kwargs[key] = dict(value)
            else:
                profile_kwargs[key] = value

    return ExecutionProfile(
        **profile_kwargs,
    )


def _default_fugaku_job_name(command_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", command_name).strip("-")
    if not normalized:
        return "prefect-job"
    return normalized[:63]


async def run_job_from_blocks(
    *,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    work_dir: Path,
    script_filename: str,
    user_args: list[str] | None = None,
    watch_poll_interval: float = 10.0,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "hpc-job-metrics",
    fugaku_job_name: str | None = None,
    execution_profile_overrides: dict[str, Any] | None = None,
) -> Any:
    """Resolve Prefect blocks and execute a job on the configured HPC target.

    This is the main block-driven entrypoint for workflow authors. It loads the
    command, execution profile, and HPC profile blocks; converts them into the
    internal runtime models; creates the target-specific scheduler request; and
    dispatches to the Miyabi, Fugaku, or Slurm executor.

    Args:
        command_block_name: Prefect block document name for the command to run.
        execution_profile_block_name: Prefect block document name describing
            resources, launcher, environment, and default execution behavior.
        hpc_profile_block_name: Prefect block document name describing the
            scheduler target, queues, projects, and executable mapping.
        work_dir: Working directory where scheduler scripts and job outputs are
            created.
        script_filename: Logical or scheduler-specific script filename. The
            suffix is normalized for the resolved target.
        user_args: Optional extra command-line arguments appended after the
            command block's default arguments.
        watch_poll_interval: Seconds to wait between scheduler status polls.
        timeout_seconds: Optional maximum wait time for terminal job status.
        metrics_artifact_key: Prefect artifact key used for job metrics.
        fugaku_job_name: Optional Fugaku PJM job name. When omitted, a safe name
            is derived from the command name.
        execution_profile_overrides: Optional runtime overrides for selected
            execution profile fields, such as ``num_nodes`` or ``walltime``.

    Returns:
        A target-specific result object: ``MiyabiRunResult``,
        ``FugakuRunResult``, or ``SlurmRunResult``.

    Raises:
        ValueError: If the command and execution profile blocks refer to
            different command names, if a required project/group is missing, or
            if unsupported execution profile override keys are provided.
        KeyError: If the command's executable key is missing from the HPC
            profile's executable map.
        NotImplementedError: If the resolved ``hpc_target`` is unsupported.
    """
    command_block = await _load_block(CommandBlock, command_block_name)
    execution_profile_block = await _load_block(ExecutionProfileBlock, execution_profile_block_name)
    hpc_block = await _load_block(HPCProfileBlock, hpc_profile_block_name)

    if execution_profile_block.command_name != command_block.command_name:
        raise ValueError(
            f"ExecutionProfileBlock '{execution_profile_block_name}' is for command "
            f"'{execution_profile_block.command_name}', but command block "
            f"'{command_block_name}' is '{command_block.command_name}'."
        )

    executable = hpc_block.executable_map.get(command_block.executable_key)
    if not executable:
        raise KeyError(
            f"Executable key '{command_block.executable_key}' was not found in "
            f"HPCProfileBlock '{hpc_profile_block_name}'."
        )

    submission_target = _resolve_submission_target_from_loaded_blocks(
        hpc_block, execution_profile_block.resource_class
    )
    resolved_script_filename = build_scheduler_script_filename(
        script_filename,
        submission_target.hpc_target,
    )
    if submission_target.hpc_target in {"miyabi", "fugaku"} and not submission_target.project:
        raise ValueError("Project/Group is empty. Update HPCProfileBlock project_cpu/project_gpu.")

    exec_profile = _build_execution_profile(
        command_block=command_block,
        execution_profile_block=execution_profile_block,
        user_args=user_args,
        execution_profile_overrides=execution_profile_overrides,
    )
    resolved_work_dir = Path(work_dir).expanduser().resolve()

    if submission_target.hpc_target == "miyabi":
        req = MiyabiJobRequest(
            queue_name=submission_target.queue_name,
            project=submission_target.project,
            executable=executable,
        )
        return await run_miyabi_job(
            work_dir=resolved_work_dir,
            script_filename=resolved_script_filename,
            exec_profile=exec_profile,
            req=req,
            watch_poll_interval=watch_poll_interval,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
        )

    if submission_target.hpc_target == "fugaku":
        req = FugakuJobRequest(
            queue_name=submission_target.queue_name,
            project=submission_target.project,
            executable=executable,
            job_name=fugaku_job_name or _default_fugaku_job_name(command_block.command_name),
            gfscache=hpc_block.gfscache or "/vol0002",
            spack_modules=list(hpc_block.spack_modules) if hpc_block.spack_modules else [],
            mpi_options_for_pjm=list(hpc_block.mpi_options_for_pjm)
            if hpc_block.mpi_options_for_pjm
            else [],
            pjm_resources=list(hpc_block.pjm_resources) if hpc_block.pjm_resources else [],
        )
        return await run_fugaku_job(
            work_dir=resolved_work_dir,
            script_filename=resolved_script_filename,
            exec_profile=exec_profile,
            req=req,
            watch_poll_interval=watch_poll_interval,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
        )

    if submission_target.hpc_target == "slurm":
        req = SlurmJobRequest(
            partition=submission_target.queue_name,
            account=submission_target.project or None,
            executable=executable,
            qpu=hpc_block.slurm_qpu,
        )
        return await run_slurm_job(
            work_dir=resolved_work_dir,
            script_filename=resolved_script_filename,
            exec_profile=exec_profile,
            req=req,
            watch_poll_interval=watch_poll_interval,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
        )

    raise NotImplementedError(
        f"hpc_target='{submission_target.hpc_target}' is not supported yet by run_job_from_blocks."
    )
