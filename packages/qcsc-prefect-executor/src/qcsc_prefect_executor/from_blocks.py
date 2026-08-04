from __future__ import annotations

import asyncio
import getpass
import inspect
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from prefect.artifacts import create_table_artifact
from prefect.logging import get_run_logger
from qcsc_prefect_adapters.base.recovery import (
    IdentityRecoveryNotSupportedError,
    IdentityRecoveryRuntime,
    SchedulerJobCandidate,
    SchedulerJobIdentity,
)
from qcsc_prefect_adapters.base.subprocess import (
    DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS,
)
from qcsc_prefect_adapters.fugaku import builder as fugaku_builder
from qcsc_prefect_adapters.fugaku import runtime as fugaku_runtime
from qcsc_prefect_adapters.fugaku.builder import FugakuJobRequest
from qcsc_prefect_adapters.fugaku.runtime import FugakuPJMRuntime
from qcsc_prefect_adapters.local.runtime import LocalJobRequest
from qcsc_prefect_adapters.miyabi import builder as miyabi_builder
from qcsc_prefect_adapters.miyabi import runtime as miyabi_runtime
from qcsc_prefect_adapters.miyabi.builder import MiyabiJobRequest
from qcsc_prefect_adapters.miyabi.runtime import MiyabiPBSRuntime
from qcsc_prefect_adapters.slurm import builder as slurm_builder
from qcsc_prefect_adapters.slurm import runtime as slurm_runtime
from qcsc_prefect_adapters.slurm.builder import (
    SlurmJobRequest,
    build_slurm_job_identity,
)
from qcsc_prefect_adapters.slurm.runtime import (
    CancelError as SlurmCancelError,
)
from qcsc_prefect_adapters.slurm.runtime import (
    CancelNotFoundError as SlurmCancelNotFoundError,
)
from qcsc_prefect_adapters.slurm.runtime import (
    CancelRejectedError as SlurmCancelRejectedError,
)
from qcsc_prefect_adapters.slurm.runtime import (
    SlurmRuntime,
)
from qcsc_prefect_adapters.slurm.runtime import (
    TemporaryCancelError as SlurmTemporaryCancelError,
)
from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
from qcsc_prefect_core.models.execution_profile import ExecutionProfile
from qcsc_prefect_core.queue import QueueAwareSubmitGate, QueueProbe

from qcsc_prefect_executor.bulk.exceptions import (
    CancellationRequestedError,
    DuplicateJobKeyError,
    OperatorActionRequired,
    QueueFullError,
    RecoveryPending,
    SchedulerIdentityMismatchError,
    SpecHashMismatchError,
    SubmitError,
    SubmitOutcomeUnknownError,
    TemporarySubmitError,
)
from qcsc_prefect_executor.bulk.models import (
    BulkCancelOutcome,
    BulkJobDesiredState,
    BulkJobRecord,
    BulkJobSpec,
    BulkJobStatus,
    BulkRunResult,
    SubmittedJob,
    effective_execution_profile_block,
    effective_hpc_profile_block,
)
from qcsc_prefect_executor.bulk.native_manifest import create_native_bulk_group_manifests
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry
from qcsc_prefect_executor.bulk.spec_hash import build_bulk_spec_hash
from qcsc_prefect_executor.cloud_logs import (
    CloudJobSummary,
    CloudLogPolicy,
    emit_cloud_job_logs,
    read_log_text,
    resolve_cloud_log_policy,
)
from qcsc_prefect_executor.fugaku.run import run_fugaku_job
from qcsc_prefect_executor.local.run import run_local_job
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
DEFAULT_SLURM_RECOVERY_GRACE_SECONDS = 120.0
DEFAULT_SLURM_CLOCK_SKEW_MARGIN_SECONDS = 60.0


def _validate_slurm_recovery_settings(
    *,
    recovery_grace_seconds: float,
    clock_skew_margin_seconds: float,
    scheduler_command_timeout_seconds: float | None,
) -> None:
    if recovery_grace_seconds < 0:
        raise ValueError("slurm_recovery_grace_seconds must be non-negative.")
    if clock_skew_margin_seconds < 0:
        raise ValueError("slurm_clock_skew_margin_seconds must be non-negative.")
    if scheduler_command_timeout_seconds is not None and scheduler_command_timeout_seconds <= 0:
        raise ValueError("scheduler_command_timeout_seconds must be greater than 0.")


def resolve_identity_recovery_runtime(hpc_target: str) -> IdentityRecoveryRuntime:
    """Resolve the optional identity-search capability for one HPC target."""

    normalized_target = str(hpc_target).strip().lower()
    if normalized_target == "slurm":
        return SlurmRuntime()
    raise IdentityRecoveryNotSupportedError(
        f"hpc_target={hpc_target!r} does not implement find_candidates_by_identity()."
    )


@dataclass(frozen=True)
class SubmissionTarget:
    """Execution routing information resolved from Prefect blocks.

    Attributes:
        hpc_target: Runtime target name, such as ``"local"``, ``"miyabi"``,
            ``"fugaku"``, or ``"slurm"``.
        queue_name: Queue, partition, or resource-group name selected for the
            execution profile's resource class. Empty for local execution.
        project: Project, group, or account name selected for the resource
            class. Empty for local execution and scheduler targets that do not
            require an account.
    """

    hpc_target: str
    queue_name: str
    project: str


@dataclass(frozen=True)
class _PreparedBlockJob:
    submission_target: SubmissionTarget
    work_dir: Path
    script_filename: str | None
    exec_profile: ExecutionProfile
    req: Any


def _resolved_bulk_spec_payload(
    prepared: _PreparedBlockJob,
    *,
    input_digest: str | None,
    code_digest: str | None,
    environment_digest: str | None,
) -> dict[str, Any]:
    scheduler_request = asdict(prepared.req)
    for derived_identity_field in ("job_name", "job_comment", "comment"):
        scheduler_request.pop(derived_identity_field, None)

    return {
        "command": {
            "command_key": prepared.exec_profile.command_key,
            "executable": scheduler_request.get("executable"),
            "arguments": list(prepared.exec_profile.arguments),
        },
        "execution_profile": asdict(prepared.exec_profile),
        "scheduler": {
            "target": prepared.submission_target.hpc_target,
            "queue": prepared.submission_target.queue_name,
            "account_or_project": prepared.submission_target.project,
            "request": scheduler_request,
        },
        "caller_digests": {
            "input": input_digest,
            "code": code_digest,
            "environment": environment_digest,
        },
    }


def _resolved_bulk_spec_hash(
    prepared: _PreparedBlockJob,
    *,
    input_digest: str | None = None,
    code_digest: str | None = None,
    environment_digest: str | None = None,
) -> str:
    return build_bulk_spec_hash(
        _resolved_bulk_spec_payload(
            prepared,
            input_digest=input_digest,
            code_digest=code_digest,
            environment_digest=environment_digest,
        )
    )


def _with_slurm_identity(
    prepared: _PreparedBlockJob,
    *,
    job_key: str,
    spec_hash: str,
) -> tuple[_PreparedBlockJob, str, str]:
    if prepared.submission_target.hpc_target != "slurm":
        raise ValueError("Slurm identity can only be added to a Slurm job request.")
    if not isinstance(prepared.req, SlurmJobRequest):
        raise TypeError("Prepared Slurm job has an unexpected request type.")
    identity = build_slurm_job_identity(job_key=job_key, spec_hash=spec_hash)
    return (
        replace(
            prepared,
            req=replace(
                prepared.req,
                job_name=identity.job_name,
                comment=identity.comment,
            ),
        ),
        identity.job_name,
        identity.comment,
    )


async def _resolve_loaded_block(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def _load_block(block_cls, block_name: str):
    return await _resolve_loaded_block(block_cls.load(block_name))


def _resolve_submission_target_from_loaded_blocks(
    hpc_block: HPCProfileBlock, resource_class: str
) -> SubmissionTarget:
    if hpc_block.hpc_target == "local":
        return SubmissionTarget(hpc_target="local", queue_name="", project="")
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
    """Load an ``HPCProfileBlock`` and return its execution target name.

    Args:
        hpc_profile_block_name: Prefect block document name for
            `qcsc_prefect_blocks.common.blocks.HPCProfileBlock`.

    Returns:
        The configured ``hpc_target`` value, for example ``"local"``,
        ``"miyabi"``, ``"fugaku"``, or ``"slurm"``.
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


def _command_args_to_user_args(command_args: dict[str, Any] | None) -> list[str] | None:
    if not command_args:
        return None

    user_args: list[str] = []
    for key, value in sorted(command_args.items(), key=lambda item: str(item[0])):
        option = str(key) if str(key).startswith("-") else "--" + str(key).replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            user_args.append(option)
        elif isinstance(value, (list, tuple)):
            for item in value:
                user_args.extend([option, str(item)])
        else:
            user_args.extend([option, str(value)])
    return user_args


def _resolve_named_argument(
    *,
    preferred: str | None,
    alias: str | None,
    label: str,
) -> str:
    value = preferred if preferred is not None else alias
    if value is None or not str(value).strip():
        raise ValueError(f"{label} is required.")
    return str(value)


def _ensure_registry_can_submit(*, registry: BulkJobRegistry, job_key: str) -> None:
    record = registry.get_job(job_key)
    if record is None:
        return
    if record.desired_state == BulkJobDesiredState.CANCEL_REQUESTED:
        raise CancellationRequestedError(job_key=job_key)
    if record.status.is_submit_candidate:
        return

    scheduler_part = (
        f", scheduler_job_id={record.scheduler_job_id}" if record.scheduler_job_id else ""
    )
    raise DuplicateJobKeyError(
        f"Bulk job key {job_key!r} already has status {record.status.value}"
        f"{scheduler_part}. Use a fresh job_key or registry for a new scheduler job."
    )


async def _resolve_registered_bulk_spec_hashes(
    *,
    jobs: list[BulkJobSpec],
    registry: BulkJobRegistry,
    command_block: str,
    execution_profile_block: str,
    hpc_profile_block: str,
) -> list[BulkJobSpec]:
    resolved_jobs: list[BulkJobSpec] = []
    for job in jobs:
        existing = registry.get_job(job.job_key)
        if job.spec_hash is None and (existing is None or existing.spec_hash is None):
            resolved_jobs.append(job)
            continue

        prepared = await _prepare_job_from_blocks(
            command_block_name=command_block,
            execution_profile_block_name=effective_execution_profile_block(
                job,
                execution_profile_block,
            ),
            hpc_profile_block_name=effective_hpc_profile_block(
                job,
                hpc_profile_block,
            ),
            work_dir=job.work_dir,
            script_filename=job.job_key,
            user_args=_command_args_to_user_args(job.command_args),
        )
        resolved_jobs.append(
            replace(
                job,
                spec_hash=_resolved_bulk_spec_hash(
                    prepared,
                    input_digest=job.input_digest,
                    code_digest=job.code_digest,
                    environment_digest=job.environment_digest,
                ),
            )
        )
    return resolved_jobs


async def _resolve_default_bulk_queue_probe(
    *,
    hpc_profile_block: str,
    execution_profile_block: str,
    max_active_jobs: int,
    safety_margin: int,
    submit_mode: Literal["single", "native_bulk"] = "single",
    slurm_user: str | None = None,
    scheduler_command_timeout_seconds: float | None = (DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS),
) -> QueueProbe:
    submission_target = await resolve_submission_target(
        hpc_profile_block_name=hpc_profile_block,
        execution_profile_block_name=execution_profile_block,
    )
    if submission_target.hpc_target == "fugaku":
        from qcsc_prefect_adapters.fugaku.queue import FugakuQueueProbe

        return FugakuQueueProbe(
            max_active_jobs=max_active_jobs,
            safety_margin=safety_margin,
            project=submission_target.project,
            queue=submission_target.queue_name,
            capacity_mode="native_bulk" if submit_mode == "native_bulk" else "single",
        )

    if submission_target.hpc_target == "slurm":
        from qcsc_prefect_adapters.slurm.queue import SlurmQueueProbe

        resolved_user = str(slurm_user or getpass.getuser()).strip()
        if not resolved_user:
            raise ValueError("slurm_user must be non-empty for the default queue probe.")
        return SlurmQueueProbe(
            max_active_jobs=max_active_jobs,
            user=resolved_user,
            account=submission_target.project,
            partition=submission_target.queue_name,
            scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
        )

    raise ValueError(
        "queue_probe is required for bulk execution when hpc_target is "
        f"{submission_target.hpc_target!r}. Pass a scheduler-specific QueueProbe."
    )


def _build_bulk_run_result(
    *,
    registry: BulkJobRegistry,
    total_jobs: int,
) -> BulkRunResult:
    counts = registry.status_counts()
    failed_jobs = [
        record.job_key
        for record in registry.get_all_jobs()
        if record.status == BulkJobStatus.FAILED
    ]
    operator_action_required_jobs = list(
        dict.fromkeys(
            [record.job_key for record in registry.get_awaiting_operator_jobs()]
            + [record.job_key for record in registry.get_ambiguous_cancel_dispatches()]
        )
    )
    return BulkRunResult(
        total_jobs=total_jobs,
        status_counts=counts,
        succeeded=counts.get(BulkJobStatus.SUCCEEDED.value, 0),
        failed=counts.get(BulkJobStatus.FAILED.value, 0),
        cancelled=counts.get(BulkJobStatus.CANCELLED.value, 0),
        submit_deferred=counts.get(BulkJobStatus.SUBMIT_DEFERRED.value, 0),
        unknown=counts.get(BulkJobStatus.UNKNOWN.value, 0),
        registry_path=registry.path,
        failed_jobs=failed_jobs,
        prepared=counts.get(BulkJobStatus.PREPARED.value, 0),
        awaiting_operator=counts.get(BulkJobStatus.AWAITING_OPERATOR.value, 0),
        operator_action_required_jobs=operator_action_required_jobs,
    )


def _has_failed_jobs(registry: BulkJobRegistry) -> bool:
    return registry.status_counts().get(BulkJobStatus.FAILED.value, 0) > 0


def _has_operator_holds(registry: BulkJobRegistry) -> bool:
    return bool(registry.get_awaiting_operator_jobs() or registry.get_ambiguous_cancel_dispatches())


def _safe_bulk_group_key(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.=-]+", "-", value).strip("-")
    return safe or "bulk-group"


def _bulk_group_key_for_jobs(jobs: list[BulkJobRecord]) -> str:
    first = _safe_bulk_group_key(jobs[0].job_key)
    last = _safe_bulk_group_key(jobs[-1].job_key)
    stage = _safe_bulk_group_key(str(jobs[0].stage_id or "stage"))
    return f"native-bulk-{stage}-{first}-{last}-{len(jobs)}"[:180]


def _chunk_records(
    records: list[BulkJobRecord],
    *,
    chunk_size: int,
) -> list[list[BulkJobRecord]]:
    if chunk_size <= 0:
        raise ValueError("max_bulk_group_size must be positive.")
    return [records[index : index + chunk_size] for index in range(0, len(records), chunk_size)]


def _submit_limit_for_cycle(
    *,
    registry: BulkJobRegistry,
    initial_submit_count: int | None,
    max_submit_per_refill: int,
) -> int:
    if registry.bootstrap_done():
        return max(0, int(max_submit_per_refill))
    if initial_submit_count is None:
        return max(0, int(max_submit_per_refill))
    return max(0, int(initial_submit_count))


def _queue_available_slots(queue_probe: QueueProbe) -> int:
    try:
        capacity = queue_probe.get_capacity()
    except Exception:
        return 0
    return max(0, int(capacity.available_slots))


def _target_active_slots(
    *,
    registry: BulkJobRegistry,
    target_active_jobs: int | None,
) -> int | None:
    if target_active_jobs is None:
        return None
    return max(0, int(target_active_jobs) - registry.count_active_jobs())


def _native_bulk_submit_count(
    *,
    registry: BulkJobRegistry,
    queue_probe: QueueProbe,
    submit_limit: int,
    target_active_jobs: int | None,
) -> int:
    limits = [
        max(0, int(submit_limit)),
        _queue_available_slots(queue_probe),
        registry.count_submit_candidates(),
    ]
    target_slots = _target_active_slots(
        registry=registry,
        target_active_jobs=target_active_jobs,
    )
    if target_slots is not None:
        limits.append(target_slots)
    return max(0, min(limits))


def _validate_native_bulk_candidates(jobs: list[BulkJobRecord]) -> None:
    missing_stage = [
        job.job_key for job in jobs if not job.stage_id or not str(job.stage_id).strip()
    ]
    if missing_stage:
        raise ValueError(
            "submit_mode='native_bulk' requires stage_id for every selected job: "
            + ", ".join(missing_stage)
        )

    overridden_blocks = [
        job.job_key
        for job in jobs
        if job.execution_profile_block is not None or job.hpc_profile_block is not None
    ]
    if overridden_blocks:
        raise ValueError(
            "submit_mode='native_bulk' does not support per-job execution_profile_block "
            "or hpc_profile_block overrides: " + ", ".join(overridden_blocks)
        )


def _validate_native_bulk_specs(jobs: list[BulkJobSpec]) -> None:
    overridden_blocks = [
        job.job_key
        for job in jobs
        if job.execution_profile_block is not None or job.hpc_profile_block is not None
    ]
    if overridden_blocks:
        raise ValueError(
            "submit_mode='native_bulk' does not support per-job execution_profile_block "
            "or hpc_profile_block overrides: " + ", ".join(overridden_blocks)
        )


def _mark_deferred_if_needed(
    *,
    registry: BulkJobRegistry,
    job_key: str,
    error: str | None,
) -> None:
    record = registry.get_job(job_key)
    if record is not None and record.status == BulkJobStatus.SUBMIT_DEFERRED:
        return
    registry.mark_submit_deferred(job_key, error=error)


def _mark_failed_if_needed(
    *,
    registry: BulkJobRegistry,
    job_key: str,
    error: str | None,
) -> None:
    record = registry.get_job(job_key)
    if record is not None and record.status in {
        BulkJobStatus.FAILED,
        BulkJobStatus.SUCCEEDED,
        BulkJobStatus.CANCELLED,
    }:
        return
    registry.mark_failed(job_key, error=error)


async def _prepare_job_from_blocks(
    *,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    work_dir: Path,
    script_filename: str | None,
    user_args: list[str] | None = None,
    fugaku_job_name: str | None = None,
    execution_profile_overrides: dict[str, Any] | None = None,
) -> _PreparedBlockJob:
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
    if submission_target.hpc_target == "local":
        resolved_script_filename = None
    else:
        if not script_filename:
            raise ValueError("script_filename is required for scheduler execution targets.")
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

    if submission_target.hpc_target == "local":
        req = LocalJobRequest(executable=executable)
    elif submission_target.hpc_target == "miyabi":
        req = MiyabiJobRequest(
            queue_name=submission_target.queue_name,
            project=submission_target.project,
            executable=executable,
        )
    elif submission_target.hpc_target == "fugaku":
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
    elif submission_target.hpc_target == "slurm":
        req = SlurmJobRequest(
            partition=submission_target.queue_name,
            account=submission_target.project or None,
            executable=executable,
            qpu=hpc_block.slurm_qpu,
            memory=getattr(hpc_block, "slurm_memory", None),
            ntasks=getattr(hpc_block, "slurm_ntasks", None),
        )
    else:
        raise NotImplementedError(
            f"hpc_target='{submission_target.hpc_target}' is not supported yet by "
            "run_job_from_blocks."
        )

    return _PreparedBlockJob(
        submission_target=submission_target,
        work_dir=resolved_work_dir,
        script_filename=resolved_script_filename,
        exec_profile=exec_profile,
        req=req,
    )


def _write_script_for_prepared_job(prepared: _PreparedBlockJob) -> Path:
    target = prepared.submission_target.hpc_target
    if prepared.script_filename is None:
        raise ValueError(f"hpc_target={target!r} does not use scheduler job scripts.")
    if target == "miyabi":
        script_text = miyabi_builder.render_script(
            work_dir=prepared.work_dir,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
        )
        return miyabi_builder.write_script_file(
            work_dir=prepared.work_dir,
            filename=prepared.script_filename,
            text=script_text,
        )

    if target == "fugaku":
        script_basename = Path(prepared.script_filename).name
        script_text = fugaku_builder.render_script(
            work_dir=prepared.work_dir,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
            script_basename=script_basename,
        )
        return fugaku_builder.write_script_file(
            work_dir=prepared.work_dir,
            filename=prepared.script_filename,
            text=script_text,
        )

    if target == "slurm":
        script_text = slurm_builder.render_script(
            work_dir=prepared.work_dir,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
        )
        return slurm_builder.write_script_file(
            work_dir=prepared.work_dir,
            filename=prepared.script_filename,
            text=script_text,
        )

    raise NotImplementedError(f"Unsupported hpc_target for submit: {target}")


def _exception_text(exc: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        parts.append(str(current))
        current = current.__cause__
    return "\n".join(part for part in parts if part)


def _classify_submit_exception(exc: BaseException) -> SubmitError:
    if isinstance(exc, SubmitError):
        return exc
    if getattr(exc, "submit_outcome_unknown", False):
        return SubmitOutcomeUnknownError(_exception_text(exc))

    message = _exception_text(exc).lower()
    queue_full_patterns = {
        "queue full",
        "job limit",
        "submit limit",
        "accept limit",
        "ru-accept",
        "too many jobs",
        "maximum number of jobs",
        "exceed",
        "exceeded",
    }
    temporary_patterns = {
        "temporar",
        "try again",
        "unavailable",
        "timeout",
        "timed out",
        "busy",
        "connection",
        "rate limit",
    }

    if any(pattern in message for pattern in queue_full_patterns):
        return QueueFullError(_exception_text(exc))
    if any(pattern in message for pattern in temporary_patterns):
        return TemporarySubmitError(_exception_text(exc))
    return SubmitError(_exception_text(exc))


async def _submit_prepared_job(
    prepared: _PreparedBlockJob,
    *,
    fugaku_no_check_directory: bool = False,
    slurm_submit_timeout_seconds: float | None = DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS,
) -> str:
    target = prepared.submission_target.hpc_target
    if target == "local":
        raise ValueError(
            "Scheduler submit APIs do not support hpc_target='local'. "
            "Use run_job_from_blocks() for local execution."
        )
    script_path = _write_script_for_prepared_job(prepared)

    if target == "miyabi":
        submit = await MiyabiPBSRuntime().submit(script_path, cwd=prepared.work_dir)
    elif target == "fugaku":
        submit = await FugakuPJMRuntime(no_check_directory=fugaku_no_check_directory).submit(
            script_path, cwd=prepared.work_dir
        )
    elif target == "slurm":
        submit = await SlurmRuntime().submit(
            script_path,
            cwd=prepared.work_dir,
            timeout_seconds=slurm_submit_timeout_seconds,
        )
    else:
        raise NotImplementedError(f"Unsupported hpc_target for submit: {target}")

    return submit.job_id


def _parse_registry_datetime(value: str | None, *, field_name: str) -> datetime:
    if not value:
        raise ValueError(f"{field_name} is required for Slurm recovery.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _raise_operator_hold(
    *,
    registry: BulkJobRegistry,
    job_key: str,
    reason: str,
) -> None:
    registry.mark_awaiting_operator(job_key, reason)
    raise OperatorActionRequired(job_keys=[job_key], reason=reason)


def _attach_recovered_slurm_candidate(
    *,
    registry: BulkJobRegistry,
    record: BulkJobRecord,
    candidate: SchedulerJobCandidate,
) -> SubmittedJob:
    try:
        attached = registry.mark_submitted(record.job_key, candidate.job_id)
        if not attached:
            raise SubmitOutcomeUnknownError(
                f"Slurm job {candidate.job_id} was found, but registry row "
                f"{record.job_key!r} no longer permits automatic attachment."
            )
    except SchedulerIdentityMismatchError:
        registry.mark_awaiting_operator(
            record.job_key,
            "A different Slurm job ID was stored during candidate attachment",
        )
        raise
    except Exception as exc:
        registry.record_prepared_error(
            record.job_key,
            f"Found Slurm job {candidate.job_id} but could not persist the attachment: "
            f"{_exception_text(exc)}",
        )
        raise SubmitOutcomeUnknownError(
            f"Found Slurm job {candidate.job_id} for {record.job_key!r}, but its "
            "registry attachment could not be persisted. Do not resubmit."
        ) from exc

    attached = registry.get_job(record.job_key)
    if attached is None:
        raise SubmitOutcomeUnknownError(
            f"Slurm job {candidate.job_id} was attached but registry row "
            f"{record.job_key!r} could not be reloaded."
        )
    status, error = _monitor_status_from_scheduler_row(
        hpc_target="slurm",
        row={"JobID": candidate.job_id, "State": candidate.state},
        record=attached,
    )
    if status == BulkJobStatus.AWAITING_OPERATOR:
        reason = error or f"Slurm job {candidate.job_id} requires operator reconciliation"
        _raise_operator_hold(
            registry=registry,
            job_key=record.job_key,
            reason=reason,
        )
    _update_registry_for_monitor_status(
        registry=registry,
        job_key=record.job_key,
        status=status,
        error=error,
    )
    updated = registry.get_job(record.job_key)
    if updated is None:
        raise SubmitOutcomeUnknownError(
            f"Attached Slurm job {candidate.job_id}, but registry row "
            f"{record.job_key!r} could not be reloaded."
        )
    return SubmittedJob(
        job_key=record.job_key,
        scheduler_job_id=candidate.job_id,
        status=updated.status,
        work_dir=updated.work_dir,
    )


async def _reconcile_prepared_slurm_job(
    *,
    registry: BulkJobRegistry,
    record: BulkJobRecord,
    prepared: _PreparedBlockJob,
    slurm_user: str,
    recovery_grace_seconds: float,
    clock_skew_margin_seconds: float,
    scheduler_command_timeout_seconds: float | None,
    now: datetime | None = None,
) -> SubmittedJob:
    if record.status != BulkJobStatus.PREPARED:
        raise DuplicateJobKeyError(
            f"Bulk job key {record.job_key!r} is not PREPARED; current status is "
            f"{record.status.value}."
        )
    if not record.spec_hash or not record.job_name or not record.job_comment:
        _raise_operator_hold(
            registry=registry,
            job_key=record.job_key,
            reason="PREPARED row is missing immutable spec or Slurm identity fields",
        )
    if not isinstance(prepared.req, SlurmJobRequest):
        _raise_operator_hold(
            registry=registry,
            job_key=record.job_key,
            reason="prepared request is not a SlurmJobRequest",
        )

    prepared_at = _parse_registry_datetime(record.prepared_at, field_name="prepared_at")
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("recovery current time must be timezone-aware.")
    skew = timedelta(seconds=max(0.0, float(clock_skew_margin_seconds)))
    search_start = prepared_at - skew
    search_end = current_time + skew

    identity = SchedulerJobIdentity(
        search_token=record.job_name,
        stable_identity=record.job_comment,
        owner=slurm_user,
        search_start=search_start,
        search_end=search_end,
        metadata={
            "account": prepared.req.account or "",
            "partition": prepared.req.partition,
        },
        timeout_seconds=scheduler_command_timeout_seconds,
    )

    try:
        recovery_runtime = resolve_identity_recovery_runtime(prepared.submission_target.hpc_target)
        candidates = await recovery_runtime.find_candidates_by_identity(identity)
    except asyncio.CancelledError:
        raise
    except IdentityRecoveryNotSupportedError:
        raise
    except Exception as exc:
        registry.record_prepared_error(
            record.job_key,
            f"Slurm identity search failed; PREPARED is preserved: {_exception_text(exc)}",
        )
        elapsed = max(0.0, (current_time - prepared_at).total_seconds())
        raise RecoveryPending(
            job_key=record.job_key,
            retry_after_seconds=max(0.0, recovery_grace_seconds - elapsed),
        ) from exc

    if any(not candidate.identity_matches for candidate in candidates):
        reason = (
            "Slurm candidate comment does not match the immutable spec-derived "
            "identity; automatic attach is forbidden"
        )
        registry.mark_awaiting_operator(record.job_key, reason)
        raise SchedulerIdentityMismatchError(
            job_key=record.job_key,
            stored_spec_hash=record.spec_hash,
        )

    metadata_errors = [
        candidate.metadata_error for candidate in candidates if candidate.metadata_error
    ]
    if metadata_errors:
        _raise_operator_hold(
            registry=registry,
            job_key=record.job_key,
            reason="; ".join(metadata_errors),
        )

    if len(candidates) > 1:
        _raise_operator_hold(
            registry=registry,
            job_key=record.job_key,
            reason=(
                f"Found {len(candidates)} matching Slurm allocations; automatic "
                "selection is forbidden"
            ),
        )
    if len(candidates) == 1:
        return _attach_recovered_slurm_candidate(
            registry=registry,
            record=record,
            candidate=candidates[0],
        )

    elapsed = max(0.0, (current_time - prepared_at).total_seconds())
    if elapsed < recovery_grace_seconds:
        remaining = max(0.0, recovery_grace_seconds - elapsed)
        registry.record_prepared_error(
            record.job_key,
            "No Slurm identity candidate is visible yet; PREPARED is preserved "
            f"for {remaining:g} more grace seconds.",
        )
        raise RecoveryPending(
            job_key=record.job_key,
            retry_after_seconds=remaining,
        )

    _raise_operator_hold(
        registry=registry,
        job_key=record.job_key,
        reason=(
            "No matching Slurm allocation was found after the scheduler-visibility grace period"
        ),
    )


async def _submit_claimed_slurm_job(
    *,
    registry: BulkJobRegistry,
    record: BulkJobRecord,
    prepared: _PreparedBlockJob,
    slurm_user: str,
    recovery_grace_seconds: float,
    clock_skew_margin_seconds: float,
    scheduler_command_timeout_seconds: float | None,
) -> SubmittedJob:
    try:
        script_path = _write_script_for_prepared_job(prepared)
    except Exception as exc:
        classified = SubmitError(
            f"Failed to write the Slurm script before sbatch: {_exception_text(exc)}"
        )
        registry.mark_failed(record.job_key, error=str(classified))
        raise classified from exc

    try:
        submit = await SlurmRuntime().submit(
            script_path,
            cwd=prepared.work_dir,
            timeout_seconds=scheduler_command_timeout_seconds,
        )
    except asyncio.CancelledError:
        registry.record_prepared_error(
            record.job_key,
            "Submission coroutine was cancelled after PREPARED; scheduler outcome "
            "is unknown and automatic resubmit is forbidden.",
        )
        raise
    except Exception as exc:
        if getattr(exc, "submission_definitely_rejected", False) or isinstance(
            exc,
            QueueFullError | TemporarySubmitError,
        ):
            classified = _classify_submit_exception(exc)
            if isinstance(classified, QueueFullError | TemporarySubmitError):
                registry.release_prepared_for_retry(record.job_key, str(classified))
            else:
                registry.mark_failed(record.job_key, error=str(classified))
            raise classified from exc

        registry.record_prepared_error(
            record.job_key,
            "Slurm submission outcome is unknown; automatic resubmit is forbidden: "
            f"{_exception_text(exc)}",
        )
        refreshed = registry.get_job(record.job_key)
        if refreshed is None:
            raise SubmitOutcomeUnknownError(
                f"Slurm submission outcome is unknown and registry row "
                f"{record.job_key!r} cannot be reloaded."
            ) from exc
        return await _reconcile_prepared_slurm_job(
            registry=registry,
            record=refreshed,
            prepared=prepared,
            slurm_user=slurm_user,
            recovery_grace_seconds=recovery_grace_seconds,
            clock_skew_margin_seconds=clock_skew_margin_seconds,
            scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
        )

    try:
        submitted = registry.mark_submitted(record.job_key, submit.job_id)
        if not submitted:
            registry.mark_awaiting_operator(
                record.job_key,
                "Slurm accepted the job after its PREPARED claim changed state; "
                "automatic attachment is forbidden",
            )
            raise SubmitOutcomeUnknownError(
                f"Slurm accepted job {submit.job_id} for {record.job_key!r}, but "
                "the registry claim no longer permits automatic attachment."
            )
    except SchedulerIdentityMismatchError:
        registry.mark_awaiting_operator(
            record.job_key,
            "Slurm returned a job ID different from the ID attached concurrently",
        )
        raise
    except Exception as exc:
        registry.record_prepared_error(
            record.job_key,
            f"sbatch returned job id {submit.job_id}, but the registry update failed: "
            f"{_exception_text(exc)}",
        )
        raise SubmitOutcomeUnknownError(
            f"Slurm accepted job {submit.job_id} for {record.job_key!r}, but the "
            "registry update failed. PREPARED must be reconciled; do not resubmit."
        ) from exc

    submitted_record = registry.get_job(record.job_key)
    return SubmittedJob(
        job_key=record.job_key,
        scheduler_job_id=submit.job_id,
        status=(
            submitted_record.status if submitted_record is not None else BulkJobStatus.SUBMITTED
        ),
        work_dir=(submitted_record.work_dir if submitted_record is not None else prepared.work_dir),
    )


def _write_fugaku_native_bulk_script(
    *,
    prepared: _PreparedBlockJob,
    bulk_manifest_dir: Path,
) -> Path:
    if prepared.submission_target.hpc_target != "fugaku":
        raise ValueError("submit_mode='native_bulk' is only supported for Fugaku/PJM.")

    script_basename = Path(prepared.script_filename).name
    script_text = fugaku_builder.render_manifest_bulk_script(
        work_dir=prepared.work_dir,
        bulk_manifest_dir=bulk_manifest_dir,
        exec_profile=prepared.exec_profile,
        req=prepared.req,
        script_basename=script_basename,
    )
    return fugaku_builder.write_script_file(
        work_dir=prepared.work_dir,
        filename=prepared.script_filename,
        text=script_text,
    )


async def _submit_native_bulk_group_from_blocks(
    *,
    registry: BulkJobRegistry,
    jobs: list[BulkJobRecord],
    command_block: str,
    execution_profile_block: str,
    hpc_profile_block: str,
    fugaku_no_check_directory: bool = False,
) -> str:
    resolved_specs: list[BulkJobSpec] = []
    for job in jobs:
        resolved_job = await _prepare_job_from_blocks(
            command_block_name=command_block,
            execution_profile_block_name=execution_profile_block,
            hpc_profile_block_name=hpc_profile_block,
            work_dir=job.work_dir,
            script_filename=job.job_key,
            user_args=_command_args_to_user_args(job.command_args),
        )
        resolved_specs.append(
            BulkJobSpec(
                job_key=job.job_key,
                work_dir=job.work_dir,
                command_args=job.command_args,
                wave_id=job.wave_id,
                target_id=job.target_id,
                stage_id=job.stage_id,
                priority=job.priority,
                expected_outputs=job.expected_outputs,
                max_submit_attempts=job.max_submit_attempts,
                execution_profile_block=job.execution_profile_block,
                hpc_profile_block=job.hpc_profile_block,
                spec_hash=_resolved_bulk_spec_hash(
                    resolved_job,
                    input_digest=job.input_digest,
                    code_digest=job.code_digest,
                    environment_digest=job.environment_digest,
                ),
                input_digest=job.input_digest,
                code_digest=job.code_digest,
                environment_digest=job.environment_digest,
                job_name=job.job_name,
                job_comment=job.job_comment,
            )
        )
    registry.upsert_jobs(resolved_specs)

    bulk_group_key = _bulk_group_key_for_jobs(jobs)
    bulk_group_dir = registry.path.parent / "native-bulk" / bulk_group_key
    manifest_group = create_native_bulk_group_manifests(
        bulk_group_dir=bulk_group_dir,
        jobs=jobs,
    )
    prepared = await _prepare_job_from_blocks(
        command_block_name=command_block,
        execution_profile_block_name=execution_profile_block,
        hpc_profile_block_name=hpc_profile_block,
        work_dir=manifest_group.bulk_group_dir,
        script_filename=bulk_group_key,
        user_args=None,
        fugaku_job_name=bulk_group_key[:63],
    )
    script_path = _write_fugaku_native_bulk_script(
        prepared=prepared,
        bulk_manifest_dir=manifest_group.manifest_dir,
    )
    parent_job_id = await FugakuPJMRuntime(
        no_check_directory=fugaku_no_check_directory
    ).submit_bulk(
        script_path,
        manifest_group.bulk_count,
        cwd=manifest_group.bulk_group_dir,
    )

    for bulk_index, job in enumerate(jobs):
        scheduler_subjob_id = f"{parent_job_id}[{bulk_index}]"
        registry.mark_submitted(
            job.job_key,
            scheduler_subjob_id,
            submit_mode="native_bulk",
            bulk_group_key=bulk_group_key,
            bulk_parent_job_id=parent_job_id,
            bulk_index=bulk_index,
            scheduler_subjob_id=scheduler_subjob_id,
        )

    return parent_job_id


async def _submit_native_bulk_cycle_from_blocks(
    *,
    registry: BulkJobRegistry,
    command_block: str,
    execution_profile_block: str,
    hpc_profile_block: str,
    queue_probe: QueueProbe,
    submit_limit: int,
    max_bulk_group_size: int,
    target_active_jobs: int | None,
    stop_on_first_failure: bool,
    fugaku_no_check_directory: bool = False,
) -> bool:
    submit_count = _native_bulk_submit_count(
        registry=registry,
        queue_probe=queue_probe,
        submit_limit=submit_limit,
        target_active_jobs=target_active_jobs,
    )
    if submit_count <= 0:
        return False

    selected_jobs = registry.get_submit_candidates_fifo(limit=submit_count)
    if not selected_jobs:
        return False

    _validate_native_bulk_candidates(selected_jobs)
    for chunk in _chunk_records(selected_jobs, chunk_size=max_bulk_group_size):
        try:
            await _submit_native_bulk_group_from_blocks(
                registry=registry,
                jobs=chunk,
                command_block=command_block,
                execution_profile_block=execution_profile_block,
                hpc_profile_block=hpc_profile_block,
                fugaku_no_check_directory=fugaku_no_check_directory,
            )
        except SpecHashMismatchError:
            raise
        except Exception as exc:
            classified = _classify_submit_exception(exc)
            if isinstance(classified, QueueFullError | TemporarySubmitError):
                for job in chunk:
                    _mark_deferred_if_needed(
                        registry=registry,
                        job_key=job.job_key,
                        error=str(classified),
                    )
                return True

            for job in chunk:
                _mark_failed_if_needed(
                    registry=registry,
                    job_key=job.job_key,
                    error=str(classified),
                )
            if stop_on_first_failure:
                return False

    return False


async def submit_job_from_blocks(
    *,
    work_dir: Path,
    job_key: str,
    command_block: str | None = None,
    execution_profile_block: str | None = None,
    hpc_profile_block: str | None = None,
    command_args: dict[str, Any] | None = None,
    input_digest: str | None = None,
    code_digest: str | None = None,
    environment_digest: str | None = None,
    registry: BulkJobRegistry | None = None,
    command_block_name: str | None = None,
    execution_profile_block_name: str | None = None,
    hpc_profile_block_name: str | None = None,
    fugaku_no_check_directory: bool = False,
    slurm_user: str | None = None,
    slurm_recovery_grace_seconds: float = DEFAULT_SLURM_RECOVERY_GRACE_SECONDS,
    slurm_clock_skew_margin_seconds: float = DEFAULT_SLURM_CLOCK_SKEW_MARGIN_SECONDS,
    scheduler_command_timeout_seconds: float | None = (DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS),
    cloud_log_policy: CloudLogPolicy | None = None,
) -> SubmittedJob:
    """Submit one block-defined HPC job without waiting for completion.

    Queue-full and retryable scheduler failures are recorded as
    ``SUBMIT_DEFERRED`` when a registry is provided, then raised so a future
    refill loop can stop submitting more jobs in the current cycle. Set
    ``fugaku_no_check_directory`` to opt into ``pjsub --no-check-directory`` for
    Fugaku submissions only.

    For Slurm with a registry, this function stores a durable ``PREPARED``
    compare-and-set claim before writing the script or invoking ``sbatch``.
    Ambiguous submission outcomes remain ``PREPARED`` and are reconciled by
    deterministic scheduler identity. They are never changed into an automatic
    resubmit candidate.

    An explicit non-legacy ``cloud_log_policy`` is also applied to submission
    and recovery outcomes. The omitted policy preserves the old silent bulk
    submission behavior.
    """

    _validate_slurm_recovery_settings(
        recovery_grace_seconds=slurm_recovery_grace_seconds,
        clock_skew_margin_seconds=slurm_clock_skew_margin_seconds,
        scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
    )

    resolved_command_block = _resolve_named_argument(
        preferred=command_block,
        alias=command_block_name,
        label="command_block",
    )
    resolved_execution_profile_block = _resolve_named_argument(
        preferred=execution_profile_block,
        alias=execution_profile_block_name,
        label="execution_profile_block",
    )
    resolved_hpc_profile_block = _resolve_named_argument(
        preferred=hpc_profile_block,
        alias=hpc_profile_block_name,
        label="hpc_profile_block",
    )

    prepared = await _prepare_job_from_blocks(
        command_block_name=resolved_command_block,
        execution_profile_block_name=resolved_execution_profile_block,
        hpc_profile_block_name=resolved_hpc_profile_block,
        work_dir=work_dir,
        script_filename=job_key,
        user_args=_command_args_to_user_args(command_args),
    )
    spec_hash = _resolved_bulk_spec_hash(
        prepared,
        input_digest=input_digest,
        code_digest=code_digest,
        environment_digest=environment_digest,
    )
    job_name: str | None = None
    job_comment: str | None = None
    if prepared.submission_target.hpc_target == "slurm":
        prepared, job_name, job_comment = _with_slurm_identity(
            prepared,
            job_key=job_key,
            spec_hash=spec_hash,
        )

    async def publish_submitted(result: SubmittedJob) -> SubmittedJob:
        record = registry.get_job(job_key) if registry is not None else None
        await _publish_bulk_cloud_result(
            cloud_log_policy=cloud_log_policy,
            hpc_target=prepared.submission_target.hpc_target,
            scheduler_job_id=result.scheduler_job_id,
            status=result.status,
            row=None,
            record=record,
        )
        return result

    async def publish_recovery_state() -> None:
        if registry is None:
            return
        record = registry.get_job(job_key)
        if record is None:
            return
        await _publish_bulk_cloud_result(
            cloud_log_policy=cloud_log_policy,
            hpc_target=prepared.submission_target.hpc_target,
            scheduler_job_id=str(record.effective_scheduler_job_id or "unknown"),
            status=record.status,
            row=None,
            record=record,
        )

    if registry is not None:
        existing_record = registry.get_job(job_key)
        registry.upsert_jobs(
            [
                BulkJobSpec(
                    job_key=job_key,
                    work_dir=Path(work_dir),
                    command_args=dict(command_args or {}),
                    wave_id=existing_record.wave_id if existing_record is not None else None,
                    target_id=existing_record.target_id if existing_record is not None else None,
                    stage_id=existing_record.stage_id if existing_record is not None else None,
                    priority=existing_record.priority if existing_record is not None else 0,
                    expected_outputs=(
                        existing_record.expected_outputs if existing_record is not None else []
                    ),
                    max_submit_attempts=(
                        existing_record.max_submit_attempts if existing_record is not None else 5
                    ),
                    execution_profile_block=resolved_execution_profile_block,
                    hpc_profile_block=resolved_hpc_profile_block,
                    spec_hash=spec_hash,
                    input_digest=input_digest,
                    code_digest=code_digest,
                    environment_digest=environment_digest,
                    job_name=job_name,
                    job_comment=job_comment,
                )
            ]
        )
        if prepared.submission_target.hpc_target == "slurm":
            if job_name is None or job_comment is None:
                raise RuntimeError("Slurm identity was not generated before registry claim.")
            resolved_slurm_user = str(slurm_user or getpass.getuser()).strip()
            if not resolved_slurm_user:
                raise ValueError("slurm_user must be non-empty for resumable submission.")

            record = registry.get_job(job_key)
            if record is None:
                raise SubmitError(f"Bulk job {job_key!r} was not persisted before submission.")
            if record.status == BulkJobStatus.AWAITING_OPERATOR:
                raise OperatorActionRequired(
                    job_keys=[job_key],
                    reason=record.last_error or "the registry row is in durable operator hold",
                )
            if record.scheduler_job_id:
                return await publish_submitted(
                    SubmittedJob(
                        job_key=job_key,
                        scheduler_job_id=record.scheduler_job_id,
                        status=record.status,
                        work_dir=record.work_dir,
                    )
                )
            if record.status == BulkJobStatus.UNKNOWN:
                _raise_operator_hold(
                    registry=registry,
                    job_key=job_key,
                    reason="UNKNOWN Slurm row has no scheduler job id",
                )
            if record.status == BulkJobStatus.PREPARED:
                try:
                    recovered = await _reconcile_prepared_slurm_job(
                        registry=registry,
                        record=record,
                        prepared=prepared,
                        slurm_user=resolved_slurm_user,
                        recovery_grace_seconds=slurm_recovery_grace_seconds,
                        clock_skew_margin_seconds=slurm_clock_skew_margin_seconds,
                        scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
                    )
                except (
                    OperatorActionRequired,
                    RecoveryPending,
                    SchedulerIdentityMismatchError,
                    SubmitOutcomeUnknownError,
                ):
                    await publish_recovery_state()
                    raise
                return await publish_submitted(recovered)

            _ensure_registry_can_submit(registry=registry, job_key=job_key)
            claimed = registry.claim_prepared(
                job_key=job_key,
                spec_hash=spec_hash,
                job_name=job_name,
                job_comment=job_comment,
            )
            claimed_record = registry.get_job(job_key)
            if claimed_record is None:
                raise SubmitError(f"Bulk job {job_key!r} disappeared during PREPARED claim.")
            if not claimed:
                if claimed_record.status == BulkJobStatus.AWAITING_OPERATOR:
                    raise OperatorActionRequired(
                        job_keys=[job_key],
                        reason=(
                            claimed_record.last_error
                            or "the registry row entered durable operator hold"
                        ),
                    )
                if claimed_record.scheduler_job_id:
                    return await publish_submitted(
                        SubmittedJob(
                            job_key=job_key,
                            scheduler_job_id=claimed_record.scheduler_job_id,
                            status=claimed_record.status,
                            work_dir=claimed_record.work_dir,
                        )
                    )
                if claimed_record.status == BulkJobStatus.PREPARED:
                    try:
                        recovered = await _reconcile_prepared_slurm_job(
                            registry=registry,
                            record=claimed_record,
                            prepared=prepared,
                            slurm_user=resolved_slurm_user,
                            recovery_grace_seconds=slurm_recovery_grace_seconds,
                            clock_skew_margin_seconds=slurm_clock_skew_margin_seconds,
                            scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
                        )
                    except (
                        OperatorActionRequired,
                        RecoveryPending,
                        SchedulerIdentityMismatchError,
                        SubmitOutcomeUnknownError,
                    ):
                        await publish_recovery_state()
                        raise
                    return await publish_submitted(recovered)
                _ensure_registry_can_submit(registry=registry, job_key=job_key)
                raise DuplicateJobKeyError(
                    f"Bulk job key {job_key!r} lost the PREPARED claim race."
                )

            try:
                submitted = await _submit_claimed_slurm_job(
                    registry=registry,
                    record=claimed_record,
                    prepared=prepared,
                    slurm_user=resolved_slurm_user,
                    recovery_grace_seconds=slurm_recovery_grace_seconds,
                    clock_skew_margin_seconds=slurm_clock_skew_margin_seconds,
                    scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
                )
            except (
                OperatorActionRequired,
                RecoveryPending,
                SchedulerIdentityMismatchError,
                SubmitOutcomeUnknownError,
            ):
                await publish_recovery_state()
                raise
            return await publish_submitted(submitted)

        _ensure_registry_can_submit(registry=registry, job_key=job_key)

    try:
        scheduler_job_id = await _submit_prepared_job(
            prepared,
            fugaku_no_check_directory=fugaku_no_check_directory,
            slurm_submit_timeout_seconds=scheduler_command_timeout_seconds,
        )
    except Exception as exc:
        classified = _classify_submit_exception(exc)
        if registry is not None:
            if isinstance(classified, QueueFullError | TemporarySubmitError):
                registry.mark_submit_deferred(job_key, error=str(classified))
            else:
                registry.mark_failed(job_key, error=str(classified))
        raise classified from exc

    if registry is not None:
        registry.mark_submitted(job_key, scheduler_job_id)

    return await publish_submitted(
        SubmittedJob(
            job_key=job_key,
            scheduler_job_id=scheduler_job_id,
            status=BulkJobStatus.SUBMITTED,
            work_dir=prepared.work_dir,
        )
    )


def _parse_fugaku_pjstat_rows(stdout: str) -> dict[str, dict[str, Any]]:
    return fugaku_runtime.parse_pjstat_rows(stdout)


_FUGAKU_SUBJOB_ID_RE = re.compile(r"^(\d+)\[(\d+)\]$")
_FUGAKU_SUBJOB_RANGE_RE = re.compile(r"^(\d+)\[(\d+)-(\d+)\]$")
_PARENT_FALLBACK_JOB_ID_KEY = "_qcsc_parent_fallback_job_id"


def _parse_fugaku_subjob_id(job_id: str) -> tuple[str, int] | None:
    match = _FUGAKU_SUBJOB_ID_RE.match(str(job_id).strip())
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _parse_fugaku_subjob_range(job_id: str) -> tuple[str, int, int] | None:
    match = _FUGAKU_SUBJOB_RANGE_RE.match(str(job_id).strip())
    if match is None:
        return None
    start = int(match.group(2))
    end = int(match.group(3))
    if end < start:
        return None
    return match.group(1), start, end


def _format_fugaku_subjob_ranges(parent_job_id: str, indices: list[int]) -> list[str]:
    if not indices:
        return []

    formatted: list[str] = []
    sorted_indices = sorted(set(indices))
    range_start = sorted_indices[0]
    previous = range_start
    for index in sorted_indices[1:]:
        if index == previous + 1:
            previous = index
            continue

        formatted.append(
            f"{parent_job_id}[{range_start}]"
            if range_start == previous
            else f"{parent_job_id}[{range_start}-{previous}]"
        )
        range_start = previous = index

    formatted.append(
        f"{parent_job_id}[{range_start}]"
        if range_start == previous
        else f"{parent_job_id}[{range_start}-{previous}]"
    )
    return formatted


def _fugaku_pjstat_query_ids(scheduler_job_ids: list[str]) -> list[str]:
    parent_indices: dict[str, list[int]] = {}
    passthrough: list[str] = []

    for scheduler_job_id in dict.fromkeys(str(job_id).strip() for job_id in scheduler_job_ids):
        if not scheduler_job_id:
            continue
        parsed_subjob = _parse_fugaku_subjob_id(scheduler_job_id)
        if parsed_subjob is not None:
            parent_job_id, bulk_index = parsed_subjob
            parent_indices.setdefault(parent_job_id, []).append(bulk_index)
            continue

        parsed_range = _parse_fugaku_subjob_range(scheduler_job_id)
        if parsed_range is not None:
            parent_job_id, start, end = parsed_range
            passthrough.append(f"{parent_job_id}[{start}-{end}]")
            continue

        passthrough.append(scheduler_job_id)

    query_ids = list(passthrough)
    for parent_job_id, indices in parent_indices.items():
        query_ids.extend(_format_fugaku_subjob_ranges(parent_job_id, indices))
    return query_ids


def _select_fugaku_rows_for_requested_ids(
    *,
    requested_ids: list[str],
    rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for scheduler_job_id in requested_ids:
        if scheduler_job_id in rows:
            selected[scheduler_job_id] = rows[scheduler_job_id]
            continue

        parsed_subjob = _parse_fugaku_subjob_id(scheduler_job_id)
        if parsed_subjob is None:
            continue

        parent_job_id, _bulk_index = parsed_subjob
        parent_row = rows.get(parent_job_id)
        if parent_row is None:
            continue

        fallback_row = dict(parent_row)
        fallback_row["JOB_ID"] = scheduler_job_id
        fallback_row[_PARENT_FALLBACK_JOB_ID_KEY] = parent_job_id
        selected[scheduler_job_id] = fallback_row

    return selected


async def _query_fugaku_history_statuses(
    query_ids: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    suffix = tuple(query_ids or [])
    for args in [
        ("pjstat", "-v", "-H"),
        ("pjstat", "-H", "-v"),
        ("pjstat", "-H"),
    ]:
        try:
            stdout = await fugaku_runtime.run_command(*args, *suffix)
        except Exception:
            continue
        return _parse_fugaku_pjstat_rows(stdout)
    return {}


def _parse_slurm_sacct_rows(stdout: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 6:
            continue
        job_id = fields[0].strip()
        if not job_id or "." in job_id:
            continue
        rows[job_id] = {
            "JobID": job_id,
            "State": fields[1].strip(),
            "ExitCode": fields[2].strip(),
            "Elapsed": fields[3].strip(),
            "AllocCPUS": fields[4].strip(),
            "NodeList": fields[5].strip(),
        }
    return rows


def _parse_miyabi_qstat_rows(stdout: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    current_job_id: str | None = None
    current_key = ""
    current_row: dict[str, Any] = {}

    def save_current() -> None:
        if current_job_id:
            rows[current_job_id] = dict(current_row)

    for line in stdout.splitlines():
        if line.startswith("Job Id:"):
            save_current()
            current_job_id = line.split(":", 1)[1].strip()
            current_row = {"Job_Id": current_job_id}
            current_key = ""
            continue

        if current_job_id is None or not line.strip():
            continue

        if line.startswith("\t") and current_key:
            current_row[current_key] = str(current_row[current_key]) + line.strip()
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        current_key = key.strip()
        current_row[current_key] = value.strip()

    save_current()
    return rows


async def _query_scheduler_statuses(
    *,
    hpc_target: str,
    scheduler_job_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not scheduler_job_ids:
        return {}

    requested = set(scheduler_job_ids)
    if hpc_target == "fugaku":
        query_ids = _fugaku_pjstat_query_ids(scheduler_job_ids)
        active_stdout = await fugaku_runtime.run_command("pjstat", "-v", *query_ids)
        rows = _parse_fugaku_pjstat_rows(active_stdout)
        missing = sorted(requested - set(rows))
        if missing:
            rows.update(await _query_fugaku_history_statuses(_fugaku_pjstat_query_ids(missing)))
        return _select_fugaku_rows_for_requested_ids(
            requested_ids=scheduler_job_ids,
            rows=rows,
        )

    if hpc_target == "slurm":
        stdout = await slurm_runtime.run_command(
            "sacct",
            "-j",
            ",".join(scheduler_job_ids),
            "--format=JobID,State,ExitCode,Elapsed,AllocCPUS,NodeList",
            "--parsable2",
            "--noheader",
        )
        return {
            job_id: row
            for job_id, row in _parse_slurm_sacct_rows(stdout).items()
            if job_id in requested
        }

    if hpc_target == "miyabi":
        stdout = await miyabi_runtime.run_command("qstat", "-f", *scheduler_job_ids)
        rows = _parse_miyabi_qstat_rows(stdout)
        return {job_id: row for job_id, row in rows.items() if job_id in requested}

    raise NotImplementedError(f"Unsupported hpc_target for monitor: {hpc_target}")


def _bulk_status_from_scheduler_row(hpc_target: str, row: dict[str, Any]) -> BulkJobStatus:
    if hpc_target == "fugaku":
        state = str(row.get("ST", "")).strip().upper()
        if state == "ACC":
            return BulkJobStatus.SUBMITTED
        if state in {"QUE", "Q", "HLD"}:
            return BulkJobStatus.QUEUED
        if state in {"RUN", "R", "RNA", "RNE", "RNO", "RNP", "RSM", "SPD", "SPP"}:
            return BulkJobStatus.RUNNING
        if state == "EXT":
            return BulkJobStatus.UNKNOWN
        if state == "CCL":
            return BulkJobStatus.CANCELLED
        if state in {"ERR", "RJT"}:
            return BulkJobStatus.FAILED
        return BulkJobStatus.UNKNOWN

    if hpc_target == "slurm":
        state = str(row.get("State", "")).strip().upper().split()[0].rstrip("+")
        if state in {"PENDING", "CONFIGURING"}:
            return BulkJobStatus.QUEUED
        if state in {"RUNNING", "COMPLETING", "SUSPENDED"}:
            return BulkJobStatus.RUNNING
        if state == "COMPLETED":
            return BulkJobStatus.SUCCEEDED
        if state == "CANCELLED":
            return BulkJobStatus.CANCELLED
        if state in {
            "BOOT_FAIL",
            "DEADLINE",
            "FAILED",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "TIMEOUT",
        }:
            return BulkJobStatus.FAILED
        return BulkJobStatus.UNKNOWN

    if hpc_target == "miyabi":
        state = str(row.get("job_state", row.get("state", ""))).strip().upper()
        exit_status = str(row.get("Exit_status", "")).strip()
        if state in {"Q", "H", "W", "T"}:
            return BulkJobStatus.QUEUED
        if state in {"R", "E"}:
            return BulkJobStatus.RUNNING
        if state in {"C", "F"} or exit_status:
            return BulkJobStatus.SUCCEEDED if exit_status in {"", "0"} else BulkJobStatus.FAILED
        return BulkJobStatus.UNKNOWN

    return BulkJobStatus.UNKNOWN


def _record_has_success_evidence(record) -> bool:
    if not record.expected_outputs:
        return False
    paths = [
        path if path.is_absolute() else record.work_dir / path for path in record.expected_outputs
    ]
    return all(path.exists() for path in paths)


def _monitor_status_from_scheduler_row(
    *,
    hpc_target: str,
    row: dict[str, Any],
    record: Any | None,
) -> tuple[BulkJobStatus, str | None]:
    status = _bulk_status_from_scheduler_row(hpc_target, row)
    if hpc_target == "slurm" and status == BulkJobStatus.SUCCEEDED and record is not None:
        if _record_has_success_evidence(record):
            return BulkJobStatus.SUCCEEDED, None
        return (
            BulkJobStatus.AWAITING_OPERATOR,
            "Slurm reported COMPLETED but expected output evidence is missing",
        )
    if hpc_target != "fugaku":
        error = None if status != BulkJobStatus.UNKNOWN else "unknown scheduler state"
        return status, error

    state = str(row.get("ST", "")).strip().upper()
    parent_fallback_job_id = row.get(_PARENT_FALLBACK_JOB_ID_KEY)
    if parent_fallback_job_id is not None:
        if record is not None and _record_has_success_evidence(record):
            return BulkJobStatus.SUCCEEDED, None
        return (
            BulkJobStatus.UNKNOWN,
            f"subjob row was not found; parent job {parent_fallback_job_id} is weak evidence only",
        )

    if state == "EXT":
        if record is not None and record.expected_outputs:
            if _record_has_success_evidence(record):
                return BulkJobStatus.SUCCEEDED, None
            return (
                BulkJobStatus.FAILED,
                "PJM reported EXT but expected outputs are missing",
            )
        return (
            BulkJobStatus.UNKNOWN,
            "PJM reported EXT without expected_outputs evidence",
        )

    error = None if status != BulkJobStatus.UNKNOWN else "unknown scheduler state"
    return status, error


def _records_by_scheduler_id(
    registry: BulkJobRegistry | None,
) -> dict[str, Any]:
    if registry is None:
        return {}
    records_by_scheduler_id: dict[str, Any] = {}
    for record in registry.get_all_jobs():
        scheduler_id = record.effective_scheduler_job_id
        if scheduler_id:
            records_by_scheduler_id[str(scheduler_id)] = record
    return records_by_scheduler_id


def _local_scheduler_path(value: object | None) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip().strip('"').strip("'")
    if not raw:
        return None
    if ":" in raw:
        _, raw = raw.split(":", 1)
    return Path(raw) if raw else None


def _one_log_candidate(work_dir: Path, suffix: str) -> Path | None:
    candidates = sorted(path for path in work_dir.glob(f"*{suffix}") if path.is_file())
    return candidates[0] if len(candidates) == 1 else None


def _bulk_log_paths(
    *,
    hpc_target: str,
    row: dict[str, Any] | None,
    record: BulkJobRecord | None,
) -> tuple[Path | None, Path | None]:
    row = row or {}
    if hpc_target == "miyabi":
        return (
            _local_scheduler_path(row.get("Output_Path")),
            _local_scheduler_path(row.get("Error_Path")),
        )
    if record is None:
        return None, None
    if hpc_target == "slurm":
        return record.work_dir / "output.out", record.work_dir / "output.err"
    if hpc_target == "fugaku":
        stdout_path = _local_scheduler_path(
            row.get("Output_Path") or row.get("OUT") or row.get("STDOUT")
        )
        stderr_path = _local_scheduler_path(
            row.get("Error_Path") or row.get("ERR") or row.get("STDERR")
        )
        return (
            stdout_path or _one_log_candidate(record.work_dir, ".out"),
            stderr_path or _one_log_candidate(record.work_dir, ".err"),
        )
    return None, None


def _bulk_summary(
    *,
    scheduler_job_id: str,
    hpc_target: str,
    status: BulkJobStatus,
    row: dict[str, Any] | None,
    stdout_path: Path | None,
    stderr_path: Path | None,
) -> CloudJobSummary:
    row = row or {}
    if hpc_target == "slurm":
        exit_code = str(row.get("ExitCode", "")).partition(":")[0] or None
        elapsed = row.get("Elapsed")
        node = row.get("NodeList")
    elif hpc_target == "miyabi":
        exit_code = row.get("Exit_status")
        elapsed = row.get("resources_used.walltime")
        node = row.get("exec_host") or row.get("exec_vnode")
    else:
        exit_code = row.get("EC")
        elapsed = row.get("ELAPSE") or row.get("ELAPSED")
        node = row.get("NODE") or row.get("HOST")
    return CloudJobSummary(
        job_id=scheduler_job_id,
        state=status.value,
        exit_code=exit_code,
        elapsed=elapsed,
        node=node,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _bulk_artifact_key(record: BulkJobRecord | None, scheduler_job_id: str) -> str:
    source = record.job_key if record is not None else scheduler_job_id
    safe = re.sub(r"[^a-z0-9-]+", "-", source.lower()).strip("-")
    return f"hpc-job-summary-{safe or 'job'}"[:200].rstrip("-")


async def _publish_bulk_cloud_result(
    *,
    cloud_log_policy: CloudLogPolicy | None,
    hpc_target: str,
    scheduler_job_id: str,
    status: BulkJobStatus,
    row: dict[str, Any] | None,
    record: BulkJobRecord | None,
) -> None:
    policy = resolve_cloud_log_policy(cloud_log_policy)
    # Bulk historically emitted neither scheduler output nor artifacts.
    if policy.mode == "legacy":
        return

    stdout_path, stderr_path = _bulk_log_paths(
        hpc_target=hpc_target,
        row=row,
        record=record,
    )
    summary = _bulk_summary(
        scheduler_job_id=scheduler_job_id,
        hpc_target=hpc_target,
        status=status,
        row=row,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if policy.mode != "none":
        emit_cloud_job_logs(
            logger=get_run_logger(),
            policy=policy,
            summary=summary,
            stdout=read_log_text(stdout_path),
            stderr=read_log_text(stderr_path),
        )
    if policy.should_create_artifact(legacy_default=False):
        artifact = {
            "job_id": summary.job_id,
            "state": summary.state,
            "exit_code": summary.exit_code,
            "elapsed": summary.elapsed,
            "node": summary.node,
            "stdout_path": str(stdout_path) if stdout_path is not None else None,
            "stderr_path": str(stderr_path) if stderr_path is not None else None,
        }
        await create_table_artifact(
            table=[list(artifact.keys()), list(artifact.values())],
            key=_bulk_artifact_key(record, scheduler_job_id),
        )


def _update_registry_for_monitor_status(
    *,
    registry: BulkJobRegistry,
    job_key: str,
    status: BulkJobStatus,
    error: str | None = None,
) -> None:
    if status == BulkJobStatus.QUEUED:
        registry.record_monitor_attempt(job_key)
        registry.mark_queued(job_key)
    elif status == BulkJobStatus.RUNNING:
        registry.record_monitor_attempt(job_key)
        registry.mark_running(job_key)
    elif status == BulkJobStatus.SUCCEEDED:
        registry.record_monitor_attempt(job_key)
        registry.mark_succeeded(job_key)
    elif status == BulkJobStatus.FAILED:
        registry.record_monitor_attempt(job_key)
        registry.mark_failed(job_key, error=error)
    elif status == BulkJobStatus.CANCELLED:
        registry.record_monitor_attempt(job_key)
        registry.mark_cancelled(job_key, error=error)
    elif status == BulkJobStatus.UNKNOWN:
        registry.mark_unknown(job_key, error=error)
    elif status == BulkJobStatus.AWAITING_OPERATOR:
        registry.mark_awaiting_operator(
            job_key,
            error or "scheduler state requires operator reconciliation",
        )
    else:
        registry.record_monitor_attempt(job_key)


async def monitor_jobs_many(
    *,
    scheduler_job_ids: list[str],
    hpc_profile_block: str | None = None,
    registry: BulkJobRegistry | None = None,
    hpc_profile_block_name: str | None = None,
    cloud_log_policy: CloudLogPolicy | None = None,
) -> dict[str, BulkJobStatus]:
    """Monitor many jobs with one query and optional bounded Cloud summaries."""

    resolved_hpc_profile_block = _resolve_named_argument(
        preferred=hpc_profile_block,
        alias=hpc_profile_block_name,
        label="hpc_profile_block",
    )
    hpc_target = await resolve_hpc_target(hpc_profile_block_name=resolved_hpc_profile_block)

    requested_ids = list(dict.fromkeys(scheduler_job_ids))
    if not requested_ids:
        return {}

    query_error: str | None = None
    try:
        scheduler_rows = await _query_scheduler_statuses(
            hpc_target=hpc_target,
            scheduler_job_ids=requested_ids,
        )
    except Exception as exc:
        scheduler_rows = {}
        query_error = _exception_text(exc)

    records_by_scheduler_id = _records_by_scheduler_id(registry)
    results: dict[str, BulkJobStatus] = {}

    for scheduler_job_id in requested_ids:
        row = scheduler_rows.get(scheduler_job_id)
        record = records_by_scheduler_id.get(scheduler_job_id)
        if record is not None and record.status == BulkJobStatus.SUCCEEDED:
            status = BulkJobStatus.SUCCEEDED
            error = None
        elif row is not None:
            status, error = _monitor_status_from_scheduler_row(
                hpc_target=hpc_target,
                row=row,
                record=record,
            )
        elif record is not None and _record_has_success_evidence(record):
            status = BulkJobStatus.SUCCEEDED
            error = None
        else:
            status = BulkJobStatus.UNKNOWN
            error = query_error or "job was not found in scheduler output"

        results[scheduler_job_id] = status
        previous_status = record.status if record is not None else None
        if registry is not None and record is not None:
            _update_registry_for_monitor_status(
                registry=registry,
                job_key=record.job_key,
                status=status,
                error=error,
            )
        if record is None or previous_status != status:
            await _publish_bulk_cloud_result(
                cloud_log_policy=cloud_log_policy,
                hpc_target=hpc_target,
                scheduler_job_id=scheduler_job_id,
                status=status,
                row=row,
                record=record,
            )

    return results


async def execute_cancel_request(
    *,
    registry: BulkJobRegistry,
    job_key: str,
    hpc_profile_block: str,
    scheduler_command_timeout_seconds: float | None = (DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS),
) -> BulkCancelOutcome | None:
    """Execute one durable Slurm cancellation intent at most once.

    No scheduler command is issued unless ``request_cancel()`` has persisted
    ``CANCEL_REQUESTED`` and the registry contains a scheduler job ID. The
    dispatch claim is durable, so a cancelled/crashed executor leaves
    ``DISPATCHING`` for operator reconciliation instead of calling ``scancel``
    again automatically.
    """

    record = registry.get_job(job_key)
    if record is None:
        raise KeyError(f"Unknown bulk job key {job_key!r}.")
    if record.desired_state != BulkJobDesiredState.CANCEL_REQUESTED:
        return None
    if record.cancel_outcome is not None:
        return record.cancel_outcome

    if record.status.is_terminal:
        registry.record_terminal_cancel_outcome(job_key)
        updated = registry.get_job(job_key)
        return updated.cancel_outcome if updated is not None else None

    scheduler_job_id = record.effective_scheduler_job_id
    if scheduler_job_id is None:
        if registry.cancel_without_scheduler_submission(job_key):
            return BulkCancelOutcome.NOT_SUBMITTED
        # PREPARED without an ID may already have caused an external submit.
        # Reconciliation must establish the ID before cancellation can run.
        return None

    if scheduler_command_timeout_seconds is not None and scheduler_command_timeout_seconds <= 0:
        raise ValueError("scheduler_command_timeout_seconds must be greater than 0.")

    hpc_target = await resolve_hpc_target(hpc_profile_block_name=hpc_profile_block)
    if hpc_target != "slurm":
        raise NotImplementedError(
            "Durable cancel execution is implemented only for hpc_target='slurm'."
        )

    try:
        scheduler_rows = await _query_scheduler_statuses(
            hpc_target="slurm",
            scheduler_job_ids=[str(scheduler_job_id)],
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        scheduler_rows = {}

    scheduler_row = scheduler_rows.get(str(scheduler_job_id))
    if scheduler_row is not None:
        scheduler_status = _bulk_status_from_scheduler_row("slurm", scheduler_row)
        if scheduler_status.is_terminal:
            recorded_terminal = registry.record_terminal_cancel_outcome(job_key)
            status, error = _monitor_status_from_scheduler_row(
                hpc_target="slurm",
                row=scheduler_row,
                record=record,
            )
            _update_registry_for_monitor_status(
                registry=registry,
                job_key=job_key,
                status=status,
                error=error,
            )
            if recorded_terminal:
                return BulkCancelOutcome.ALREADY_TERMINAL
            updated = registry.get_job(job_key)
            return updated.cancel_outcome if updated is not None else None

    if not registry.claim_cancel_dispatch(job_key):
        updated = registry.get_job(job_key)
        return updated.cancel_outcome if updated is not None else None

    runtime = SlurmRuntime()
    try:
        await runtime.cancel(
            str(scheduler_job_id),
            intent_confirmed=True,
            timeout_seconds=scheduler_command_timeout_seconds,
        )
    except asyncio.CancelledError:
        # The remote outcome is ambiguous. Keep DISPATCHING so no automatic
        # retry can issue another scheduler side effect.
        raise
    except SlurmCancelNotFoundError as exc:
        outcome = BulkCancelOutcome.NOT_FOUND
        registry.record_cancel_outcome(job_key, outcome, error=str(exc))
        registry.mark_awaiting_operator(job_key, str(exc))
        return outcome
    except SlurmTemporaryCancelError as exc:
        outcome = BulkCancelOutcome.TEMPORARY_FAILURE
        registry.record_cancel_outcome(job_key, outcome, error=str(exc))
        registry.mark_awaiting_operator(job_key, str(exc))
        return outcome
    except (SlurmCancelRejectedError, SlurmCancelError) as exc:
        outcome = BulkCancelOutcome.REJECTED
        registry.record_cancel_outcome(job_key, outcome, error=str(exc))
        registry.mark_awaiting_operator(job_key, str(exc))
        return outcome

    if not registry.record_cancel_outcome(job_key, BulkCancelOutcome.REQUEST_ACCEPTED):
        raise RuntimeError(
            f"scancel accepted job {scheduler_job_id}, but its durable outcome "
            f"could not be recorded for {job_key!r}."
        )
    return BulkCancelOutcome.REQUEST_ACCEPTED


async def execute_cancel_requests(
    *,
    registry: BulkJobRegistry,
    hpc_profile_block: str,
    scheduler_command_timeout_seconds: float | None = (DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS),
) -> dict[str, BulkCancelOutcome | None]:
    """Process each unclaimed durable cancellation intent once."""

    results: dict[str, BulkCancelOutcome | None] = {}
    for record in registry.get_pending_cancel_requests():
        results[record.job_key] = await execute_cancel_request(
            registry=registry,
            job_key=record.job_key,
            hpc_profile_block=effective_hpc_profile_block(record, hpc_profile_block),
            scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
        )
    return results


async def _reconcile_bulk_prepared_jobs(
    *,
    registry: BulkJobRegistry,
    command_block: str,
    execution_profile_block: str,
    hpc_profile_block: str,
    slurm_user: str | None,
    slurm_recovery_grace_seconds: float,
    slurm_clock_skew_margin_seconds: float,
    scheduler_command_timeout_seconds: float | None,
    cloud_log_policy: CloudLogPolicy | None = None,
) -> bool:
    """Reconcile durable Slurm claims once and report whether a later retry is needed."""

    for job in registry.get_recovery_candidates():
        if job.status == BulkJobStatus.UNKNOWN:
            if job.scheduler_job_id is None:
                registry.mark_awaiting_operator(
                    job.job_key,
                    "UNKNOWN Slurm row has no scheduler job id",
                )
                return True
            continue
        if job.status != BulkJobStatus.PREPARED or job.scheduler_job_id is not None:
            continue
        caller_digests = {
            key: value
            for key, value in {
                "input_digest": job.input_digest,
                "code_digest": job.code_digest,
                "environment_digest": job.environment_digest,
            }.items()
            if value is not None
        }
        try:
            cloud_policy_kwargs = (
                {"cloud_log_policy": cloud_log_policy} if cloud_log_policy is not None else {}
            )
            await submit_job_from_blocks(
                command_block=command_block,
                execution_profile_block=effective_execution_profile_block(
                    job,
                    execution_profile_block,
                ),
                hpc_profile_block=effective_hpc_profile_block(
                    job,
                    hpc_profile_block,
                ),
                work_dir=job.work_dir,
                job_key=job.job_key,
                command_args=job.command_args,
                registry=registry,
                slurm_user=slurm_user,
                slurm_recovery_grace_seconds=slurm_recovery_grace_seconds,
                slurm_clock_skew_margin_seconds=slurm_clock_skew_margin_seconds,
                scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
                **caller_digests,
                **cloud_policy_kwargs,
            )
        except (
            OperatorActionRequired,
            RecoveryPending,
            SchedulerIdentityMismatchError,
            SubmitOutcomeUnknownError,
        ):
            return True
    return False


async def run_jobs_from_blocks_bulk(
    *,
    jobs: list[BulkJobSpec],
    command_block: str,
    execution_profile_block: str,
    hpc_profile_block: str,
    registry_path: Path,
    queue_probe: QueueProbe | None = None,
    max_active_jobs: int = 1000,
    safety_margin: int = 20,
    max_submit_per_refill: int = 100,
    submit_mode: Literal["single", "native_bulk"] = "single",
    initial_submit_count: int | None = None,
    max_bulk_group_size: int = 100,
    target_active_jobs: int | None = None,
    poll_interval_seconds: int = 60,
    refill_interval_seconds: int = 60,
    stop_on_first_failure: bool = False,
    fugaku_no_check_directory: bool = False,
    slurm_user: str | None = None,
    slurm_recovery_grace_seconds: float = DEFAULT_SLURM_RECOVERY_GRACE_SECONDS,
    slurm_clock_skew_margin_seconds: float = DEFAULT_SLURM_CLOCK_SKEW_MARGIN_SECONDS,
    scheduler_command_timeout_seconds: float | None = (DEFAULT_SCHEDULER_COMMAND_TIMEOUT_SECONDS),
    cloud_log_policy: CloudLogPolicy | None = None,
) -> BulkRunResult:
    """Run many block-defined HPC jobs through one queue-aware bulk loop.

    This API submits and monitors scheduler jobs from a shared pending pool. It
    does not create one Prefect task per scheduler job, and wave identifiers on
    ``BulkJobSpec`` remain registry metadata for downstream workflow readiness
    checks rather than submit units. The default ``submit_mode="single"`` keeps
    using one scheduler submit per logical job. Fugaku native bulk submission is
    an explicit opt-in path via ``submit_mode="native_bulk"``. Set
    ``fugaku_no_check_directory`` to opt into ``pjsub --no-check-directory`` for
    Fugaku submissions only.

    ``cloud_log_policy`` defaults to ``legacy``, which preserves the existing
    bulk behavior of emitting no per-job result logs or artifacts. Select
    ``summary`` explicitly to emit one bounded message per status transition.
    """

    if submit_mode not in {"single", "native_bulk"}:
        raise ValueError("submit_mode must be 'single' or 'native_bulk'.")
    _validate_slurm_recovery_settings(
        recovery_grace_seconds=slurm_recovery_grace_seconds,
        clock_skew_margin_seconds=slurm_clock_skew_margin_seconds,
        scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
    )
    cloud_policy_kwargs = (
        {"cloud_log_policy": cloud_log_policy} if cloud_log_policy is not None else {}
    )
    if submit_mode == "native_bulk":
        _validate_native_bulk_specs(jobs)

    registry = BulkJobRegistry(registry_path)
    jobs = await _resolve_registered_bulk_spec_hashes(
        jobs=jobs,
        registry=registry,
        command_block=command_block,
        execution_profile_block=execution_profile_block,
        hpc_profile_block=hpc_profile_block,
    )
    registry.upsert_jobs(jobs)
    registry.refresh_completed_jobs_from_outputs()
    total_jobs = len({job.job_key for job in jobs})

    await execute_cancel_requests(
        registry=registry,
        hpc_profile_block=hpc_profile_block,
        scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
    )

    if (
        registry.all_terminal()
        or _has_operator_holds(registry)
        or (stop_on_first_failure and _has_failed_jobs(registry))
    ):
        return _build_bulk_run_result(registry=registry, total_jobs=total_jobs)

    recovery_pending = await _reconcile_bulk_prepared_jobs(
        registry=registry,
        command_block=command_block,
        execution_profile_block=execution_profile_block,
        hpc_profile_block=hpc_profile_block,
        slurm_user=slurm_user,
        slurm_recovery_grace_seconds=slurm_recovery_grace_seconds,
        slurm_clock_skew_margin_seconds=slurm_clock_skew_margin_seconds,
        scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
        cloud_log_policy=cloud_log_policy,
    )
    registry.refresh_completed_jobs_from_outputs()
    await execute_cancel_requests(
        registry=registry,
        hpc_profile_block=hpc_profile_block,
        scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
    )
    if recovery_pending or _has_operator_holds(registry) or registry.all_terminal():
        return _build_bulk_run_result(registry=registry, total_jobs=total_jobs)

    if submit_mode == "native_bulk":
        submission_target = await resolve_submission_target(
            hpc_profile_block_name=hpc_profile_block,
            execution_profile_block_name=execution_profile_block,
        )
        if submission_target.hpc_target != "fugaku":
            raise ValueError("submit_mode='native_bulk' is only supported for Fugaku/PJM.")
        if max_bulk_group_size <= 0:
            raise ValueError("max_bulk_group_size must be positive.")

    resolved_queue_probe = queue_probe or await _resolve_default_bulk_queue_probe(
        hpc_profile_block=hpc_profile_block,
        execution_profile_block=execution_profile_block,
        max_active_jobs=max_active_jobs,
        safety_margin=safety_margin,
        submit_mode=submit_mode,
        slurm_user=slurm_user,
        scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
    )
    submit_gate = QueueAwareSubmitGate(
        queue_probe=resolved_queue_probe,
        max_active_jobs=target_active_jobs if target_active_jobs is not None else max_active_jobs,
        safety_margin=safety_margin,
        max_submit_per_refill=max_submit_per_refill,
    )

    loop = asyncio.get_running_loop()
    next_refill_at = 0.0

    while not registry.all_terminal():
        await execute_cancel_requests(
            registry=registry,
            hpc_profile_block=hpc_profile_block,
            scheduler_command_timeout_seconds=scheduler_command_timeout_seconds,
        )
        if _has_operator_holds(registry) or registry.all_terminal():
            return _build_bulk_run_result(registry=registry, total_jobs=total_jobs)

        monitorable_jobs = [
            job for job in registry.get_monitorable_jobs() if job.effective_scheduler_job_id
        ]
        if monitorable_jobs:
            grouped_scheduler_ids: dict[str, list[str]] = {}
            for job in monitorable_jobs:
                effective_hpc_block = effective_hpc_profile_block(job, hpc_profile_block)
                grouped_scheduler_ids.setdefault(effective_hpc_block, []).append(
                    str(job.effective_scheduler_job_id)
                )
            for effective_hpc_block, scheduler_job_ids in grouped_scheduler_ids.items():
                await monitor_jobs_many(
                    hpc_profile_block=effective_hpc_block,
                    scheduler_job_ids=scheduler_job_ids,
                    registry=registry,
                    **cloud_policy_kwargs,
                )

        registry.refresh_completed_jobs_from_outputs()

        if _has_operator_holds(registry):
            return _build_bulk_run_result(registry=registry, total_jobs=total_jobs)
        if stop_on_first_failure and _has_failed_jobs(registry):
            break

        now = loop.time()
        if now >= next_refill_at:
            submit_limit = _submit_limit_for_cycle(
                registry=registry,
                initial_submit_count=initial_submit_count,
                max_submit_per_refill=max_submit_per_refill,
            )
            stop_after_deferred_submit = False
            recovery_waiting = False
            if submit_mode == "native_bulk":
                stop_after_deferred_submit = await _submit_native_bulk_cycle_from_blocks(
                    registry=registry,
                    command_block=command_block,
                    execution_profile_block=execution_profile_block,
                    hpc_profile_block=hpc_profile_block,
                    queue_probe=resolved_queue_probe,
                    submit_limit=submit_limit,
                    max_bulk_group_size=max_bulk_group_size,
                    target_active_jobs=target_active_jobs,
                    stop_on_first_failure=stop_on_first_failure,
                    fugaku_no_check_directory=fugaku_no_check_directory,
                )
            else:
                pre_candidates = registry.get_submit_candidates(limit=submit_limit)
                if pre_candidates:
                    submit_gate.max_submit_per_refill = submit_limit
                    submit_count = min(
                        submit_gate.allowed_submit_count(),
                        len(pre_candidates),
                    )
                    for job in pre_candidates[:submit_count]:
                        try:
                            caller_digests = {
                                key: value
                                for key, value in {
                                    "input_digest": job.input_digest,
                                    "code_digest": job.code_digest,
                                    "environment_digest": job.environment_digest,
                                }.items()
                                if value is not None
                            }
                            await submit_job_from_blocks(
                                command_block=command_block,
                                execution_profile_block=effective_execution_profile_block(
                                    job,
                                    execution_profile_block,
                                ),
                                hpc_profile_block=effective_hpc_profile_block(
                                    job,
                                    hpc_profile_block,
                                ),
                                work_dir=job.work_dir,
                                job_key=job.job_key,
                                command_args=job.command_args,
                                registry=registry,
                                fugaku_no_check_directory=fugaku_no_check_directory,
                                slurm_user=slurm_user,
                                slurm_recovery_grace_seconds=slurm_recovery_grace_seconds,
                                slurm_clock_skew_margin_seconds=(slurm_clock_skew_margin_seconds),
                                scheduler_command_timeout_seconds=(
                                    scheduler_command_timeout_seconds
                                ),
                                **caller_digests,
                                **cloud_policy_kwargs,
                            )
                        except (
                            OperatorActionRequired,
                            RecoveryPending,
                            SchedulerIdentityMismatchError,
                            SubmitOutcomeUnknownError,
                        ):
                            recovery_waiting = True
                            break
                        except QueueFullError as exc:
                            _mark_deferred_if_needed(
                                registry=registry,
                                job_key=job.job_key,
                                error=str(exc),
                            )
                            break
                        except TemporarySubmitError as exc:
                            _mark_deferred_if_needed(
                                registry=registry,
                                job_key=job.job_key,
                                error=str(exc),
                            )
                            break
                        except SpecHashMismatchError:
                            raise
                        except CancellationRequestedError:
                            await execute_cancel_request(
                                registry=registry,
                                job_key=job.job_key,
                                hpc_profile_block=effective_hpc_profile_block(
                                    job,
                                    hpc_profile_block,
                                ),
                                scheduler_command_timeout_seconds=(
                                    scheduler_command_timeout_seconds
                                ),
                            )
                            continue
                        except Exception as exc:
                            _mark_failed_if_needed(
                                registry=registry,
                                job_key=job.job_key,
                                error=_exception_text(exc),
                            )
                            if stop_on_first_failure:
                                break

            if stop_after_deferred_submit:
                break
            if recovery_waiting or _has_operator_holds(registry):
                return _build_bulk_run_result(registry=registry, total_jobs=total_jobs)

            next_refill_at = now + max(0.0, float(refill_interval_seconds))

        if stop_on_first_failure and _has_failed_jobs(registry):
            break
        if registry.all_terminal():
            break

        if submit_mode == "native_bulk":
            active_jobs = registry.count_active_jobs()
            submit_candidates = registry.count_submit_candidates()
            if active_jobs == 0 and submit_candidates == 0:
                break

        sleep_seconds = max(0.0, float(poll_interval_seconds))
        if sleep_seconds == 0 and next_refill_at > loop.time():
            sleep_seconds = max(0.0, next_refill_at - loop.time())
        await asyncio.sleep(sleep_seconds)

    return _build_bulk_run_result(registry=registry, total_jobs=total_jobs)


async def run_job_from_blocks(
    *,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    work_dir: Path,
    script_filename: str | None = None,
    user_args: list[str] | None = None,
    watch_poll_interval: float = 10.0,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "hpc-job-metrics",
    cloud_log_policy: CloudLogPolicy | None = None,
    fugaku_job_name: str | None = None,
    execution_profile_overrides: dict[str, Any] | None = None,
) -> Any:
    """Resolve Prefect blocks and execute a job on the configured target.

    This is the main block-driven entrypoint for workflow authors. It loads the
    command, execution profile, and HPC profile blocks; converts them into the
    internal runtime models; and dispatches to local execution or the Miyabi,
    Fugaku, or Slurm executor.

    Args:
        command_block_name: Prefect block document name for the command to run.
        execution_profile_block_name: Prefect block document name describing
            resources, launcher, environment, and default execution behavior.
        hpc_profile_block_name: Prefect block document name describing the
            execution target and executable mapping, plus scheduler routing
            fields when applicable.
        work_dir: Working directory for the process or scheduler job.
        script_filename: Logical or scheduler-specific script filename. The
            suffix is normalized for scheduler targets. It is ignored for local
            execution and may be omitted.
        user_args: Optional extra command-line arguments appended after the
            command block's default arguments.
        watch_poll_interval: Seconds to wait between scheduler status polls.
        timeout_seconds: Optional maximum wait time for terminal job status.
        metrics_artifact_key: Prefect artifact key used for job metrics.
        cloud_log_policy: Prefect Cloud output and artifact policy. Omission
            preserves historical target-specific behavior.
        fugaku_job_name: Optional Fugaku PJM job name. When omitted, a safe name
            is derived from the command name.
        execution_profile_overrides: Optional runtime overrides for selected
            execution profile fields, such as ``num_nodes`` or ``walltime``.

    Returns:
        A target-specific result object: ``LocalRunResult``,
        ``MiyabiRunResult``, ``FugakuRunResult``, or ``SlurmRunResult``.

    Raises:
        ValueError: If the command and execution profile blocks refer to
            different command names, if a required project/group is missing,
            if local execution receives ``modules`` or ``pre_commands``, or if
            unsupported execution profile override keys are provided.
        KeyError: If the command's executable key is missing from the HPC
            profile's executable map.
        NotImplementedError: If the resolved ``hpc_target`` is unsupported.
    """
    prepared = await _prepare_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=work_dir,
        script_filename=script_filename,
        user_args=user_args,
        fugaku_job_name=fugaku_job_name,
        execution_profile_overrides=execution_profile_overrides,
    )

    if prepared.submission_target.hpc_target == "local":
        return await run_local_job(
            work_dir=prepared.work_dir,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
            cloud_log_policy=cloud_log_policy,
        )

    if prepared.submission_target.hpc_target == "miyabi":
        return await run_miyabi_job(
            work_dir=prepared.work_dir,
            script_filename=prepared.script_filename,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
            watch_poll_interval=watch_poll_interval,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
            cloud_log_policy=cloud_log_policy,
        )

    if prepared.submission_target.hpc_target == "fugaku":
        return await run_fugaku_job(
            work_dir=prepared.work_dir,
            script_filename=prepared.script_filename,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
            watch_poll_interval=watch_poll_interval,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
            cloud_log_policy=cloud_log_policy,
        )

    if prepared.submission_target.hpc_target == "slurm":
        return await run_slurm_job(
            work_dir=prepared.work_dir,
            script_filename=prepared.script_filename,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
            watch_poll_interval=watch_poll_interval,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
            cloud_log_policy=cloud_log_policy,
        )

    raise NotImplementedError(
        f"hpc_target='{prepared.submission_target.hpc_target}' is not supported yet by "
        "run_job_from_blocks."
    )
