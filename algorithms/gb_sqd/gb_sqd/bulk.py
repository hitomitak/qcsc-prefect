"""Bulk GB-SQD flows for recursively discovered input directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from prefect import flow, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

from .discovery import discover_target_directories
from .target_overrides import merge_target_job_parameters, prepare_target_overrides


def _batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _default_command_block_name(mode: str) -> str:
    return "cmd-gb-sqd-ext" if mode == "ext_sqd" else "cmd-gb-sqd-trim"


def _default_execution_profile_block_name(mode: str) -> str:
    return "exec-gb-sqd-ext-fugaku" if mode == "ext_sqd" else "exec-gb-sqd-trim-fugaku"


def _summary_path(output_root_dir: str | Path) -> Path:
    return Path(output_root_dir).expanduser().resolve() / "_bulk_summary" / "run_summary.json"


def _write_summary(output_root_dir: str | Path, summary: dict[str, Any]) -> None:
    path = _summary_path(output_root_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def _get_bulk_target_run_task():
    from .tasks import bulk_target_run_task

    return bulk_target_run_task


@flow(name="GB-SQD-Bulk", task_runner=ConcurrentTaskRunner())
def bulk_gb_sqd_flow(
    *,
    mode: str,
    input_root_dir: str,
    output_root_dir: str,
    count_dict_filename: str = "count_dict.txt",
    fcidump_filename: str = "fci_dump.txt",
    leaf_only: bool = True,
    skip_completed: bool = True,
    fail_fast: bool = False,
    max_jobs_in_queue: int = 10,
    queue_limit_scope: str = "user_queue",
    queue_poll_interval_seconds: float = 120.0,
    max_target_task_retries: int = 1,
    max_prefect_concurrency: int | None = None,
    job_name_prefix: str = "gbsqd-bulk",
    command_block_name: str | None = None,
    execution_profile_block_name: str | None = None,
    hpc_profile_block_name: str = "hpc-fugaku-gb-sqd",
    target_overrides: dict[str, dict[str, Any]] | None = None,
    num_recovery: int = 1,
    num_batches: int = 1,
    num_samples_per_batch: int = 1,
    num_samples_per_recovery: int = 100,
    iteration: int = 1,
    block: int = 10,
    tolerance: float = 1.0e-2,
    max_time: float = 3600.0,
    adet_comm_size: int = 1,
    bdet_comm_size: int = 1,
    task_comm_size: int = 1,
    adet_comm_size_combined: int | None = None,
    bdet_comm_size_combined: int | None = None,
    task_comm_size_combined: int | None = None,
    adet_comm_size_final: int | None = None,
    bdet_comm_size_final: int | None = None,
    task_comm_size_final: int | None = None,
    do_carryover_in_recovery: bool = False,
    carryover_ratio: float = 0.5,
    carryover_ratio_batch: float = 0.1,
    carryover_ratio_combined: float = 0.5,
    carryover_threshold: float = 1.0e-2,
    with_hf: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a bulk GB-SQD submission over recursively discovered target directories."""

    logger = get_run_logger()

    if mode not in {"ext_sqd", "trim_sqd"}:
        raise ValueError(f"Unsupported GB-SQD mode: {mode}")
    if max_jobs_in_queue < 1:
        raise ValueError("max_jobs_in_queue must be >= 1")
    if queue_poll_interval_seconds <= 0:
        raise ValueError("queue_poll_interval_seconds must be > 0")

    discovered = discover_target_directories(
        input_root_dir,
        count_dict_filename=count_dict_filename,
        fcidump_filename=fcidump_filename,
        leaf_only=leaf_only,
    )
    if not discovered:
        raise ValueError(f"No target directories found under {Path(input_root_dir).expanduser().resolve()}")

    concurrency = max_prefect_concurrency or max_jobs_in_queue
    if concurrency < 1:
        raise ValueError("max_prefect_concurrency must be >= 1")

    resolved_output_root = Path(output_root_dir).expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    command_name = command_block_name or _default_command_block_name(mode)
    execution_name = execution_profile_block_name or _default_execution_profile_block_name(mode)
    input_root_path = Path(input_root_dir).expanduser().resolve()

    shared_job_parameters = {
        "num_recovery": num_recovery,
        "num_batches": num_batches,
        "num_samples_per_batch": num_samples_per_batch,
        "num_samples_per_recovery": num_samples_per_recovery,
        "iteration": iteration,
        "block": block,
        "tolerance": tolerance,
        "max_time": max_time,
        "adet_comm_size": adet_comm_size,
        "bdet_comm_size": bdet_comm_size,
        "task_comm_size": task_comm_size,
        "adet_comm_size_combined": adet_comm_size_combined,
        "bdet_comm_size_combined": bdet_comm_size_combined,
        "task_comm_size_combined": task_comm_size_combined,
        "adet_comm_size_final": adet_comm_size_final,
        "bdet_comm_size_final": bdet_comm_size_final,
        "task_comm_size_final": task_comm_size_final,
        "do_carryover_in_recovery": do_carryover_in_recovery,
        "carryover_ratio": carryover_ratio,
        "carryover_ratio_batch": carryover_ratio_batch,
        "carryover_ratio_combined": carryover_ratio_combined,
        "carryover_threshold": carryover_threshold,
        "with_hf": with_hf,
        "verbose": verbose,
    }
    discovered_relative_paths = [target.relative_to(input_root_path).as_posix() for target in discovered]
    prepared_target_overrides = prepare_target_overrides(
        discovered_relative_paths=discovered_relative_paths,
        target_overrides=target_overrides,
        allowed_parameter_names=shared_job_parameters.keys(),
    )
    bulk_target_run_task = _get_bulk_target_run_task()

    results: list[dict[str, Any]] = []
    stop_early = False
    for batch in _batched(discovered, concurrency):
        future_entries: list[tuple[str, Any]] = []
        for target in batch:
            relative_path = target.relative_to(input_root_path).as_posix()
            target_name = relative_path.replace("/", "__") or "root"
            job_parameters, parameter_overrides = merge_target_job_parameters(
                base_job_parameters=shared_job_parameters,
                target_overrides=prepared_target_overrides,
                relative_path=relative_path,
            )
            future = bulk_target_run_task.submit(
                target_name=target_name,
                mode=mode,
                input_dir=str(target),
                relative_path=relative_path,
                output_root_dir=str(resolved_output_root),
                count_dict_filename=count_dict_filename,
                fcidump_filename=fcidump_filename,
                command_block_name=command_name,
                execution_profile_block_name=execution_name,
                hpc_profile_block_name=hpc_profile_block_name,
                max_jobs_in_queue=max_jobs_in_queue,
                queue_limit_scope=queue_limit_scope,
                queue_poll_interval_seconds=queue_poll_interval_seconds,
                job_name_prefix=job_name_prefix,
                skip_completed=skip_completed,
                max_attempts=max_target_task_retries + 1,
                job_parameters=job_parameters,
                parameter_overrides=parameter_overrides,
            )
            future_entries.append((relative_path, future))

        for relative_path, future in future_entries:
            try:
                result = future.result()
            except Exception as exc:
                logger.exception("Bulk target task crashed for %s", relative_path)
                result = {
                    "status": "failed",
                    "relative_path": relative_path,
                    "error": str(exc),
                }
            results.append(result)
            if fail_fast and result.get("status") == "failed":
                stop_early = True
        if stop_early:
            break

    summary = {
        "mode": mode,
        "input_root_dir": str(input_root_path),
        "output_root_dir": str(resolved_output_root),
        "total_discovered_targets": len(discovered),
        "configured_target_overrides": prepared_target_overrides,
        "processed_targets": len(results),
        "skipped_targets": sum(1 for result in results if result.get("status") == "skipped"),
        "succeeded_targets": sum(1 for result in results if result.get("status") == "success"),
        "failed_targets": sum(1 for result in results if result.get("status") == "failed"),
        "results": results,
    }
    _write_summary(resolved_output_root, summary)

    if summary["failed_targets"] > 0:
        raise RuntimeError(
            f"Bulk GB-SQD flow finished with {summary['failed_targets']} failed targets. "
            f"See {_summary_path(resolved_output_root)}"
        )

    logger.info(
        "Bulk GB-SQD flow complete: discovered=%s succeeded=%s skipped=%s failed=%s",
        summary["total_discovered_targets"],
        summary["succeeded_targets"],
        summary["skipped_targets"],
        summary["failed_targets"],
    )
    return summary
