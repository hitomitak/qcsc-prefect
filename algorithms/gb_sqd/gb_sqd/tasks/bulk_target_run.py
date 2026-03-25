"""Per-target task for GB-SQD bulk submission."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

from prefect import get_run_logger, task

_project_root = Path(__file__).resolve().parents[4]
if (_project_root / "packages").exists():
    sys.path.insert(0, str(_project_root / "packages" / "qcsc-prefect-core" / "src"))
    sys.path.insert(0, str(_project_root / "packages" / "qcsc-prefect-adapters" / "src"))
    sys.path.insert(0, str(_project_root / "packages" / "qcsc-prefect-blocks" / "src"))
    sys.path.insert(0, str(_project_root / "packages" / "qcsc-prefect-executor" / "src"))

from qcsc_prefect_blocks.common.blocks import HPCProfileBlock
from qcsc_prefect_executor.from_blocks import run_job_from_blocks

from ..artifact_keys import bulk_metrics_artifact_key
from ..cli_args import build_ext_sqd_user_args, build_trim_sqd_user_args
from ..fugaku_queue import wait_for_queue_slot


class NonRetryableBulkError(RuntimeError):
    """Raised when retrying the target would not help."""


def _script_filename(mode: str, hpc_target: str) -> str:
    stem = "gb_sqd_ext" if mode == "ext_sqd" else "gb_sqd_trim"
    suffix = ".pjm" if hpc_target == "fugaku" else ".pbs"
    return stem + suffix


def _make_fugaku_job_name(job_name_prefix: str, mode: str, relative_path: str, attempt: int) -> str:
    digest = hashlib.sha1(f"{mode}:{relative_path}:{attempt}".encode("utf-8")).hexdigest()[:10]
    normalized_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in job_name_prefix).strip("-")
    normalized_prefix = normalized_prefix or "gbsqd"
    mode_tag = "ext" if mode == "ext_sqd" else "trim"
    return f"{normalized_prefix}-{mode_tag}-{digest}"[:63]


def _status_file(target_output_root: Path) -> Path:
    return target_output_root / "target_status.json"


def _load_status(target_output_root: Path) -> dict[str, Any]:
    status_file = _status_file(target_output_root)
    if not status_file.exists():
        return {"attempts": []}
    return json.loads(status_file.read_text())


def _write_status(target_output_root: Path, status: dict[str, Any]) -> None:
    target_output_root.mkdir(parents=True, exist_ok=True)
    _status_file(target_output_root).write_text(json.dumps(status, indent=2))


def _is_successful_status(status: dict[str, Any]) -> bool:
    if status.get("status") != "success":
        return False
    energy_log_file = status.get("energy_log_file")
    return bool(energy_log_file and Path(energy_log_file).exists())


def _next_attempt_number(target_output_root: Path, status: dict[str, Any]) -> int:
    seen_attempts = [int(entry.get("attempt", 0)) for entry in status.get("attempts", [])]
    for candidate in target_output_root.glob("attempt_*"):
        if candidate.is_dir():
            try:
                seen_attempts.append(int(candidate.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return max(seen_attempts, default=0) + 1


async def _resolve_loaded_block(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _build_user_args(
    *,
    mode: str,
    fcidump_file: str | Path,
    count_dict_file: str | Path,
    output_dir: str | Path,
    job_parameters: dict[str, Any],
) -> list[str]:
    if mode == "ext_sqd":
        return build_ext_sqd_user_args(
            fcidump_file=fcidump_file,
            count_dict_file=count_dict_file,
            output_dir=output_dir,
            num_recovery=job_parameters["num_recovery"],
            num_batches=job_parameters["num_batches"],
            num_samples_per_batch=job_parameters["num_samples_per_batch"],
            iteration=job_parameters["iteration"],
            block=job_parameters["block"],
            tolerance=job_parameters["tolerance"],
            max_time=job_parameters["max_time"],
            adet_comm_size=job_parameters["adet_comm_size"],
            bdet_comm_size=job_parameters["bdet_comm_size"],
            task_comm_size=job_parameters["task_comm_size"],
            adet_comm_size_final=job_parameters.get("adet_comm_size_final"),
            bdet_comm_size_final=job_parameters.get("bdet_comm_size_final"),
            task_comm_size_final=job_parameters.get("task_comm_size_final"),
            do_carryover_in_recovery=job_parameters.get("do_carryover_in_recovery", False),
            carryover_threshold=job_parameters["carryover_threshold"],
            carryover_ratio=job_parameters["carryover_ratio"],
            with_hf=job_parameters["with_hf"],
            verbose=job_parameters["verbose"],
        )
    if mode == "trim_sqd":
        return build_trim_sqd_user_args(
            fcidump_file=fcidump_file,
            count_dict_file=count_dict_file,
            output_dir=output_dir,
            num_recovery=job_parameters["num_recovery"],
            num_batches=job_parameters["num_batches"],
            num_samples_per_recovery=job_parameters["num_samples_per_recovery"],
            iteration=job_parameters["iteration"],
            block=job_parameters["block"],
            tolerance=job_parameters["tolerance"],
            max_time=job_parameters["max_time"],
            adet_comm_size=job_parameters["adet_comm_size"],
            bdet_comm_size=job_parameters["bdet_comm_size"],
            task_comm_size=job_parameters["task_comm_size"],
            adet_comm_size_combined=job_parameters.get("adet_comm_size_combined"),
            bdet_comm_size_combined=job_parameters.get("bdet_comm_size_combined"),
            task_comm_size_combined=job_parameters.get("task_comm_size_combined"),
            adet_comm_size_final=job_parameters.get("adet_comm_size_final"),
            bdet_comm_size_final=job_parameters.get("bdet_comm_size_final"),
            task_comm_size_final=job_parameters.get("task_comm_size_final"),
            carryover_ratio_batch=job_parameters["carryover_ratio_batch"],
            carryover_ratio_combined=job_parameters["carryover_ratio_combined"],
            carryover_threshold=job_parameters["carryover_threshold"],
            with_hf=job_parameters["with_hf"],
            verbose=job_parameters["verbose"],
        )
    raise NonRetryableBulkError(f"Unsupported GB-SQD mode: {mode}")


@task(name="bulk_target_run", task_run_name="bulk_target_{target_name}")
async def bulk_target_run_task(
    *,
    target_name: str,
    mode: str,
    input_dir: str,
    relative_path: str,
    output_root_dir: str,
    count_dict_filename: str,
    fcidump_filename: str,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    max_jobs_in_queue: int,
    queue_limit_scope: str,
    queue_poll_interval_seconds: float,
    job_name_prefix: str,
    skip_completed: bool,
    max_attempts: int,
    job_parameters: dict[str, Any],
    parameter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one monolithic GB-SQD job for a discovered target directory."""

    logger = get_run_logger()

    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.exists():
        raise NonRetryableBulkError(f"Target input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NonRetryableBulkError(f"Target input path is not a directory: {input_path}")

    fcidump_path = input_path / fcidump_filename
    count_dict_path = input_path / count_dict_filename
    if not fcidump_path.exists():
        raise NonRetryableBulkError(f"FCIDUMP file not found: {fcidump_path}")
    if not count_dict_path.exists():
        raise NonRetryableBulkError(f"Count dictionary file not found: {count_dict_path}")

    target_output_root = Path(output_root_dir).expanduser().resolve() / Path(relative_path)
    applied_overrides = dict(parameter_overrides or {})
    status = _load_status(target_output_root)
    if skip_completed and _is_successful_status(status):
        logger.info("Skipping completed target: %s", relative_path)
        return {
            "status": "skipped",
            "relative_path": relative_path,
            "input_dir": str(input_path),
            "latest_attempt": status.get("latest_attempt"),
            "latest_output_dir": status.get("latest_output_dir"),
            "latest_job_id": status.get("latest_job_id"),
            "energy_final": status.get("energy_final"),
            "energy_log_file": status.get("energy_log_file"),
            "parameter_overrides": {},
        }

    hpc_block = await _resolve_loaded_block(HPCProfileBlock.load(hpc_profile_block_name))
    queue_name = hpc_block.queue_cpu
    hpc_target = hpc_block.hpc_target
    script_filename = _script_filename(mode, hpc_target)

    attempts_used = 0
    next_attempt = _next_attempt_number(target_output_root, status)
    for attempt_number in range(next_attempt, next_attempt + max_attempts):
        attempts_used += 1
        attempt_dir = target_output_root / f"attempt_{attempt_number:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        try:
            if hpc_target == "fugaku":
                active_count = await wait_for_queue_slot(
                    resource_group=queue_name,
                    max_jobs_in_queue=max_jobs_in_queue,
                    scope=queue_limit_scope,
                    job_name_prefix=job_name_prefix,
                    poll_interval_seconds=queue_poll_interval_seconds,
                )
                logger.info(
                    "Queue slot acquired for %s (active jobs before submit: %s)",
                    relative_path,
                    active_count,
                )

            user_args = _build_user_args(
                mode=mode,
                fcidump_file=fcidump_path,
                count_dict_file=count_dict_path,
                output_dir=attempt_dir,
                job_parameters=job_parameters,
            )
            job_name = _make_fugaku_job_name(job_name_prefix, mode, relative_path, attempt_number)
            result = await run_job_from_blocks(
                command_block_name=command_block_name,
                execution_profile_block_name=execution_profile_block_name,
                hpc_profile_block_name=hpc_profile_block_name,
                work_dir=attempt_dir,
                script_filename=script_filename,
                user_args=user_args,
                watch_poll_interval=10.0,
                timeout_seconds=float(job_parameters["max_time"]) * 2,
                metrics_artifact_key=bulk_metrics_artifact_key(mode),
                fugaku_job_name=job_name if hpc_target == "fugaku" else None,
            )
            if result.exit_status != 0:
                raise RuntimeError(f"gb-demo failed with exit_status={result.exit_status}")

            energy_log_file = attempt_dir / "energy_log.json"
            if not energy_log_file.exists():
                raise RuntimeError(f"energy_log.json not found: {energy_log_file}")

            energy_log = json.loads(energy_log_file.read_text())
            status.setdefault("attempts", []).append(
                {
                    "attempt": attempt_number,
                    "status": "success",
                    "work_dir": str(attempt_dir),
                    "job_id": getattr(result, "job_id", None),
                    "exit_status": result.exit_status,
                    "state": getattr(result, "state", None),
                    "energy_log_file": str(energy_log_file),
                    "parameter_overrides": applied_overrides,
                }
            )
            status.update(
                {
                    "status": "success",
                    "relative_path": relative_path,
                    "input_dir": str(input_path),
                    "latest_attempt": attempt_number,
                    "latest_output_dir": str(attempt_dir),
                    "latest_job_id": getattr(result, "job_id", None),
                    "energy_log_file": str(energy_log_file),
                    "energy_final": energy_log.get("energy_final"),
                    "latest_parameter_overrides": applied_overrides,
                }
            )
            _write_status(target_output_root, status)
            return {
                "status": "success",
                "relative_path": relative_path,
                "input_dir": str(input_path),
                "latest_attempt": attempt_number,
                "latest_output_dir": str(attempt_dir),
                "latest_job_id": getattr(result, "job_id", None),
                "energy_log_file": str(energy_log_file),
                "energy_final": energy_log.get("energy_final"),
                "parameter_overrides": applied_overrides,
            }
        except NonRetryableBulkError as exc:
            status.setdefault("attempts", []).append(
                {
                    "attempt": attempt_number,
                    "status": "failed",
                    "work_dir": str(attempt_dir),
                    "error": str(exc),
                    "retryable": False,
                    "parameter_overrides": applied_overrides,
                }
            )
            status.update(
                {
                    "status": "failed",
                    "relative_path": relative_path,
                    "input_dir": str(input_path),
                    "latest_attempt": attempt_number,
                    "latest_output_dir": str(attempt_dir),
                    "error": str(exc),
                    "latest_parameter_overrides": applied_overrides,
                }
            )
            _write_status(target_output_root, status)
            return {
                "status": "failed",
                "relative_path": relative_path,
                "input_dir": str(input_path),
                "latest_attempt": attempt_number,
                "latest_output_dir": str(attempt_dir),
                "error": str(exc),
                "retryable": False,
                "parameter_overrides": applied_overrides,
            }
        except Exception as exc:
            logger.warning("Attempt %s failed for %s: %s", attempt_number, relative_path, exc)
            status.setdefault("attempts", []).append(
                {
                    "attempt": attempt_number,
                    "status": "failed",
                    "work_dir": str(attempt_dir),
                    "error": str(exc),
                    "retryable": True,
                    "parameter_overrides": applied_overrides,
                }
            )
            status.update(
                {
                    "status": "failed",
                    "relative_path": relative_path,
                    "input_dir": str(input_path),
                    "latest_attempt": attempt_number,
                    "latest_output_dir": str(attempt_dir),
                    "error": str(exc),
                    "latest_parameter_overrides": applied_overrides,
                }
            )
            _write_status(target_output_root, status)

    return {
        "status": "failed",
        "relative_path": relative_path,
        "input_dir": str(input_path),
        "latest_attempt": status.get("latest_attempt"),
        "latest_output_dir": status.get("latest_output_dir"),
        "error": status.get("error", "Target failed after retries"),
        "retryable": True,
        "attempts_used": attempts_used,
        "parameter_overrides": applied_overrides,
    }
