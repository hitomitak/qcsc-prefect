"""Final diagonalization task for GB SQD workflow."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from prefect import get_run_logger, task

_project_root = Path(__file__).resolve().parents[4]
if (_project_root / "packages").exists():
    sys.path.insert(0, str(_project_root / "packages" / "qcsc-prefect-executor" / "src"))

from qcsc_prefect_executor.from_blocks import run_job_from_blocks


@task(
    name="final_diagonalization",
    retries=1,
    retry_delay_seconds=30,
    task_run_name="final_diagonalization",
)
async def final_diagonalization_task(
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    recovery_results: list[dict[str, Any]],
    work_dir: str | Path,
    carryover_ratio_batch: float | None = None,
    carryover_ratio_combined: float | None = None,
    adet_comm_size_combined: int | None = None,
    bdet_comm_size_combined: int | None = None,
    task_comm_size_combined: int | None = None,
    verbose: bool = True,
    max_time: float = 3600.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Perform final diagonalization on all recovery results.
    
    This task runs `gb-demo finalize` and prepares the final output.
    
    Args:
        recovery_results: List of results from recovery_iteration_task
        work_dir: Working directory
        **kwargs: Additional parameters
    
    Returns:
        Dictionary containing final results
    """
    logger = get_run_logger()
    logger.info("Starting final diagonalization")
    
    work_path = Path(work_dir).expanduser().resolve()
    
    if not recovery_results:
        raise RuntimeError("No recovery results were provided to final_diagonalization_task")
    state_in = Path(recovery_results[-1]["state_file"]).expanduser().resolve()
    if not state_in.exists():
        raise RuntimeError(f"Finalization input state file not found: {state_in}")

    user_args = [
        "--state-in", str(state_in),
        "--output_dir", str(work_path),
    ]
    if carryover_ratio_batch is not None:
        user_args.extend(["--carryover_ratio_batch", str(carryover_ratio_batch)])
    if carryover_ratio_combined is not None:
        user_args.extend(["--carryover_ratio_combined", str(carryover_ratio_combined)])
    if adet_comm_size_combined is not None:
        user_args.extend(["--adet_comm_size_combined", str(adet_comm_size_combined)])
    if bdet_comm_size_combined is not None:
        user_args.extend(["--bdet_comm_size_combined", str(bdet_comm_size_combined)])
    if task_comm_size_combined is not None:
        user_args.extend(["--task_comm_size_combined", str(task_comm_size_combined)])
    if verbose:
        user_args.append("-v")

    result = await run_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=work_path,
        script_filename="finalize.pbs",
        user_args=user_args,
        watch_poll_interval=10.0,
        timeout_seconds=max_time * 2,
        metrics_artifact_key="gb-sqd-finalize-metrics",
    )
    if result.exit_status != 0:
        raise RuntimeError(f"gb-demo finalize failed: exit_status={result.exit_status}")

    energy_log_file = work_path / "energy_log.json"
    if not energy_log_file.exists():
        raise RuntimeError(f"energy_log.json was not generated: {energy_log_file}")
    energy_log = json.loads(energy_log_file.read_text())
    energy_history = energy_log.get("energy_history", [])
    energies = [{"iteration": i, "energy": e} for i, e in enumerate(energy_history)]
    final_energy = energy_log.get("energy_final")
    logger.info(f"Final energy: {final_energy}")
    
    # Prepare final result
    final_result = {
        "status": "success",
        "num_iterations": len(recovery_results),
        "energies": energies,
        "energy_final": final_energy,
        "energy_log_file": str(energy_log_file),
        "job_result": {
            "job_id": getattr(result, "job_id", None),
            "exit_status": result.exit_status,
        },
        "recovery_results": recovery_results,
    }
    
    # Save final result
    final_result_file = work_path / "final_result.json"
    with open(final_result_file, "w") as f:
        # Create a serializable version (exclude job_result objects)
        serializable_result = {
            "status": final_result["status"],
            "num_iterations": final_result["num_iterations"],
            "energies": final_result["energies"],
            "energy_final": final_result["energy_final"],
        }
        json.dump(serializable_result, f, indent=2)
    
    logger.info(f"✓ Final diagonalization complete: {final_result_file}")
    
    return final_result
