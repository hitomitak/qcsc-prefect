"""Recovery iteration task for GB SQD workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prefect import get_run_logger, task

import sys
_project_root = Path(__file__).resolve().parents[4]
if (_project_root / "packages").exists():
    sys.path.insert(0, str(_project_root / "packages" / "hpc-prefect-executor" / "src"))

from hpc_prefect_executor.from_blocks import run_job_from_blocks


@task(
    name="recovery_iteration_{iteration_id}",
    retries=1,
    retry_delay_seconds=30,
    task_run_name="recovery_iter_{iteration_id}",
)
async def recovery_iteration_task(
    iteration_id: int,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    init_data: dict[str, Any],
    previous_result: dict[str, Any] | None,
    mode: str,
    num_batches: int,
    num_samples_per_batch: int | None,
    num_samples_per_recovery: int | None,
    iteration: int,
    block: int,
    tolerance: float,
    max_time: float,
    adet_comm_size: int,
    bdet_comm_size: int,
    task_comm_size: int,
    carryover_threshold: float,
    carryover_ratio: float | None,
    carryover_ratio_batch: float | None,
    carryover_ratio_combined: float | None,
    adet_comm_size_combined: int | None,
    bdet_comm_size_combined: int | None,
    task_comm_size_combined: int | None,
    with_hf: bool,
    verbose: bool,
    work_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Execute one recovery iteration with all batches.
    
    This task runs the gb-demo binary for a single recovery iteration,
    which internally processes all batches.
    
    Args:
        iteration_id: Recovery iteration index (0-based)
        command_block_name: Name of the CommandBlock
        execution_profile_block_name: Name of the ExecutionProfileBlock
        hpc_profile_block_name: Name of the HPCProfileBlock
        init_data: Initialization data from initialize_task
        previous_result: Result from previous iteration (None for first iteration)
        mode: "ext_sqd" or "trim_sqd"
        num_batches: Number of batches
        num_samples_per_batch: Samples per batch (ExtSQD)
        num_samples_per_recovery: Samples per recovery (TrimSQD)
        iteration: Maximum Davidson iterations
        block: Maximum Ritz vector space size
        tolerance: Convergence tolerance
        max_time: Maximum time in seconds
        adet_comm_size: Alpha-determinant comm size
        bdet_comm_size: Beta-determinant comm size
        task_comm_size: Task-level comm size
        carryover_threshold: Carryover threshold
        carryover_ratio: Carryover ratio (ExtSQD)
        carryover_ratio_batch: Batch carryover ratio (TrimSQD)
        carryover_ratio_combined: Combined carryover ratio (TrimSQD)
        adet_comm_size_combined: Combined alpha comm size (TrimSQD)
        bdet_comm_size_combined: Combined beta comm size (TrimSQD)
        task_comm_size_combined: Combined task comm size (TrimSQD)
        with_hf: Include HF state
        verbose: Enable verbose logging
        work_dir: Working directory
        **kwargs: Additional parameters
    
    Returns:
        Dictionary containing iteration results and paths
    
    Raises:
        RuntimeError: If the iteration fails or output files are not found
    """
    logger = get_run_logger()
    logger.info(f"Starting recovery iteration {iteration_id}")
    
    work_path = Path(work_dir).expanduser().resolve()
    
    # Create iteration-specific directory
    iter_dir = work_path / f"recovery_{iteration_id}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for previous carryover
    carryover_file = None
    if previous_result:
        carryover_file = previous_result.get("carryover_file")
        if carryover_file and Path(carryover_file).exists():
            logger.info(f"Using carryover from iteration {iteration_id-1}: {carryover_file}")
    
    # Build command arguments
    user_args = [
        "--fcidump", init_data["fcidump_file"],
        "--count_dict_file", init_data["count_dict_file"],
        "--mode", mode,
        "--num_batches", str(num_batches),
        "--num_recovery", "1",  # Single iteration per task
        "--iteration", str(iteration),
        "--block", str(block),
        "--tolerance", str(tolerance),
        "--max_time", str(max_time),
        "--adet_comm_size", str(adet_comm_size),
        "--bdet_comm_size", str(bdet_comm_size),
        "--task_comm_size", str(task_comm_size),
        "--carryover_threshold", str(carryover_threshold),
        "--output_dir", str(iter_dir),
    ]
    
    # Mode-specific parameters
    if mode == "ext_sqd":
        if num_samples_per_batch is not None:
            user_args.extend(["--num_samples_per_batch", str(num_samples_per_batch)])
        if carryover_ratio is not None:
            user_args.extend(["--carryover_ratio", str(carryover_ratio)])
    elif mode == "trim_sqd":
        if num_samples_per_recovery is not None:
            user_args.extend(["--num_samples_per_recovery", str(num_samples_per_recovery)])
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
    
    # Optional flags
    if with_hf:
        user_args.append("--with_hf")
    if verbose:
        user_args.append("-v")
    
    # Add carryover file if available
    if carryover_file:
        user_args.extend(["--initial_occupancies", str(carryover_file)])
    
    logger.info(f"Iteration directory: {iter_dir}")
    logger.info(f"Mode: {mode}, Batches: {num_batches}")
    
    # Execute the job
    script_filename = f"recovery_{iteration_id}.pbs"  # or .pjm for Fugaku
    
    result = await run_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=iter_dir,
        script_filename=script_filename,
        user_args=user_args,
        watch_poll_interval=10.0,
        timeout_seconds=max_time * 2,  # Allow buffer
        metrics_artifact_key=f"gb-sqd-recovery-{iteration_id}-metrics",
    )
    
    # Verify output files
    energy_log_file = iter_dir / "energy_log.json"
    carryover_output = iter_dir / "carryover_bitstrings.txt"
    
    if not energy_log_file.exists():
        logger.warning(f"energy_log.json not found in {iter_dir}")
        energy_data = None
    else:
        with open(energy_log_file, "r") as f:
            energy_data = json.load(f)
        logger.info(f"Energy: {energy_data.get('energy_final', 'N/A')}")
    
    # Check for carryover file
    if not carryover_output.exists():
        logger.warning(f"Carryover file not found: {carryover_output}")
        carryover_output = None
    else:
        logger.info(f"Carryover saved: {carryover_output}")
    
    logger.info(f"✓ Recovery iteration {iteration_id} complete")
    
    return {
        "iteration_id": iteration_id,
        "status": "success",
        "iter_dir": str(iter_dir),
        "energy_log_file": str(energy_log_file) if energy_log_file.exists() else None,
        "carryover_file": str(carryover_output) if carryover_output else None,
        "energy_data": energy_data,
        "job_result": result,
    }

