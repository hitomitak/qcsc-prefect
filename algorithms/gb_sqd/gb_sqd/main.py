"""GB SQD Prefect workflows."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger

# hpc-prefectパッケージのインポート
# 開発時はローカルのhpc-prefectを参照
_project_root = Path(__file__).resolve().parents[3]
if (_project_root / "packages").exists():
    sys.path.insert(0, str(_project_root / "packages" / "hpc-prefect-executor" / "src"))
    sys.path.insert(0, str(_project_root / "packages" / "hpc-prefect-blocks" / "src"))

from hpc_prefect_executor.from_blocks import run_job_from_blocks


@flow(name="GB-SQD-ExtSQD-Simple")
async def ext_sqd_simple_flow(
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    fcidump_file: str,
    count_dict_file: str,
    work_dir: str,
    num_recovery: int = 1,
    num_batches: int = 1,
    num_samples_per_batch: int = 1000,
    iteration: int = 1,
    block: int = 10,
    tolerance: float = 1.0e-4,
    max_time: float = 3600.0,
    adet_comm_size: int = 1,
    bdet_comm_size: int = 1,
    task_comm_size: int = 1,
    carryover_threshold: float = 1.0e-2,
    carryover_ratio: float = 0.5,
    with_hf: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run a simple ExtSQD workflow (single execution).
    
    This is a minimal implementation that wraps the existing gb-demo binary.
    
    Args:
        command_block_name: Name of the CommandBlock
        execution_profile_block_name: Name of the ExecutionProfileBlock
        hpc_profile_block_name: Name of the HPCProfileBlock
        fcidump_file: Path to FCIDUMP file
        count_dict_file: Path to count dictionary file
        work_dir: Working directory for outputs
        num_recovery: Number of configuration recovery iterations
        num_batches: Number of batches
        num_samples_per_batch: Number of samples per batch
        iteration: Maximum number of Davidson iterations
        block: Maximum size of Ritz vector space
        tolerance: Convergence tolerance
        max_time: Maximum allowed time (seconds)
        adet_comm_size: Number of nodes for alpha-determinants
        bdet_comm_size: Number of nodes for beta-determinants
        task_comm_size: MPI communicator size for task-level parallelism
        carryover_threshold: Threshold for carryover selection
        carryover_ratio: Fraction of bitstrings to retain
        with_hf: Whether to include HF state
        verbose: Enable verbose logging
    
    Returns:
        Dictionary containing execution results
    """
    logger = get_run_logger()
    logger.info("Starting GB-SQD ExtSQD workflow")
    
    work_path = Path(work_dir).expanduser().resolve()
    work_path.mkdir(parents=True, exist_ok=True)
    
    # Build command arguments
    user_args = [
        "--fcidump", str(fcidump_file),
        "--count_dict_file", str(count_dict_file),
        "--mode", "ext_sqd",
        "--num_recovery", str(num_recovery),
        "--num_batches", str(num_batches),
        "--num_samples_per_batch", str(num_samples_per_batch),
        "--iteration", str(iteration),
        "--block", str(block),
        "--tolerance", str(tolerance),
        "--max_time", str(max_time),
        "--adet_comm_size", str(adet_comm_size),
        "--bdet_comm_size", str(bdet_comm_size),
        "--task_comm_size", str(task_comm_size),
        "--carryover_threshold", str(carryover_threshold),
        "--carryover_ratio", str(carryover_ratio),
        "--output_dir", str(work_path),
    ]
    
    if with_hf:
        user_args.append("--with_hf")
    
    if verbose:
        user_args.append("-v")
    
    logger.info(f"Work directory: {work_path}")
    logger.info(f"FCIDUMP: {fcidump_file}")
    logger.info(f"Count dict: {count_dict_file}")
    
    # Execute the job
    result = await run_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=work_path,
        script_filename="gb_sqd_ext.pbs",  # or .pjm for Fugaku
        user_args=user_args,
        watch_poll_interval=10.0,
        timeout_seconds=max_time * 2,  # Allow some buffer
        metrics_artifact_key="gb-sqd-ext-metrics",
    )
    
    # Load and return results
    energy_log_file = work_path / "energy_log.json"
    if energy_log_file.exists():
        energy_log = json.loads(energy_log_file.read_text())
        logger.info(f"Final energy: {energy_log.get('energy_final', 'N/A')}")
        return {
            "status": "success",
            "work_dir": str(work_path),
            "energy_log": energy_log,
            "job_result": result,
        }
    else:
        logger.warning("energy_log.json not found")
        return {
            "status": "completed_no_log",
            "work_dir": str(work_path),
            "job_result": result,
        }


@flow(name="GB-SQD-TrimSQD-Simple")
async def trim_sqd_simple_flow(
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    fcidump_file: str,
    count_dict_file: str,
    work_dir: str,
    num_recovery: int = 1,
    num_batches: int = 1,
    num_samples_per_recovery: int = 10000,
    iteration: int = 1,
    block: int = 10,
    tolerance: float = 1.0e-4,
    max_time: float = 3600.0,
    adet_comm_size: int = 1,
    bdet_comm_size: int = 1,
    task_comm_size: int = 1,
    adet_comm_size_combined: int | None = None,
    bdet_comm_size_combined: int | None = None,
    task_comm_size_combined: int | None = None,
    carryover_ratio_batch: float = 0.1,
    carryover_ratio_combined: float = 0.5,
    carryover_threshold: float = 1.0e-2,
    with_hf: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run a simple TrimSQD workflow (single execution).
    
    Args:
        command_block_name: Name of the CommandBlock
        execution_profile_block_name: Name of the ExecutionProfileBlock
        hpc_profile_block_name: Name of the HPCProfileBlock
        fcidump_file: Path to FCIDUMP file
        count_dict_file: Path to count dictionary file
        work_dir: Working directory for outputs
        num_recovery: Number of configuration recovery iterations
        num_batches: Number of batches
        num_samples_per_recovery: Number of samples per recovery iteration
        iteration: Maximum number of Davidson iterations
        block: Maximum size of Ritz vector space
        tolerance: Convergence tolerance
        max_time: Maximum allowed time (seconds)
        adet_comm_size: Number of nodes for alpha-determinants (batch)
        bdet_comm_size: Number of nodes for beta-determinants (batch)
        task_comm_size: MPI communicator size (batch)
        adet_comm_size_combined: Number of nodes for alpha-determinants (combined)
        bdet_comm_size_combined: Number of nodes for beta-determinants (combined)
        task_comm_size_combined: MPI communicator size (combined)
        carryover_ratio_batch: Carryover ratio for batch diagonalization
        carryover_ratio_combined: Carryover ratio for combined diagonalization
        carryover_threshold: Threshold for carryover selection
        with_hf: Whether to include HF state
        verbose: Enable verbose logging
    
    Returns:
        Dictionary containing execution results
    """
    logger = get_run_logger()
    logger.info("Starting GB-SQD TrimSQD workflow")
    
    work_path = Path(work_dir).expanduser().resolve()
    work_path.mkdir(parents=True, exist_ok=True)
    
    # Build command arguments
    user_args = [
        "--fcidump", str(fcidump_file),
        "--count_dict_file", str(count_dict_file),
        "--mode", "trim_sqd",
        "--num_recovery", str(num_recovery),
        "--num_batches", str(num_batches),
        "--num_samples_per_recovery", str(num_samples_per_recovery),
        "--iteration", str(iteration),
        "--block", str(block),
        "--tolerance", str(tolerance),
        "--max_time", str(max_time),
        "--adet_comm_size", str(adet_comm_size),
        "--bdet_comm_size", str(bdet_comm_size),
        "--task_comm_size", str(task_comm_size),
        "--carryover_ratio_batch", str(carryover_ratio_batch),
        "--carryover_ratio_combined", str(carryover_ratio_combined),
        "--carryover_threshold", str(carryover_threshold),
        "--output_dir", str(work_path),
    ]
    
    if adet_comm_size_combined is not None:
        user_args.extend(["--adet_comm_size_combined", str(adet_comm_size_combined)])
    if bdet_comm_size_combined is not None:
        user_args.extend(["--bdet_comm_size_combined", str(bdet_comm_size_combined)])
    if task_comm_size_combined is not None:
        user_args.extend(["--task_comm_size_combined", str(task_comm_size_combined)])
    
    if with_hf:
        user_args.append("--with_hf")
    
    if verbose:
        user_args.append("-v")
    
    logger.info(f"Work directory: {work_path}")
    logger.info(f"FCIDUMP: {fcidump_file}")
    logger.info(f"Count dict: {count_dict_file}")
    
    # Execute the job
    result = await run_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=work_path,
        script_filename="gb_sqd_trim.pbs",  # or .pjm for Fugaku
        user_args=user_args,
        watch_poll_interval=10.0,
        timeout_seconds=max_time * 2,
        metrics_artifact_key="gb-sqd-trim-metrics",
    )
    
    # Load and return results
    energy_log_file = work_path / "energy_log.json"
    if energy_log_file.exists():
        energy_log = json.loads(energy_log_file.read_text())
        logger.info(f"Final energy: {energy_log.get('energy_final', 'N/A')}")
        return {
            "status": "success",
            "work_dir": str(work_path),
            "energy_log": energy_log,
            "job_result": result,
        }
    else:
        logger.warning("energy_log.json not found")
        return {
            "status": "completed_no_log",
            "work_dir": str(work_path),
            "job_result": result,
        }

