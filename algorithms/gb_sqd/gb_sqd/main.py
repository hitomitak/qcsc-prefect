"""GB SQD Prefect workflows."""

import json
import os
import pathlib
import sys
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger

# qcsc-prefectパッケージのインポート
# 開発時はローカルのqcsc-prefectを参照
_project_root = Path(__file__).resolve().parents[3]
if (_project_root / "packages").exists():
    sys.path.insert(0, str(_project_root / "packages" / "qcsc-prefect-executor" / "src"))
    sys.path.insert(0, str(_project_root / "packages" / "qcsc-prefect-blocks" / "src"))

from qcsc_prefect_executor.from_blocks import run_job_from_blocks

from .bulk import bulk_gb_sqd_flow
from .cli_args import build_ext_sqd_user_args, build_trim_sqd_user_args


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
    num_samples_per_batch: int = 1,
    iteration: int = 1,
    block: int = 10,
    tolerance: float = 1.0e-2,
    max_time: float = 3600.0,
    adet_comm_size: int = 1,
    bdet_comm_size: int = 1,
    task_comm_size: int = 1,
    adet_comm_size_final: int | None = None,
    bdet_comm_size_final: int | None = None,
    task_comm_size_final: int | None = None,
    do_carryover_in_recovery: bool = False,
    carryover_threshold: float = 1.0e-2,
    carryover_ratio: float = 0.5,
    with_hf: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run a simple ExtSQD workflow (single execution).
    
    This is a minimal implementation that wraps the existing gb-demo binary.
    
    Args:
        init_command_block_name: Name of init CommandBlock
        recovery_command_block_name: Name of recovery CommandBlock
        finalize_command_block_name: Name of finalize CommandBlock
        init_execution_profile_block_name: Name of init ExecutionProfileBlock
        recovery_execution_profile_block_name: Name of recovery ExecutionProfileBlock
        finalize_execution_profile_block_name: Name of finalize ExecutionProfileBlock
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
    
    user_args = build_ext_sqd_user_args(
        fcidump_file=fcidump_file,
        count_dict_file=count_dict_file,
        output_dir=work_path,
        num_recovery=num_recovery,
        num_batches=num_batches,
        num_samples_per_batch=num_samples_per_batch,
        iteration=iteration,
        block=block,
        tolerance=tolerance,
        max_time=max_time,
        adet_comm_size=adet_comm_size,
        bdet_comm_size=bdet_comm_size,
        task_comm_size=task_comm_size,
        adet_comm_size_final=adet_comm_size_final,
        bdet_comm_size_final=bdet_comm_size_final,
        task_comm_size_final=task_comm_size_final,
        do_carryover_in_recovery=do_carryover_in_recovery,
        carryover_threshold=carryover_threshold,
        carryover_ratio=carryover_ratio,
        with_hf=with_hf,
        verbose=verbose,
    )
    
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
    carryover_ratio_batch: float = 0.1,
    carryover_ratio_combined: float = 0.5,
    carryover_threshold: float = 1.0e-2,
    with_hf: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run a simple TrimSQD workflow (single execution).
    
    Args:
        init_command_block_name: Name of init CommandBlock
        recovery_command_block_name: Name of recovery CommandBlock
        finalize_command_block_name: Name of finalize CommandBlock
        init_execution_profile_block_name: Name of init ExecutionProfileBlock
        recovery_execution_profile_block_name: Name of recovery ExecutionProfileBlock
        finalize_execution_profile_block_name: Name of finalize ExecutionProfileBlock
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
    
    user_args = build_trim_sqd_user_args(
        fcidump_file=fcidump_file,
        count_dict_file=count_dict_file,
        output_dir=work_path,
        num_recovery=num_recovery,
        num_batches=num_batches,
        num_samples_per_recovery=num_samples_per_recovery,
        iteration=iteration,
        block=block,
        tolerance=tolerance,
        max_time=max_time,
        adet_comm_size=adet_comm_size,
        bdet_comm_size=bdet_comm_size,
        task_comm_size=task_comm_size,
        adet_comm_size_combined=adet_comm_size_combined,
        bdet_comm_size_combined=bdet_comm_size_combined,
        task_comm_size_combined=task_comm_size_combined,
        adet_comm_size_final=adet_comm_size_final,
        bdet_comm_size_final=bdet_comm_size_final,
        task_comm_size_final=task_comm_size_final,
        carryover_ratio_batch=carryover_ratio_batch,
        carryover_ratio_combined=carryover_ratio_combined,
        carryover_threshold=carryover_threshold,
        with_hf=with_hf,
        verbose=verbose,
    )
    
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

# ============================================================================
# Task-based workflows with improved visibility and restart capability
# ============================================================================

from .tasks import (
    initialize_task,
    recovery_iteration_task,
    final_diagonalization_task,
    output_results_task,
)


@flow(name="GB-SQD-ExtSQD")
async def ext_sqd_flow(
    fcidump_file: str,
    count_dict_file: str,
    work_dir: str,
    init_command_block_name: str = "cmd-gb-sqd-init",
    recovery_command_block_name: str = "cmd-gb-sqd-recovery",
    finalize_command_block_name: str = "cmd-gb-sqd-finalize",
    init_execution_profile_block_name: str = "exec-gb-sqd-init-miyabi",
    recovery_execution_profile_block_name: str = "exec-gb-sqd-recovery-miyabi",
    finalize_execution_profile_block_name: str = "exec-gb-sqd-finalize-miyabi",
    hpc_profile_block_name: str = "hpc-miyabi-gb-sqd",
    num_recovery: int = 3,
    num_iters_per_recovery: int = 1,
    num_batches: int = 8,
    num_samples_per_batch: int = 1000,
    iteration: int = 1,
    block: int = 10,
    tolerance: float = 1.0e-2,
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
    Run ExtSQD workflow with task-based execution for improved visibility.
    
    This workflow splits execution into multiple tasks for better observability.
    
    Args:
        init_command_block_name: Name of init CommandBlock
        recovery_command_block_name: Name of recovery CommandBlock
        finalize_command_block_name: Name of finalize CommandBlock
        init_execution_profile_block_name: Name of init ExecutionProfileBlock
        recovery_execution_profile_block_name: Name of recovery ExecutionProfileBlock
        finalize_execution_profile_block_name: Name of finalize ExecutionProfileBlock
        hpc_profile_block_name: Name of the HPCProfileBlock
        fcidump_file: Path to FCIDUMP file
        count_dict_file: Path to count dictionary file
        work_dir: Working directory for outputs
        num_recovery: Number of recovery tasks/checkpoints
        num_iters_per_recovery: Number of gb-demo iterations per recovery task
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
    logger.info("Starting GB-SQD ExtSQD workflow (task-based)")

    if num_recovery < 1:
        raise ValueError("num_recovery must be >= 1")
    if num_iters_per_recovery < 1:
        raise ValueError("num_iters_per_recovery must be >= 1")
    
    # Step 1: Initialize
    init_data = await initialize_task(
        mode="ext_sqd",
        command_block_name=init_command_block_name,
        execution_profile_block_name=init_execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        fcidump_file=fcidump_file,
        count_dict_file=count_dict_file,
        work_dir=work_dir,
        verbose=verbose,
        max_time=max_time,
    )
    
    # Step 2: Recovery iterations (sequential)
    recovery_results = []
    for iter_id in range(num_recovery):
        logger.info(f"Recovery iteration {iter_id+1}/{num_recovery}")
        
        result = await recovery_iteration_task(
            iteration_id=iter_id,
            command_block_name=recovery_command_block_name,
            execution_profile_block_name=recovery_execution_profile_block_name,
            hpc_profile_block_name=hpc_profile_block_name,
            init_data=init_data,
            previous_result=recovery_results[-1] if recovery_results else None,
            mode="ext_sqd",
            num_recovery=num_recovery,
            num_iters_per_recovery=num_iters_per_recovery,
            num_batches=num_batches,
            num_samples_per_batch=num_samples_per_batch,
            num_samples_per_recovery=None,
            iteration=iteration,
            block=block,
            tolerance=tolerance,
            max_time=max_time,
            adet_comm_size=adet_comm_size,
            bdet_comm_size=bdet_comm_size,
            task_comm_size=task_comm_size,
            carryover_threshold=carryover_threshold,
            carryover_ratio=carryover_ratio,
            carryover_ratio_batch=None,
            carryover_ratio_combined=None,
            adet_comm_size_combined=None,
            bdet_comm_size_combined=None,
            task_comm_size_combined=None,
            with_hf=with_hf,
            verbose=verbose,
            work_dir=work_dir,
        )
        recovery_results.append(result)
    
    # Step 3: Final diagonalization
    final_result = await final_diagonalization_task(
        command_block_name=finalize_command_block_name,
        execution_profile_block_name=finalize_execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        recovery_results=recovery_results,
        work_dir=work_dir,
        verbose=verbose,
        max_time=max_time,
    )
    
    # Step 4: Output results
    output = output_results_task(
        final_result=final_result,
        work_dir=work_dir,
    )
    
    logger.info("✓ GB-SQD ExtSQD workflow complete")
    return output


@flow(name="GB-SQD-TrimSQD")
async def trim_sqd_flow(
    fcidump_file: str,
    count_dict_file: str,
    work_dir: str,
    init_command_block_name: str = "cmd-gb-sqd-init",
    recovery_command_block_name: str = "cmd-gb-sqd-recovery",
    finalize_command_block_name: str = "cmd-gb-sqd-finalize",
    init_execution_profile_block_name: str = "exec-gb-sqd-init-miyabi",
    recovery_execution_profile_block_name: str = "exec-gb-sqd-recovery-miyabi",
    finalize_execution_profile_block_name: str = "exec-gb-sqd-finalize-miyabi",
    hpc_profile_block_name: str = "hpc-miyabi-gb-sqd",
    num_recovery: int = 3,
    num_iters_per_recovery: int = 1,
    num_batches: int = 8,
    num_samples_per_recovery: int = 10000,
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
    carryover_ratio_batch: float = 0.1,
    carryover_ratio_combined: float = 0.5,
    carryover_threshold: float = 1.0e-2,
    with_hf: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run TrimSQD workflow with task-based execution for improved visibility.
    
    This workflow splits execution into multiple tasks for better observability.
    
    Args:
        init_command_block_name: Name of init CommandBlock
        recovery_command_block_name: Name of recovery CommandBlock
        finalize_command_block_name: Name of finalize CommandBlock
        init_execution_profile_block_name: Name of init ExecutionProfileBlock
        recovery_execution_profile_block_name: Name of recovery ExecutionProfileBlock
        finalize_execution_profile_block_name: Name of finalize ExecutionProfileBlock
        hpc_profile_block_name: Name of the HPCProfileBlock
        fcidump_file: Path to FCIDUMP file
        count_dict_file: Path to count dictionary file
        work_dir: Working directory for outputs
        num_recovery: Number of recovery tasks/checkpoints
        num_iters_per_recovery: Number of gb-demo iterations per recovery task
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
    logger.info("Starting GB-SQD TrimSQD workflow (task-based)")

    if num_recovery < 1:
        raise ValueError("num_recovery must be >= 1")
    if num_iters_per_recovery < 1:
        raise ValueError("num_iters_per_recovery must be >= 1")
    
    # Step 1: Initialize
    init_data = await initialize_task(
        mode="trim_sqd",
        command_block_name=init_command_block_name,
        execution_profile_block_name=init_execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        fcidump_file=fcidump_file,
        count_dict_file=count_dict_file,
        work_dir=work_dir,
        verbose=verbose,
        max_time=max_time,
    )
    
    # Step 2: Recovery iterations (sequential)
    recovery_results = []
    for iter_id in range(num_recovery):
        logger.info(f"Recovery iteration {iter_id+1}/{num_recovery}")
        
        result = await recovery_iteration_task(
            iteration_id=iter_id,
            command_block_name=recovery_command_block_name,
            execution_profile_block_name=recovery_execution_profile_block_name,
            hpc_profile_block_name=hpc_profile_block_name,
            init_data=init_data,
            previous_result=recovery_results[-1] if recovery_results else None,
            mode="trim_sqd",
            num_recovery=num_recovery,
            num_iters_per_recovery=num_iters_per_recovery,
            num_batches=num_batches,
            num_samples_per_batch=None,
            num_samples_per_recovery=num_samples_per_recovery,
            iteration=iteration,
            block=block,
            tolerance=tolerance,
            max_time=max_time,
            adet_comm_size=adet_comm_size,
            bdet_comm_size=bdet_comm_size,
            task_comm_size=task_comm_size,
            carryover_threshold=carryover_threshold,
            carryover_ratio=None,
            carryover_ratio_batch=carryover_ratio_batch,
            carryover_ratio_combined=carryover_ratio_combined,
            adet_comm_size_combined=adet_comm_size_combined,
            bdet_comm_size_combined=bdet_comm_size_combined,
            task_comm_size_combined=task_comm_size_combined,
            with_hf=with_hf,
            verbose=verbose,
            work_dir=work_dir,
        )
        recovery_results.append(result)
    
    # Step 3: Final diagonalization
    final_result = await final_diagonalization_task(
        command_block_name=finalize_command_block_name,
        execution_profile_block_name=finalize_execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        recovery_results=recovery_results,
        work_dir=work_dir,
        carryover_ratio_batch=carryover_ratio_batch,
        carryover_ratio_combined=carryover_ratio_combined,
        adet_comm_size_combined=adet_comm_size_combined,
        bdet_comm_size_combined=bdet_comm_size_combined,
        task_comm_size_combined=task_comm_size_combined,
        verbose=verbose,
        max_time=max_time,
    )
    
    # Step 4: Output results
    output = output_results_task(
        final_result=final_result,
        work_dir=work_dir,
    )
    
    logger.info("✓ GB-SQD TrimSQD workflow complete")
    return output


# ============================================================================
# Deployment
# ============================================================================

def deploy_ext_sqd():
    """Deploy ExtSQD workflow with a local worker."""
    os.chdir(pathlib.Path(__file__).parent)
    ext_sqd_flow.serve(
        name="gb-sqd-ext-sqd",
        description="GB-SQD ExtSQD workflow with task-based execution for improved visibility.",
    )


def deploy_trim_sqd():
    """Deploy TrimSQD workflow with a local worker."""
    os.chdir(pathlib.Path(__file__).parent)
    trim_sqd_flow.serve(
        name="gb-sqd-trim-sqd",
        description="GB-SQD TrimSQD workflow with task-based execution for improved visibility.",
    )
