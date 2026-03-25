"""Initialization task for GB SQD workflow."""

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
    name="initialize",
    retries=2,
    retry_delay_seconds=10,
    task_run_name="initialize",
)
async def initialize_task(
    mode: str,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    fcidump_file: str,
    count_dict_file: str,
    work_dir: str | Path,
    initial_occupancies_file: str | None = None,
    verbose: bool = True,
    max_time: float = 3600.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Initialize GB-SQD workflow and validate inputs.
    
    Args:
        fcidump_file: Path to FCIDUMP file
        count_dict_file: Path to count dictionary file
        work_dir: Working directory for outputs
        **kwargs: Additional parameters to save
    
    Returns:
        Dictionary containing initialization data
    
    Raises:
        FileNotFoundError: If required input files are not found
    """
    logger = get_run_logger()
    logger.info("Initializing GB-SQD workflow via gb-demo init")
    
    # Convert to Path objects
    fcidump_path = Path(fcidump_file).expanduser().resolve()
    count_dict_path = Path(count_dict_file).expanduser().resolve()
    work_path = Path(work_dir).expanduser().resolve()
    
    # Validate input files
    if not fcidump_path.exists():
        raise FileNotFoundError(f"FCIDUMP file not found: {fcidump_path}")
    if not count_dict_path.exists():
        raise FileNotFoundError(f"Count dictionary file not found: {count_dict_path}")
    
    logger.info(f"FCIDUMP: {fcidump_path}")
    logger.info(f"Count dict: {count_dict_path}")
    
    # Create working directory
    work_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Work directory: {work_path}")

    # Prepare init command args
    init_dir = work_path / "init"
    init_dir.mkdir(parents=True, exist_ok=True)
    state_file = init_dir / "state_iter_000.json"

    user_args = [
        "--mode", mode,
        "--fcidump", str(fcidump_path),
        "--count_dict_file", str(count_dict_path),
        "--output_dir", str(init_dir),
        "--state_out", str(state_file),
    ]
    if initial_occupancies_file:
        initial_occ_path = Path(initial_occupancies_file).expanduser().resolve()
        if not initial_occ_path.exists():
            raise FileNotFoundError(f"Initial occupancies file not found: {initial_occ_path}")
        user_args.extend(["--initial_occupancies", str(initial_occ_path)])
    if verbose:
        user_args.append("-v")

    result = await run_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=init_dir,
        script_filename="init.pbs",
        user_args=user_args,
        watch_poll_interval=10.0,
        timeout_seconds=max_time * 2,
        metrics_artifact_key="gb-sqd-init-metrics",
    )
    if result.exit_status != 0:
        raise RuntimeError(f"gb-demo init failed: exit_status={result.exit_status}")
    if not state_file.exists():
        raise RuntimeError(f"State file was not created: {state_file}")

    # Prepare initialization data
    init_data = {
        "mode": mode,
        "fcidump_file": str(fcidump_path),
        "count_dict_file": str(count_dict_path),
        "work_dir": str(work_path),
        "init_dir": str(init_dir),
        "state_file": str(state_file),
        "job_result": {
            "job_id": getattr(result, "job_id", None),
            "exit_status": result.exit_status,
        },
        "parameters": kwargs,
    }
    
    # Save initialization data
    init_file = work_path / "init_data.json"
    with open(init_file, "w") as f:
        json.dump(init_data, f, indent=2)
    
    logger.info(f"✓ Initialization complete: {init_file}")
    
    return init_data
