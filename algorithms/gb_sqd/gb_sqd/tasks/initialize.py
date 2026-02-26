"""Initialization task for GB SQD workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prefect import get_run_logger, task


@task(
    name="initialize",
    retries=2,
    retry_delay_seconds=10,
    task_run_name="initialize",
)
def initialize_task(
    fcidump_file: str,
    count_dict_file: str,
    work_dir: str | Path,
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
    logger.info("Initializing GB-SQD workflow")
    
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
    
    # Prepare initialization data
    init_data = {
        "fcidump_file": str(fcidump_path),
        "count_dict_file": str(count_dict_path),
        "work_dir": str(work_path),
        "parameters": kwargs,
    }
    
    # Save initialization data
    init_file = work_path / "init_data.json"
    with open(init_file, "w") as f:
        json.dump(init_data, f, indent=2)
    
    logger.info(f"✓ Initialization complete: {init_file}")
    
    return init_data

