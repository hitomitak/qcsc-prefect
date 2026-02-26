"""Output results task for GB SQD workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prefect import get_run_logger, task
from prefect.artifacts import create_table_artifact


@task(
    name="output_results",
    retries=2,
    retry_delay_seconds=10,
    task_run_name="output_results",
)
def output_results_task(
    final_result: dict[str, Any],
    work_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Generate output files and telemetry data.
    
    This task creates the final energy_log.json file and saves
    telemetry data as Prefect artifacts.
    
    Args:
        final_result: Result from final_diagonalization_task
        work_dir: Working directory
        **kwargs: Additional parameters
    
    Returns:
        Dictionary containing output information
    """
    logger = get_run_logger()
    logger.info("Generating output files")
    
    work_path = Path(work_dir).expanduser().resolve()
    
    # Create energy_log.json
    energy_log = {
        "status": final_result.get("status", "unknown"),
        "num_iterations": final_result.get("num_iterations", 0),
        "energy_final": final_result.get("energy_final"),
        "energies": final_result.get("energies", []),
    }
    
    energy_log_file = work_path / "energy_log.json"
    with open(energy_log_file, "w") as f:
        json.dump(energy_log, f, indent=2)
    
    logger.info(f"✓ Energy log saved: {energy_log_file}")
    
    if energy_log["energy_final"] is not None:
        logger.info(f"Final energy: {energy_log['energy_final']}")
    
    # Create Prefect artifact with summary
    try:
        summary_table = []
        for energy_data in energy_log.get("energies", []):
            summary_table.append({
                "Iteration": energy_data.get("iteration", "N/A"),
                "Energy": f"{energy_data.get('energy', 'N/A'):.10f}" if isinstance(energy_data.get('energy'), (int, float)) else "N/A",
            })
        
        if summary_table:
            create_table_artifact(
                key="gb-sqd-energy-summary",
                table=summary_table,
                description="GB-SQD energy convergence summary",
            )
            logger.info("✓ Telemetry artifact created")
    except Exception as e:
        logger.warning(f"Failed to create artifact: {e}")
    
    # Create execution summary
    summary = {
        "status": "success",
        "work_dir": str(work_path),
        "energy_log_file": str(energy_log_file),
        "energy_final": energy_log["energy_final"],
        "num_iterations": energy_log["num_iterations"],
    }
    
    logger.info("✓ Output generation complete")
    
    return summary
