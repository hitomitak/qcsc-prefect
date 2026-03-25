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
    
    # Prefer energy_log.json produced by `gb-demo finalize`.
    energy_log_file_value = final_result.get("energy_log_file")
    energy_log_file = (
        Path(energy_log_file_value).expanduser().resolve()
        if energy_log_file_value
        else None
    )
    if energy_log_file is not None and energy_log_file.is_file():
        energy_log = json.loads(energy_log_file.read_text())
        logger.info(f"Using generated energy log: {energy_log_file}")
    else:
        # Fallback for older paths that still aggregate in Python.
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
        raw_energies = energy_log.get("energies")
        if raw_energies is None:
            raw_energies = [
                {"iteration": i, "energy": e}
                for i, e in enumerate(energy_log.get("energy_history", []))
            ]
        for energy_data in raw_energies:
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
        "energy_final": energy_log.get("energy_final"),
        "num_iterations": energy_log.get("num_iterations", energy_log.get("total_iterations", 0)),
    }
    
    logger.info("✓ Output generation complete")
    
    return summary
