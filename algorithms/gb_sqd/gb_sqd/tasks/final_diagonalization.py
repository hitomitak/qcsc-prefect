"""Final diagonalization task for GB SQD workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prefect import get_run_logger, task


@task(
    name="final_diagonalization",
    retries=1,
    retry_delay_seconds=30,
    task_run_name="final_diagonalization",
)
def final_diagonalization_task(
    recovery_results: list[dict[str, Any]],
    work_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Perform final diagonalization on all recovery results.
    
    This task aggregates results from all recovery iterations and
    prepares the final output.
    
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
    
    # Collect all energy data
    energies = []
    for result in recovery_results:
        if result.get("energy_data"):
            energy = result["energy_data"].get("energy_final")
            if energy is not None:
                energies.append({
                    "iteration": result["iteration_id"],
                    "energy": energy,
                })
                logger.info(f"Iteration {result['iteration_id']}: E = {energy}")
    
    if not energies:
        logger.warning("No energy data found in recovery results")
        final_energy = None
    else:
        # Use the last iteration's energy as final
        final_energy = energies[-1]["energy"]
        logger.info(f"Final energy: {final_energy}")
    
    # Prepare final result
    final_result = {
        "status": "success",
        "num_iterations": len(recovery_results),
        "energies": energies,
        "energy_final": final_energy,
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

