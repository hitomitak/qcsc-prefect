"""GB SQD workflow tasks."""

__all__ = [
    "initialize_task",
    "recovery_iteration_task",
    "final_diagonalization_task",
    "output_results_task",
    "bulk_target_run_task",
]


def __getattr__(name: str):
    if name == "initialize_task":
        from .initialize import initialize_task

        return initialize_task
    if name == "recovery_iteration_task":
        from .recovery_iteration import recovery_iteration_task

        return recovery_iteration_task
    if name == "final_diagonalization_task":
        from .final_diagonalization import final_diagonalization_task

        return final_diagonalization_task
    if name == "output_results_task":
        from .output_results import output_results_task

        return output_results_task
    if name == "bulk_target_run_task":
        from .bulk_target_run import bulk_target_run_task

        return bulk_target_run_task
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
