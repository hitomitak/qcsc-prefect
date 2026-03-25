"""Helpers for building GB-SQD CLI arguments."""

from __future__ import annotations

from pathlib import Path


def _path_str(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def _append_optional_int(user_args: list[str], flag: str, value: int | None) -> None:
    if value is not None:
        user_args.extend([flag, str(value)])


def build_ext_sqd_user_args(
    *,
    fcidump_file: str | Path,
    count_dict_file: str | Path,
    output_dir: str | Path,
    num_recovery: int,
    num_batches: int,
    num_samples_per_batch: int,
    iteration: int,
    block: int,
    tolerance: float,
    max_time: float,
    adet_comm_size: int,
    bdet_comm_size: int,
    task_comm_size: int,
    adet_comm_size_final: int | None = None,
    bdet_comm_size_final: int | None = None,
    task_comm_size_final: int | None = None,
    do_carryover_in_recovery: bool = False,
    carryover_threshold: float = 1.0e-2,
    carryover_ratio: float = 0.5,
    with_hf: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Build CLI args for a monolithic ExtSQD run."""

    user_args = [
        "--fcidump", _path_str(fcidump_file),
        "--count_dict_file", _path_str(count_dict_file),
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
        "--output_dir", _path_str(output_dir),
    ]

    _append_optional_int(user_args, "--adet_comm_size_final", adet_comm_size_final)
    _append_optional_int(user_args, "--bdet_comm_size_final", bdet_comm_size_final)
    _append_optional_int(user_args, "--task_comm_size_final", task_comm_size_final)

    if do_carryover_in_recovery:
        user_args.append("--do_carryover_in_recovery")
    if with_hf:
        user_args.append("--with_hf")
    if verbose:
        user_args.append("-v")

    return user_args


def build_trim_sqd_user_args(
    *,
    fcidump_file: str | Path,
    count_dict_file: str | Path,
    output_dir: str | Path,
    num_recovery: int,
    num_batches: int,
    num_samples_per_recovery: int,
    iteration: int,
    block: int,
    tolerance: float,
    max_time: float,
    adet_comm_size: int,
    bdet_comm_size: int,
    task_comm_size: int,
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
) -> list[str]:
    """Build CLI args for a monolithic TrimSQD run."""

    user_args = [
        "--fcidump", _path_str(fcidump_file),
        "--count_dict_file", _path_str(count_dict_file),
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
        "--output_dir", _path_str(output_dir),
    ]

    _append_optional_int(user_args, "--adet_comm_size_combined", adet_comm_size_combined)
    _append_optional_int(user_args, "--bdet_comm_size_combined", bdet_comm_size_combined)
    _append_optional_int(user_args, "--task_comm_size_combined", task_comm_size_combined)
    _append_optional_int(user_args, "--adet_comm_size_final", adet_comm_size_final)
    _append_optional_int(user_args, "--bdet_comm_size_final", bdet_comm_size_final)
    _append_optional_int(user_args, "--task_comm_size_final", task_comm_size_final)

    if with_hf:
        user_args.append("--with_hf")
    if verbose:
        user_args.append("-v")

    return user_args
