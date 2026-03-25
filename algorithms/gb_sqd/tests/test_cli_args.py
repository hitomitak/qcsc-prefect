from __future__ import annotations

from pathlib import Path

from gb_sqd.cli_args import build_ext_sqd_user_args, build_trim_sqd_user_args


def test_build_ext_sqd_user_args_includes_final_comm_sizes_and_recovery_flag(tmp_path: Path):
    args = build_ext_sqd_user_args(
        fcidump_file=tmp_path / "fci_dump.txt",
        count_dict_file=tmp_path / "count_dict.txt",
        output_dir=tmp_path / "out",
        num_recovery=2,
        num_batches=3,
        num_samples_per_batch=1000,
        iteration=2,
        block=10,
        tolerance=1.0e-4,
        max_time=300.0,
        adet_comm_size=1,
        bdet_comm_size=1,
        task_comm_size=1,
        adet_comm_size_final=2,
        bdet_comm_size_final=1,
        task_comm_size_final=1,
        do_carryover_in_recovery=True,
        carryover_threshold=1.0e-5,
        carryover_ratio=0.5,
        with_hf=True,
        verbose=True,
    )

    assert "--adet_comm_size_final" in args
    assert "--bdet_comm_size_final" in args
    assert "--task_comm_size_final" in args
    assert "--do_carryover_in_recovery" in args
    assert "--with_hf" in args
    assert "-v" in args


def test_build_trim_sqd_user_args_includes_combined_and_final_comm_sizes(tmp_path: Path):
    args = build_trim_sqd_user_args(
        fcidump_file=tmp_path / "fci_dump.txt",
        count_dict_file=tmp_path / "count_dict.txt",
        output_dir=tmp_path / "out",
        num_recovery=2,
        num_batches=2,
        num_samples_per_recovery=1000,
        iteration=2,
        block=10,
        tolerance=1.0e-4,
        max_time=300.0,
        adet_comm_size=1,
        bdet_comm_size=1,
        task_comm_size=1,
        adet_comm_size_combined=2,
        bdet_comm_size_combined=1,
        task_comm_size_combined=1,
        adet_comm_size_final=2,
        bdet_comm_size_final=1,
        task_comm_size_final=1,
        carryover_ratio_batch=0.1,
        carryover_ratio_combined=0.5,
        carryover_threshold=1.0e-5,
        with_hf=True,
        verbose=True,
    )

    assert "--adet_comm_size_combined" in args
    assert "--bdet_comm_size_combined" in args
    assert "--task_comm_size_combined" in args
    assert "--adet_comm_size_final" in args
    assert "--bdet_comm_size_final" in args
    assert "--task_comm_size_final" in args
    assert "--with_hf" in args
    assert "-v" in args
