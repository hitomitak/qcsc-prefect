"""Manual test runner for the GB-SQD bulk-mode Fugaku scenarios."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManualBulkScenario:
    name: str
    description: str
    mode: str
    input_subdir: str
    output_subdir: str
    expected_success: bool
    queue_limit_scope: str | None = None
    skip_completed: bool = True
    fail_fast: bool = False
    target_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManualBulkRunSettings:
    command_block_name_ext: str = "cmd-gb-sqd-ext"
    execution_profile_block_name_ext: str = "exec-gb-sqd-ext-fugaku"
    command_block_name_trim: str = "cmd-gb-sqd-trim"
    execution_profile_block_name_trim: str = "exec-gb-sqd-trim-fugaku"
    hpc_profile_block_name: str = "hpc-fugaku-gb-sqd"
    max_jobs_in_queue: int = 2
    max_prefect_concurrency: int = 2
    max_target_task_retries: int = 1
    queue_limit_scope: str = "user_queue"
    queue_poll_interval_seconds: float = 120.0
    job_name_prefix_base: str = "gbsqd-manual"
    num_recovery: int = 2
    num_batches: int = 2
    num_samples_per_batch: int = 1000
    num_samples_per_recovery: int = 1000
    iteration: int = 2
    block: int = 10
    tolerance: float = 1.0e-2
    max_time: float = 300.0
    adet_comm_size: int = 1
    bdet_comm_size: int = 1
    task_comm_size: int = 1
    adet_comm_size_combined: int = 2
    bdet_comm_size_combined: int = 1
    task_comm_size_combined: int = 1
    adet_comm_size_final: int = 2
    bdet_comm_size_final: int = 1
    task_comm_size_final: int = 1
    do_carryover_in_recovery: bool = True
    carryover_ratio: float = 0.5
    carryover_ratio_batch: float = 0.10
    carryover_ratio_combined: float = 0.50
    carryover_threshold: float = 1.0e-5
    with_hf: bool = True
    verbose: bool = True


MANUAL_BULK_SCENARIOS: dict[str, ManualBulkScenario] = {
    "scenario1_ext_happy": ManualBulkScenario(
        name="scenario1_ext_happy",
        description="ExtSQD happy path with two valid targets.",
        mode="ext_sqd",
        input_subdir="scenario1_ext_happy",
        output_subdir="scenario1_ext_happy",
        expected_success=True,
    ),
    "scenario2_trim_happy": ManualBulkScenario(
        name="scenario2_trim_happy",
        description="TrimSQD happy path with one valid target.",
        mode="trim_sqd",
        input_subdir="scenario2_trim_happy",
        output_subdir="scenario2_trim_happy",
        expected_success=True,
    ),
    "scenario3_retry_runtime": ManualBulkScenario(
        name="scenario3_retry_runtime",
        description="Runtime failure caused by a corrupted fci_dump.txt.",
        mode="ext_sqd",
        input_subdir="scenario3_retry_runtime",
        output_subdir="scenario3_retry_runtime",
        expected_success=False,
    ),
    "scenario4_override_rerun": ManualBulkScenario(
        name="scenario4_override_rerun",
        description="Rerun scenario 3 with target_overrides after repairing the input.",
        mode="ext_sqd",
        input_subdir="scenario3_retry_runtime",
        output_subdir="scenario3_retry_runtime",
        expected_success=True,
        target_overrides={
            "retry_bad_fcidump/atom_0003": {
                "max_time": 600.0,
                "num_samples_per_batch": 500,
            }
        },
        notes=("Repair retry_bad_fcidump/atom_0003 before running this scenario.",),
    ),
    "scenario5_skip_completed_rerun": ManualBulkScenario(
        name="scenario5_skip_completed_rerun",
        description="Rerun scenario 1 and confirm successful targets are skipped.",
        mode="ext_sqd",
        input_subdir="scenario1_ext_happy",
        output_subdir="scenario1_ext_happy",
        expected_success=True,
        notes=("Run scenario1_ext_happy successfully before this scenario.",),
    ),
    "scenario6_mixed_failure": ManualBulkScenario(
        name="scenario6_mixed_failure",
        description="Mixed success/failure batch with fail_fast disabled.",
        mode="ext_sqd",
        input_subdir="scenario6_mixed_failure",
        output_subdir="scenario6_mixed_failure",
        expected_success=False,
        fail_fast=False,
    ),
    "scenario7_user_queue": ManualBulkScenario(
        name="scenario7_user_queue",
        description="Queue throttling with queue_limit_scope='user_queue'.",
        mode="ext_sqd",
        input_subdir="scenario7_user_queue",
        output_subdir="scenario7_user_queue",
        expected_success=True,
        queue_limit_scope="user_queue",
    ),
    "scenario8_flow_jobs_only": ManualBulkScenario(
        name="scenario8_flow_jobs_only",
        description="Queue throttling with queue_limit_scope='flow_jobs_only'.",
        mode="ext_sqd",
        input_subdir="scenario8_flow_jobs_only",
        output_subdir="scenario8_flow_jobs_only",
        expected_success=True,
        queue_limit_scope="flow_jobs_only",
    ),
    "scenario9_invalid_override": ManualBulkScenario(
        name="scenario9_invalid_override",
        description="Invalid target_overrides should fail before any submit.",
        mode="ext_sqd",
        input_subdir="scenario9_invalid_override",
        output_subdir="scenario9_invalid_override",
        expected_success=False,
        target_overrides={"does/not/exist": {"max_time": 600.0}},
    ),
}


MANUAL_INPUT_LAYOUTS: dict[str, list[tuple[str, bool]]] = {
    "scenario1_ext_happy": [
        ("success_a/atom_0001", False),
        ("success_b/atom_0002", False),
    ],
    "scenario2_trim_happy": [
        ("trim_only/atom_0001", False),
    ],
    "scenario3_retry_runtime": [
        ("success_a/atom_0001", False),
        ("retry_bad_fcidump/atom_0003", True),
    ],
    "scenario6_mixed_failure": [
        ("success_a/atom_0001", False),
        ("mixed_bad/atom_0006", True),
    ],
    "scenario7_user_queue": [
        ("queue_user_a/atom_0007", False),
        ("queue_user_b/atom_0008", False),
    ],
    "scenario8_flow_jobs_only": [
        ("queue_flow_a/atom_0009", False),
        ("queue_flow_b/atom_0010", False),
    ],
    "scenario9_invalid_override": [
        ("success_a/atom_0011", False),
    ],
}


def _manifest_path(workspace_root: str | Path) -> Path:
    return Path(workspace_root).expanduser().resolve() / "manual_bulk_test_manifest.json"


def _inputs_root(workspace_root: str | Path) -> Path:
    return Path(workspace_root).expanduser().resolve() / "inputs"


def _outputs_root(workspace_root: str | Path) -> Path:
    return Path(workspace_root).expanduser().resolve() / "outputs"


def _resolve_seed_files(seed_dir: str | Path) -> tuple[Path, Path]:
    seed_path = Path(seed_dir).expanduser().resolve()
    count_dict_file = seed_path / "count_dict.txt"
    fcidump_file = seed_path / "fci_dump.txt"
    if not count_dict_file.is_file():
        raise FileNotFoundError(f"count_dict.txt not found under seed directory: {seed_path}")
    if not fcidump_file.is_file():
        raise FileNotFoundError(f"fci_dump.txt not found under seed directory: {seed_path}")
    return count_dict_file, fcidump_file


def _copy_seed_case(seed_dir: str | Path, target_dir: Path) -> None:
    count_dict_file, fcidump_file = _resolve_seed_files(seed_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(count_dict_file, target_dir / "count_dict.txt")
    shutil.copy2(fcidump_file, target_dir / "fci_dump.txt")


def _corrupt_fcidump(target_dir: Path) -> None:
    corrupted = """THIS IS AN INTENTIONALLY CORRUPTED FCIDUMP FILE
IT EXISTS TO EXERCISE BULK RETRY BEHAVIOR
"""
    (target_dir / "fci_dump.txt").write_text(corrupted)


def _scenario_job_name_prefix(base: str, scenario_name: str) -> str:
    normalized = scenario_name.replace("_", "-")
    return f"{base}-{normalized}"


def prepare_manual_bulk_workspace(
    *,
    seed_dir: str | Path,
    workspace_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    workspace_path = Path(workspace_root).expanduser().resolve()
    inputs_root = _inputs_root(workspace_path)
    outputs_root = _outputs_root(workspace_path)
    manifest_path = _manifest_path(workspace_path)

    if force and workspace_path.exists():
        shutil.rmtree(workspace_path)
    elif workspace_path.exists() and any(workspace_path.iterdir()):
        raise FileExistsError(
            f"Workspace root already exists and is not empty: {workspace_path}. "
            "Pass --force to recreate it."
        )

    inputs_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    for input_subdir, layout in MANUAL_INPUT_LAYOUTS.items():
        scenario_root = inputs_root / input_subdir
        for relative_target_path, corrupt_fcidump in layout:
            target_dir = scenario_root / relative_target_path
            _copy_seed_case(seed_dir, target_dir)
            if corrupt_fcidump:
                _corrupt_fcidump(target_dir)

    manifest = {
        "workspace_root": str(workspace_path),
        "seed_dir": str(Path(seed_dir).expanduser().resolve()),
        "inputs_root": str(inputs_root),
        "outputs_root": str(outputs_root),
        "scenarios": {
            scenario_name: describe_manual_bulk_scenario(workspace_path, scenario_name)
            for scenario_name in MANUAL_BULK_SCENARIOS
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def repair_manual_bulk_target(
    *,
    seed_dir: str | Path,
    workspace_root: str | Path,
    scenario_name: str,
    relative_target_path: str,
) -> Path:
    scenario = MANUAL_BULK_SCENARIOS[scenario_name]
    target_dir = _inputs_root(workspace_root) / scenario.input_subdir / Path(relative_target_path)
    _copy_seed_case(seed_dir, target_dir)
    return target_dir


def describe_manual_bulk_scenario(workspace_root: str | Path, scenario_name: str) -> dict[str, Any]:
    workspace_path = Path(workspace_root).expanduser().resolve()
    scenario = MANUAL_BULK_SCENARIOS[scenario_name]
    return {
        "name": scenario.name,
        "description": scenario.description,
        "mode": scenario.mode,
        "input_root_dir": str(_inputs_root(workspace_path) / scenario.input_subdir),
        "output_root_dir": str(_outputs_root(workspace_path) / scenario.output_subdir),
        "expected_success": scenario.expected_success,
        "queue_limit_scope": scenario.queue_limit_scope,
        "skip_completed": scenario.skip_completed,
        "fail_fast": scenario.fail_fast,
        "target_overrides": copy.deepcopy(scenario.target_overrides),
        "notes": list(scenario.notes),
    }


def build_manual_bulk_flow_kwargs(
    *,
    workspace_root: str | Path,
    scenario_name: str,
    settings: ManualBulkRunSettings,
) -> dict[str, Any]:
    scenario = MANUAL_BULK_SCENARIOS[scenario_name]
    description = describe_manual_bulk_scenario(workspace_root, scenario_name)

    flow_kwargs: dict[str, Any] = {
        "mode": scenario.mode,
        "input_root_dir": description["input_root_dir"],
        "output_root_dir": description["output_root_dir"],
        "hpc_profile_block_name": settings.hpc_profile_block_name,
        "max_jobs_in_queue": settings.max_jobs_in_queue,
        "max_prefect_concurrency": settings.max_prefect_concurrency,
        "max_target_task_retries": settings.max_target_task_retries,
        "queue_limit_scope": scenario.queue_limit_scope or settings.queue_limit_scope,
        "queue_poll_interval_seconds": settings.queue_poll_interval_seconds,
        "job_name_prefix": _scenario_job_name_prefix(settings.job_name_prefix_base, scenario.name),
        "skip_completed": scenario.skip_completed,
        "fail_fast": scenario.fail_fast,
        "target_overrides": copy.deepcopy(scenario.target_overrides),
    }

    if scenario.mode == "ext_sqd":
        flow_kwargs.update(
            {
                "command_block_name": settings.command_block_name_ext,
                "execution_profile_block_name": settings.execution_profile_block_name_ext,
                "num_recovery": settings.num_recovery,
                "num_batches": settings.num_batches,
                "num_samples_per_batch": settings.num_samples_per_batch,
                "iteration": settings.iteration,
                "block": settings.block,
                "tolerance": settings.tolerance,
                "max_time": settings.max_time,
                "adet_comm_size": settings.adet_comm_size,
                "bdet_comm_size": settings.bdet_comm_size,
                "task_comm_size": settings.task_comm_size,
                "adet_comm_size_final": settings.adet_comm_size_final,
                "bdet_comm_size_final": settings.bdet_comm_size_final,
                "task_comm_size_final": settings.task_comm_size_final,
                "do_carryover_in_recovery": settings.do_carryover_in_recovery,
                "carryover_ratio": settings.carryover_ratio,
                "carryover_threshold": settings.carryover_threshold,
                "with_hf": settings.with_hf,
                "verbose": settings.verbose,
            }
        )
    else:
        flow_kwargs.update(
            {
                "command_block_name": settings.command_block_name_trim,
                "execution_profile_block_name": settings.execution_profile_block_name_trim,
                "num_recovery": settings.num_recovery,
                "num_batches": settings.num_batches,
                "num_samples_per_recovery": settings.num_samples_per_recovery,
                "iteration": settings.iteration,
                "block": settings.block,
                "tolerance": settings.tolerance,
                "max_time": settings.max_time,
                "adet_comm_size": settings.adet_comm_size,
                "bdet_comm_size": settings.bdet_comm_size,
                "task_comm_size": settings.task_comm_size,
                "adet_comm_size_combined": settings.adet_comm_size_combined,
                "bdet_comm_size_combined": settings.bdet_comm_size_combined,
                "task_comm_size_combined": settings.task_comm_size_combined,
                "adet_comm_size_final": settings.adet_comm_size_final,
                "bdet_comm_size_final": settings.bdet_comm_size_final,
                "task_comm_size_final": settings.task_comm_size_final,
                "carryover_ratio_batch": settings.carryover_ratio_batch,
                "carryover_ratio_combined": settings.carryover_ratio_combined,
                "carryover_threshold": settings.carryover_threshold,
                "with_hf": settings.with_hf,
                "verbose": settings.verbose,
            }
        )

    return flow_kwargs


def run_manual_bulk_scenario(
    *,
    workspace_root: str | Path,
    scenario_name: str,
    settings: ManualBulkRunSettings,
    dry_run: bool = False,
) -> tuple[int, dict[str, Any] | None]:
    from .bulk import bulk_gb_sqd_flow

    scenario = MANUAL_BULK_SCENARIOS[scenario_name]
    flow_kwargs = build_manual_bulk_flow_kwargs(
        workspace_root=workspace_root,
        scenario_name=scenario_name,
        settings=settings,
    )

    if dry_run:
        return 0, flow_kwargs

    try:
        summary = bulk_gb_sqd_flow(**flow_kwargs)
    except Exception as exc:
        if scenario.expected_success:
            raise
        return 0, {
            "status": "expected_failure",
            "scenario": scenario_name,
            "error": str(exc),
            "flow_kwargs": flow_kwargs,
        }

    if not scenario.expected_success:
        return 2, {
            "status": "unexpected_success",
            "scenario": scenario_name,
            "summary": summary,
            "flow_kwargs": flow_kwargs,
        }
    return 0, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run the GB-SQD bulk-mode manual Fugaku test scenarios."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create a manual bulk-test workspace from a seed target.")
    prepare_parser.add_argument("--seed-dir", required=True, help="Directory containing valid count_dict.txt and fci_dump.txt")
    prepare_parser.add_argument("--workspace-root", required=True, help="Workspace root to create for manual tests")
    prepare_parser.add_argument("--force", action="store_true", help="Recreate the workspace root if it already exists")

    repair_parser = subparsers.add_parser("repair", help="Restore a target directory from the seed input.")
    repair_parser.add_argument("--seed-dir", required=True, help="Directory containing valid count_dict.txt and fci_dump.txt")
    repair_parser.add_argument("--workspace-root", required=True, help="Existing manual-test workspace root")
    repair_parser.add_argument("--scenario", choices=sorted(MANUAL_BULK_SCENARIOS), required=True)
    repair_parser.add_argument("--relative-target-path", required=True, help="Relative target path inside the scenario input root")

    describe_parser = subparsers.add_parser("describe", help="Show the resolved paths and notes for a scenario.")
    describe_parser.add_argument("--workspace-root", required=True, help="Existing manual-test workspace root")
    describe_parser.add_argument("--scenario", choices=sorted(MANUAL_BULK_SCENARIOS), required=True)

    list_parser = subparsers.add_parser("list", help="List the available manual test scenarios.")

    run_parser = subparsers.add_parser("run", help="Run one manual bulk-test scenario.")
    run_parser.add_argument("--workspace-root", required=True, help="Existing manual-test workspace root")
    run_parser.add_argument("--scenario", choices=sorted(MANUAL_BULK_SCENARIOS), required=True)
    run_parser.add_argument("--dry-run", action="store_true", help="Print the resolved bulk flow kwargs without running")
    _add_run_settings_arguments(run_parser)

    return parser


def _add_run_settings_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--command-block-name-ext", default=ManualBulkRunSettings.command_block_name_ext)
    parser.add_argument(
        "--execution-profile-block-name-ext",
        default=ManualBulkRunSettings.execution_profile_block_name_ext,
    )
    parser.add_argument("--command-block-name-trim", default=ManualBulkRunSettings.command_block_name_trim)
    parser.add_argument(
        "--execution-profile-block-name-trim",
        default=ManualBulkRunSettings.execution_profile_block_name_trim,
    )
    parser.add_argument("--hpc-profile-block-name", default=ManualBulkRunSettings.hpc_profile_block_name)
    parser.add_argument("--max-jobs-in-queue", type=int, default=ManualBulkRunSettings.max_jobs_in_queue)
    parser.add_argument(
        "--max-prefect-concurrency", type=int, default=ManualBulkRunSettings.max_prefect_concurrency
    )
    parser.add_argument(
        "--max-target-task-retries", type=int, default=ManualBulkRunSettings.max_target_task_retries
    )
    parser.add_argument("--queue-limit-scope", default=ManualBulkRunSettings.queue_limit_scope)
    parser.add_argument(
        "--queue-poll-interval-seconds",
        type=float,
        default=ManualBulkRunSettings.queue_poll_interval_seconds,
    )
    parser.add_argument("--job-name-prefix-base", default=ManualBulkRunSettings.job_name_prefix_base)
    parser.add_argument("--num-recovery", type=int, default=ManualBulkRunSettings.num_recovery)
    parser.add_argument("--num-batches", type=int, default=ManualBulkRunSettings.num_batches)
    parser.add_argument(
        "--num-samples-per-batch", type=int, default=ManualBulkRunSettings.num_samples_per_batch
    )
    parser.add_argument(
        "--num-samples-per-recovery",
        type=int,
        default=ManualBulkRunSettings.num_samples_per_recovery,
    )
    parser.add_argument("--iteration", type=int, default=ManualBulkRunSettings.iteration)
    parser.add_argument("--block", type=int, default=ManualBulkRunSettings.block)
    parser.add_argument("--tolerance", type=float, default=ManualBulkRunSettings.tolerance)
    parser.add_argument("--max-time", type=float, default=ManualBulkRunSettings.max_time)
    parser.add_argument("--adet-comm-size", type=int, default=ManualBulkRunSettings.adet_comm_size)
    parser.add_argument("--bdet-comm-size", type=int, default=ManualBulkRunSettings.bdet_comm_size)
    parser.add_argument("--task-comm-size", type=int, default=ManualBulkRunSettings.task_comm_size)
    parser.add_argument(
        "--adet-comm-size-combined",
        type=int,
        default=ManualBulkRunSettings.adet_comm_size_combined,
    )
    parser.add_argument(
        "--bdet-comm-size-combined",
        type=int,
        default=ManualBulkRunSettings.bdet_comm_size_combined,
    )
    parser.add_argument(
        "--task-comm-size-combined",
        type=int,
        default=ManualBulkRunSettings.task_comm_size_combined,
    )
    parser.add_argument("--adet-comm-size-final", type=int, default=ManualBulkRunSettings.adet_comm_size_final)
    parser.add_argument("--bdet-comm-size-final", type=int, default=ManualBulkRunSettings.bdet_comm_size_final)
    parser.add_argument("--task-comm-size-final", type=int, default=ManualBulkRunSettings.task_comm_size_final)
    parser.add_argument(
        "--do-carryover-in-recovery",
        action=argparse.BooleanOptionalAction,
        default=ManualBulkRunSettings.do_carryover_in_recovery,
    )
    parser.add_argument("--carryover-ratio", type=float, default=ManualBulkRunSettings.carryover_ratio)
    parser.add_argument(
        "--carryover-ratio-batch",
        type=float,
        default=ManualBulkRunSettings.carryover_ratio_batch,
    )
    parser.add_argument(
        "--carryover-ratio-combined",
        type=float,
        default=ManualBulkRunSettings.carryover_ratio_combined,
    )
    parser.add_argument(
        "--carryover-threshold",
        type=float,
        default=ManualBulkRunSettings.carryover_threshold,
    )
    parser.add_argument("--with-hf", action=argparse.BooleanOptionalAction, default=ManualBulkRunSettings.with_hf)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=ManualBulkRunSettings.verbose)


def _settings_from_namespace(args: argparse.Namespace) -> ManualBulkRunSettings:
    return ManualBulkRunSettings(
        command_block_name_ext=args.command_block_name_ext,
        execution_profile_block_name_ext=args.execution_profile_block_name_ext,
        command_block_name_trim=args.command_block_name_trim,
        execution_profile_block_name_trim=args.execution_profile_block_name_trim,
        hpc_profile_block_name=args.hpc_profile_block_name,
        max_jobs_in_queue=args.max_jobs_in_queue,
        max_prefect_concurrency=args.max_prefect_concurrency,
        max_target_task_retries=args.max_target_task_retries,
        queue_limit_scope=args.queue_limit_scope,
        queue_poll_interval_seconds=args.queue_poll_interval_seconds,
        job_name_prefix_base=args.job_name_prefix_base,
        num_recovery=args.num_recovery,
        num_batches=args.num_batches,
        num_samples_per_batch=args.num_samples_per_batch,
        num_samples_per_recovery=args.num_samples_per_recovery,
        iteration=args.iteration,
        block=args.block,
        tolerance=args.tolerance,
        max_time=args.max_time,
        adet_comm_size=args.adet_comm_size,
        bdet_comm_size=args.bdet_comm_size,
        task_comm_size=args.task_comm_size,
        adet_comm_size_combined=args.adet_comm_size_combined,
        bdet_comm_size_combined=args.bdet_comm_size_combined,
        task_comm_size_combined=args.task_comm_size_combined,
        adet_comm_size_final=args.adet_comm_size_final,
        bdet_comm_size_final=args.bdet_comm_size_final,
        task_comm_size_final=args.task_comm_size_final,
        do_carryover_in_recovery=args.do_carryover_in_recovery,
        carryover_ratio=args.carryover_ratio,
        carryover_ratio_batch=args.carryover_ratio_batch,
        carryover_ratio_combined=args.carryover_ratio_combined,
        carryover_threshold=args.carryover_threshold,
        with_hf=args.with_hf,
        verbose=args.verbose,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        manifest = prepare_manual_bulk_workspace(
            seed_dir=args.seed_dir,
            workspace_root=args.workspace_root,
            force=args.force,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "repair":
        repaired = repair_manual_bulk_target(
            seed_dir=args.seed_dir,
            workspace_root=args.workspace_root,
            scenario_name=args.scenario,
            relative_target_path=args.relative_target_path,
        )
        print(json.dumps({"repaired_target_dir": str(repaired)}, indent=2))
        return 0

    if args.command == "describe":
        print(json.dumps(describe_manual_bulk_scenario(args.workspace_root, args.scenario), indent=2))
        return 0

    if args.command == "list":
        listed = [asdict(scenario) for scenario in MANUAL_BULK_SCENARIOS.values()]
        print(json.dumps(listed, indent=2))
        return 0

    if args.command == "run":
        exit_code, payload = run_manual_bulk_scenario(
            workspace_root=args.workspace_root,
            scenario_name=args.scenario,
            settings=_settings_from_namespace(args),
            dry_run=args.dry_run,
        )
        if payload is not None:
            print(json.dumps(payload, indent=2))
        return exit_code

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
