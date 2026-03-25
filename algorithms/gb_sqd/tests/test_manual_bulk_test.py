from __future__ import annotations

import json
from pathlib import Path

from gb_sqd.manual_bulk_test import (
    ManualBulkRunSettings,
    build_manual_bulk_flow_kwargs,
    describe_manual_bulk_scenario,
    prepare_manual_bulk_workspace,
    repair_manual_bulk_target,
    run_manual_bulk_scenario,
)


def _write_seed_case(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "count_dict.txt").write_text("seed-count")
    (path / "fci_dump.txt").write_text("seed-fcidump")


def test_prepare_manual_bulk_workspace_creates_manifest_and_corrupted_retry_case(tmp_path: Path):
    seed_dir = tmp_path / "seed"
    workspace_root = tmp_path / "workspace"
    _write_seed_case(seed_dir)

    manifest = prepare_manual_bulk_workspace(seed_dir=seed_dir, workspace_root=workspace_root)

    manifest_path = workspace_root / "manual_bulk_test_manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text())["workspace_root"] == str(workspace_root.resolve())
    assert manifest["scenarios"]["scenario4_override_rerun"]["input_root_dir"] == manifest["scenarios"][
        "scenario3_retry_runtime"
    ]["input_root_dir"]

    success_fcidump = workspace_root / "inputs" / "scenario1_ext_happy" / "success_a" / "atom_0001" / "fci_dump.txt"
    retry_fcidump = (
        workspace_root / "inputs" / "scenario3_retry_runtime" / "retry_bad_fcidump" / "atom_0003" / "fci_dump.txt"
    )
    assert success_fcidump.read_text() == "seed-fcidump"
    assert "INTENTIONALLY CORRUPTED" in retry_fcidump.read_text()


def test_build_manual_bulk_flow_kwargs_uses_shared_rerun_paths_and_overrides(tmp_path: Path):
    seed_dir = tmp_path / "seed"
    workspace_root = tmp_path / "workspace"
    _write_seed_case(seed_dir)
    prepare_manual_bulk_workspace(seed_dir=seed_dir, workspace_root=workspace_root)

    settings = ManualBulkRunSettings(job_name_prefix_base="manual-bulk")
    flow_kwargs = build_manual_bulk_flow_kwargs(
        workspace_root=workspace_root,
        scenario_name="scenario4_override_rerun",
        settings=settings,
    )

    assert flow_kwargs["mode"] == "ext_sqd"
    assert flow_kwargs["input_root_dir"].endswith("/inputs/scenario3_retry_runtime")
    assert flow_kwargs["output_root_dir"].endswith("/outputs/scenario3_retry_runtime")
    assert flow_kwargs["target_overrides"] == {
        "retry_bad_fcidump/atom_0003": {
            "max_time": 600.0,
            "num_samples_per_batch": 500,
        }
    }
    assert flow_kwargs["job_name_prefix"] == "manual-bulk-scenario4-override-rerun"

    flow_jobs_only_kwargs = build_manual_bulk_flow_kwargs(
        workspace_root=workspace_root,
        scenario_name="scenario8_flow_jobs_only",
        settings=settings,
    )
    assert flow_jobs_only_kwargs["queue_limit_scope"] == "flow_jobs_only"


def test_repair_manual_bulk_target_restores_seed_files_and_dry_run_returns_kwargs(tmp_path: Path):
    seed_dir = tmp_path / "seed"
    workspace_root = tmp_path / "workspace"
    _write_seed_case(seed_dir)
    prepare_manual_bulk_workspace(seed_dir=seed_dir, workspace_root=workspace_root)

    target_dir = repair_manual_bulk_target(
        seed_dir=seed_dir,
        workspace_root=workspace_root,
        scenario_name="scenario3_retry_runtime",
        relative_target_path="retry_bad_fcidump/atom_0003",
    )
    assert (target_dir / "fci_dump.txt").read_text() == "seed-fcidump"

    exit_code, payload = run_manual_bulk_scenario(
        workspace_root=workspace_root,
        scenario_name="scenario2_trim_happy",
        settings=ManualBulkRunSettings(),
        dry_run=True,
    )
    assert exit_code == 0
    assert payload is not None
    assert payload["mode"] == "trim_sqd"
    assert payload["input_root_dir"] == describe_manual_bulk_scenario(
        workspace_root, "scenario2_trim_happy"
    )["input_root_dir"]
