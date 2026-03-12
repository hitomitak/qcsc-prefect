from __future__ import annotations

import json
from pathlib import Path

from gb_sqd import bulk_rerun


def test_build_failed_target_overrides_applies_override_only_to_failed_targets():
    summary = {
        "results": [
            {"relative_path": "case_a/atom_1", "status": "success"},
            {"relative_path": "case_b/atom_2", "status": "failed"},
            {"relative_path": "case_c/atom_3", "status": "failed"},
        ]
    }

    result = bulk_rerun.build_failed_target_overrides(
        summary=summary,
        override_parameters={"carryover_threshold": 1.0e-3},
        base_target_overrides={"case_c/atom_3": {"max_time": 1800}},
    )

    assert result == {
        "case_c/atom_3": {
            "max_time": 1800,
            "carryover_threshold": 1.0e-3,
        },
        "case_b/atom_2": {
            "carryover_threshold": 1.0e-3,
        },
    }


def test_bulk_rerun_wrapper_returns_without_rerun_when_initial_run_succeeds(monkeypatch):
    calls: list[dict] = []

    def fake_bulk_gb_sqd_flow(**kwargs):
        calls.append(kwargs)
        return {"failed_targets": 0, "results": []}

    monkeypatch.setattr(bulk_rerun, "bulk_gb_sqd_flow", fake_bulk_gb_sqd_flow)

    result = bulk_rerun.bulk_gb_sqd_flow_with_failed_target_rerun(
        mode="ext_sqd",
        input_root_dir="/tmp/input",
        output_root_dir="/tmp/output",
        failed_target_override_parameters={"carryover_threshold": 1.0e-3},
    )

    assert result["rerun_triggered"] is False
    assert result["rerun_run"] is None
    assert len(calls) == 1


def test_bulk_rerun_wrapper_reruns_only_failed_targets_with_override(tmp_path: Path, monkeypatch):
    output_root_dir = tmp_path / "result"
    summary_path = output_root_dir / "_bulk_summary" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    first_summary = {
        "failed_targets": 1,
        "results": [
            {"relative_path": "case_a/atom_1", "status": "success"},
            {"relative_path": "case_b/atom_2", "status": "failed"},
        ],
    }
    summary_path.write_text(json.dumps(first_summary))

    calls: list[dict] = []

    def fake_bulk_gb_sqd_flow(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("Bulk GB-SQD flow finished with 1 failed targets.")
        return {"failed_targets": 0, "results": [{"relative_path": "case_b/atom_2", "status": "success"}]}

    monkeypatch.setattr(bulk_rerun, "bulk_gb_sqd_flow", fake_bulk_gb_sqd_flow)

    result = bulk_rerun.bulk_gb_sqd_flow_with_failed_target_rerun(
        mode="ext_sqd",
        input_root_dir="/tmp/input",
        output_root_dir=str(output_root_dir),
        skip_completed=False,
        target_overrides={"case_a/atom_1": {"max_time": 900}},
        failed_target_override_parameters={"carryover_threshold": 1.0e-3},
    )

    assert result["rerun_triggered"] is True
    assert result["initial_error"] == "Bulk GB-SQD flow finished with 1 failed targets."
    assert result["rerun_target_overrides"] == {
        "case_a/atom_1": {"max_time": 900},
        "case_b/atom_2": {"carryover_threshold": 1.0e-3},
    }
    assert len(calls) == 2
    assert calls[1]["skip_completed"] is True
    assert calls[1]["target_overrides"] == {
        "case_a/atom_1": {"max_time": 900},
        "case_b/atom_2": {"carryover_threshold": 1.0e-3},
    }


def test_bulk_rerun_plan_applies_staged_overrides_until_success(tmp_path: Path, monkeypatch):
    output_root_dir = tmp_path / "result"
    summary_path = output_root_dir / "_bulk_summary" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    initial_summary = {
        "failed_targets": 2,
        "results": [
            {"relative_path": "case_a/atom_1", "status": "failed"},
            {"relative_path": "case_b/atom_2", "status": "failed"},
        ],
    }
    second_summary = {
        "failed_targets": 1,
        "results": [
            {"relative_path": "case_a/atom_1", "status": "success"},
            {"relative_path": "case_b/atom_2", "status": "failed"},
        ],
    }
    final_summary = {
        "failed_targets": 0,
        "results": [
            {"relative_path": "case_a/atom_1", "status": "success"},
            {"relative_path": "case_b/atom_2", "status": "success"},
        ],
    }
    summary_path.write_text(json.dumps(initial_summary))

    calls: list[dict] = []

    def fake_bulk_gb_sqd_flow(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            summary_path.write_text(json.dumps(initial_summary))
            raise RuntimeError("initial run failed")
        if len(calls) == 2:
            summary_path.write_text(json.dumps(second_summary))
            raise RuntimeError("stage 1 rerun failed")
        return final_summary

    monkeypatch.setattr(bulk_rerun, "bulk_gb_sqd_flow", fake_bulk_gb_sqd_flow)

    result = bulk_rerun.bulk_gb_sqd_flow_with_failed_target_rerun_plan(
        mode="ext_sqd",
        input_root_dir="/tmp/input",
        output_root_dir=str(output_root_dir),
        target_overrides={"seed/atom_0": {"max_time": 900}},
        failed_target_override_sequence=[
            {"carryover_threshold": 1.0e-3},
            {"carryover_threshold": 1.0e-2, "max_time": 1800},
        ],
    )

    assert result["completed_successfully"] is True
    assert result["final_run"] == final_summary
    assert result["initial_error"] == "initial run failed"
    assert len(result["rerun_stages"]) == 2
    assert result["rerun_stages"][0]["error"] == "stage 1 rerun failed"
    assert result["rerun_stages"][0]["target_overrides"] == {
        "seed/atom_0": {"max_time": 900},
        "case_a/atom_1": {"carryover_threshold": 1.0e-3},
        "case_b/atom_2": {"carryover_threshold": 1.0e-3},
    }
    assert result["rerun_stages"][1]["target_overrides"] == {
        "seed/atom_0": {"max_time": 900},
        "case_a/atom_1": {"carryover_threshold": 1.0e-3},
        "case_b/atom_2": {"carryover_threshold": 1.0e-2, "max_time": 1800},
    }
    assert calls[1]["skip_completed"] is True
    assert calls[2]["skip_completed"] is True


def test_bulk_rerun_plan_reports_failure_when_stages_are_exhausted(tmp_path: Path, monkeypatch):
    output_root_dir = tmp_path / "result"
    summary_path = output_root_dir / "_bulk_summary" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    failed_summary = {
        "failed_targets": 1,
        "results": [
            {"relative_path": "case_b/atom_2", "status": "failed"},
        ],
    }
    summary_path.write_text(json.dumps(failed_summary))

    calls: list[dict] = []

    def fake_bulk_gb_sqd_flow(**kwargs):
        calls.append(kwargs)
        summary_path.write_text(json.dumps(failed_summary))
        raise RuntimeError(f"failed call {len(calls)}")

    monkeypatch.setattr(bulk_rerun, "bulk_gb_sqd_flow", fake_bulk_gb_sqd_flow)

    result = bulk_rerun.bulk_gb_sqd_flow_with_failed_target_rerun_plan(
        mode="ext_sqd",
        input_root_dir="/tmp/input",
        output_root_dir=str(output_root_dir),
        failed_target_override_sequence=[
            {"carryover_threshold": 1.0e-3},
            {"carryover_threshold": 1.0e-2},
        ],
    )

    assert result["completed_successfully"] is False
    assert result["final_error"] == "failed call 3"
    assert result["failed_targets_remaining"] == ["case_b/atom_2"]
    assert len(result["rerun_stages"]) == 2
