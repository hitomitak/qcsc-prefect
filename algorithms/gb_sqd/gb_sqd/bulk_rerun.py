"""Helpers for rerunning failed GB-SQD bulk targets with override parameters."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bulk import bulk_gb_sqd_flow


def bulk_summary_path(output_root_dir: str | Path) -> Path:
    return Path(output_root_dir).expanduser().resolve() / "_bulk_summary" / "run_summary.json"


def load_bulk_summary(output_root_dir: str | Path) -> dict[str, Any]:
    summary_path = bulk_summary_path(output_root_dir)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Bulk summary not found: {summary_path}")
    return json.loads(summary_path.read_text())


def failed_relative_paths(summary: Mapping[str, Any]) -> list[str]:
    return [
        result["relative_path"]
        for result in summary.get("results", [])
        if result.get("status") == "failed" and result.get("relative_path")
    ]


def build_failed_target_overrides(
    *,
    summary: Mapping[str, Any],
    override_parameters: Mapping[str, Any],
    base_target_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    failed_paths = failed_relative_paths(summary)

    merged: dict[str, dict[str, Any]] = copy.deepcopy(base_target_overrides or {})
    for relative_path in failed_paths:
        target_override = dict(merged.get(relative_path, {}))
        target_override.update(dict(override_parameters))
        merged[relative_path] = target_override
    return merged


def bulk_gb_sqd_flow_with_failed_target_rerun_plan(
    *,
    failed_target_override_sequence: Sequence[Mapping[str, Any]],
    **bulk_flow_kwargs: Any,
) -> dict[str, Any]:
    """Run bulk GB-SQD once, then rerun failed targets with staged override parameters."""

    if not failed_target_override_sequence:
        raise ValueError("failed_target_override_sequence must contain at least one override mapping.")

    output_root_dir = bulk_flow_kwargs["output_root_dir"]
    accumulated_target_overrides = copy.deepcopy(bulk_flow_kwargs.get("target_overrides") or {})
    rerun_stages: list[dict[str, Any]] = []

    try:
        initial_summary = bulk_gb_sqd_flow(**bulk_flow_kwargs)
        return {
            "initial_run": initial_summary,
            "initial_error": None,
            "rerun_triggered": False,
            "rerun_stages": [],
            "final_run": initial_summary,
            "final_error": None,
            "completed_successfully": True,
            "failed_targets_remaining": [],
        }
    except Exception as exc:
        initial_error = str(exc)
        current_summary = load_bulk_summary(output_root_dir)

    initial_summary = current_summary
    final_error: str | None = initial_error

    for stage_index, override_parameters in enumerate(failed_target_override_sequence, start=1):
        target_overrides = build_failed_target_overrides(
            summary=current_summary,
            override_parameters=override_parameters,
            base_target_overrides=accumulated_target_overrides,
        )
        if not failed_relative_paths(current_summary):
            final_error = None
            break

        rerun_kwargs = dict(bulk_flow_kwargs)
        rerun_kwargs["skip_completed"] = True
        rerun_kwargs["target_overrides"] = target_overrides

        stage_result: dict[str, Any] = {
            "stage_index": stage_index,
            "override_parameters": dict(override_parameters),
            "target_overrides": target_overrides,
        }

        try:
            current_summary = bulk_gb_sqd_flow(**rerun_kwargs)
            stage_result["run"] = current_summary
            stage_result["error"] = None
            rerun_stages.append(stage_result)
            return {
                "initial_run": initial_summary,
                "initial_error": initial_error,
                "rerun_triggered": True,
                "rerun_stages": rerun_stages,
                "final_run": current_summary,
                "final_error": None,
                "completed_successfully": True,
                "failed_targets_remaining": [],
            }
        except Exception as exc:
            final_error = str(exc)
            current_summary = load_bulk_summary(output_root_dir)
            stage_result["run"] = current_summary
            stage_result["error"] = final_error
            rerun_stages.append(stage_result)
            accumulated_target_overrides = target_overrides

    return {
        "initial_run": initial_summary,
        "initial_error": initial_error,
        "rerun_triggered": bool(rerun_stages),
        "rerun_stages": rerun_stages,
        "final_run": current_summary,
        "final_error": final_error,
        "completed_successfully": not failed_relative_paths(current_summary),
        "failed_targets_remaining": failed_relative_paths(current_summary),
    }


def bulk_gb_sqd_flow_with_failed_target_rerun(
    *,
    failed_target_override_parameters: Mapping[str, Any],
    **bulk_flow_kwargs: Any,
) -> dict[str, Any]:
    """Run bulk GB-SQD once, then rerun only failed targets with override parameters."""

    result = bulk_gb_sqd_flow_with_failed_target_rerun_plan(
        failed_target_override_sequence=[failed_target_override_parameters],
        **bulk_flow_kwargs,
    )
    if not result["completed_successfully"] and result["final_error"] is not None:
        raise RuntimeError(result["final_error"])

    rerun_stages = result["rerun_stages"]
    if not rerun_stages:
        return {
            "initial_run": result["initial_run"],
            "initial_error": None,
            "rerun_triggered": False,
            "rerun_target_overrides": {},
            "rerun_run": None,
        }

    first_stage = rerun_stages[0]
    return {
        "initial_run": result["initial_run"],
        "initial_error": result["initial_error"],
        "rerun_triggered": True,
        "rerun_target_overrides": first_stage["target_overrides"],
        "rerun_run": first_stage["run"],
    }
