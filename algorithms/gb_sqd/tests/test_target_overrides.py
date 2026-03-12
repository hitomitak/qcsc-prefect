from __future__ import annotations

from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

from gb_sqd import bulk
from gb_sqd.target_overrides import (
    merge_target_job_parameters,
    normalize_relative_target_path,
    prepare_target_overrides,
)


def _write_case(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "count_dict.txt").write_text("count")
    (path / "fci_dump.txt").write_text("fcidump")


class _FakeFuture:
    def __init__(self, result):
        self._result = result
        self._final_state = object()

    def result(self):
        return self._result

    def add_done_callback(self, fn):
        fn(self)


class _FakeBulkTargetTask:
    def __init__(self, submitted: list[dict]):
        self.submitted = submitted

    def submit(self, **kwargs):
        self.submitted.append(kwargs)
        return _FakeFuture(
            {
                "status": "success",
                "relative_path": kwargs["relative_path"],
                "parameter_overrides": kwargs["parameter_overrides"],
            }
        )


def test_normalize_relative_target_path_converts_backslashes_and_dot_segments():
    assert normalize_relative_target_path(r"./13_18MO_Wat\atom_10129") == "13_18MO_Wat/atom_10129"


def test_prepare_target_overrides_normalizes_and_validates_known_targets():
    prepared = prepare_target_overrides(
        discovered_relative_paths=["13_18MO_Wat/atom_10129", "13_18MO_Wat/atom_10012"],
        target_overrides={
            r"./13_18MO_Wat\atom_10129": {
                "max_time": 600,
                "num_samples_per_batch": 500,
            }
        },
        allowed_parameter_names={"max_time", "num_samples_per_batch"},
    )

    assert prepared == {
        "13_18MO_Wat/atom_10129": {
            "max_time": 600,
            "num_samples_per_batch": 500,
        }
    }


def test_prepare_target_overrides_rejects_unknown_parameter_names():
    with pytest.raises(ValueError, match="unsupported GB-SQD parameter names"):
        prepare_target_overrides(
            discovered_relative_paths=["13_18MO_Wat/atom_10129"],
            target_overrides={"13_18MO_Wat/atom_10129": {"not_a_parameter": 1}},
            allowed_parameter_names={"max_time"},
        )


def test_prepare_target_overrides_rejects_unknown_target_path():
    with pytest.raises(ValueError, match="unknown target path"):
        prepare_target_overrides(
            discovered_relative_paths=["13_18MO_Wat/atom_10129"],
            target_overrides={"13_18MO_Wat/atom_99999": {"max_time": 600}},
            allowed_parameter_names={"max_time"},
        )


def test_merge_target_job_parameters_applies_only_matching_target_override():
    merged, applied = merge_target_job_parameters(
        base_job_parameters={"max_time": 300, "num_batches": 2},
        target_overrides={"13_18MO_Wat/atom_10129": {"max_time": 600}},
        relative_path="13_18MO_Wat/atom_10129",
    )

    assert merged == {"max_time": 600, "num_batches": 2}
    assert applied == {"max_time": 600}


def test_bulk_flow_passes_target_specific_parameter_overrides(tmp_path: Path, monkeypatch):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_case(input_root / "13_18MO_Wat" / "atom_10012")
    _write_case(input_root / "13_18MO_Wat" / "atom_10129")

    submitted: list[dict] = []
    monkeypatch.setattr(bulk, "_get_bulk_target_run_task", lambda: _FakeBulkTargetTask(submitted))

    with prefect_test_harness():
        summary = bulk.bulk_gb_sqd_flow(
            mode="ext_sqd",
            input_root_dir=str(input_root),
            output_root_dir=str(output_root),
            max_jobs_in_queue=1,
            max_prefect_concurrency=1,
            max_target_task_retries=0,
            num_batches=2,
            num_recovery=1,
            num_samples_per_batch=1000,
            max_time=300,
            target_overrides={
                "13_18MO_Wat/atom_10129": {
                    "max_time": 600,
                    "num_batches": 5,
                }
            },
        )

    assert summary["configured_target_overrides"] == {
        "13_18MO_Wat/atom_10129": {
            "max_time": 600,
            "num_batches": 5,
        }
    }

    submitted_by_target = {entry["relative_path"]: entry for entry in submitted}
    assert submitted_by_target["13_18MO_Wat/atom_10012"]["job_parameters"]["max_time"] == 300
    assert submitted_by_target["13_18MO_Wat/atom_10012"]["job_parameters"]["num_batches"] == 2
    assert submitted_by_target["13_18MO_Wat/atom_10012"]["parameter_overrides"] == {}

    assert submitted_by_target["13_18MO_Wat/atom_10129"]["job_parameters"]["max_time"] == 600
    assert submitted_by_target["13_18MO_Wat/atom_10129"]["job_parameters"]["num_batches"] == 5
    assert submitted_by_target["13_18MO_Wat/atom_10129"]["parameter_overrides"] == {
        "max_time": 600,
        "num_batches": 5,
    }


def test_bulk_flow_uses_miyabi_default_block_names_when_requested(tmp_path: Path, monkeypatch):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_case(input_root / "case_a" / "atom_0001")

    submitted: list[dict] = []
    monkeypatch.setattr(bulk, "_get_bulk_target_run_task", lambda: _FakeBulkTargetTask(submitted))

    with prefect_test_harness():
        summary = bulk.bulk_gb_sqd_flow(
            mode="ext_sqd",
            hpc_target="miyabi",
            input_root_dir=str(input_root),
            output_root_dir=str(output_root),
            max_jobs_in_queue=1,
            max_prefect_concurrency=1,
            max_target_task_retries=0,
            num_batches=2,
            num_recovery=1,
            num_samples_per_batch=1000,
            max_time=300,
        )

    assert summary["hpc_target"] == "miyabi"
    assert summary["execution_profile_block_name"] == "exec-gb-sqd-ext-miyabi"
    assert summary["hpc_profile_block_name"] == "hpc-miyabi-gb-sqd"
    assert submitted[0]["execution_profile_block_name"] == "exec-gb-sqd-ext-miyabi"
    assert submitted[0]["hpc_profile_block_name"] == "hpc-miyabi-gb-sqd"


def test_bulk_flow_refills_concurrency_when_one_future_completes(tmp_path: Path, monkeypatch):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_case(input_root / "case_a" / "atom_0001")
    _write_case(input_root / "case_b" / "atom_0002")
    _write_case(input_root / "case_c" / "atom_0003")

    event_log: list[str] = []
    submitted_futures: dict[str, _FakeFuture] = {}

    class _RollingTask:
        def submit(self, **kwargs):
            relative_path = kwargs["relative_path"]
            event_log.append(f"submit:{relative_path}")
            future = _FakeFuture(
                {
                    "status": "success",
                    "relative_path": relative_path,
                    "parameter_overrides": kwargs["parameter_overrides"],
                }
            )
            submitted_futures[relative_path] = future
            return future

    completion_order = [
        "case_a/atom_0001",
        "case_b/atom_0002",
        "case_c/atom_0003",
    ]

    def fake_as_completed(futures):
        for relative_path in completion_order:
            future = submitted_futures.get(relative_path)
            if future in futures:
                event_log.append(f"complete:{relative_path}")
                yield future
                return
        raise AssertionError("No matching future available for completion")

    monkeypatch.setattr(bulk, "_get_bulk_target_run_task", lambda: _RollingTask())
    monkeypatch.setattr(bulk, "as_completed", fake_as_completed)

    with prefect_test_harness():
        summary = bulk.bulk_gb_sqd_flow(
            mode="ext_sqd",
            input_root_dir=str(input_root),
            output_root_dir=str(output_root),
            max_jobs_in_queue=2,
            max_prefect_concurrency=2,
            max_target_task_retries=0,
            num_batches=2,
            num_recovery=1,
            num_samples_per_batch=1000,
            max_time=300,
        )

    assert summary["succeeded_targets"] == 3
    assert event_log == [
        "submit:case_a/atom_0001",
        "submit:case_b/atom_0002",
        "complete:case_a/atom_0001",
        "submit:case_c/atom_0003",
        "complete:case_b/atom_0002",
        "complete:case_c/atom_0003",
    ]
