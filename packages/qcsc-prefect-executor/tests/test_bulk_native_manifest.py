from __future__ import annotations

import json
from pathlib import Path

import pytest
from qcsc_prefect_executor.bulk.models import BulkJobSpec
from qcsc_prefect_executor.bulk.native_manifest import (
    create_native_bulk_group_manifests,
)
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry


def _records_for_manifest_test(tmp_path: Path):
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")
    specs = [
        BulkJobSpec(
            job_key="qpy-job",
            stage_id="stage-qpy",
            wave_id="wave-0",
            target_id="target-qpy",
            work_dir=tmp_path / "work" / "qpy-job",
            command_args={
                "kind": "qpy",
                "qpy_path": "inputs/circuit.qpy",
                "shots": 1024,
            },
            expected_outputs=[Path("counts.json"), Path("done.marker")],
        ),
        BulkJobSpec(
            job_key="trim-job",
            stage_id="stage-trim",
            wave_id="wave-0",
            target_id="target-trim",
            work_dir=tmp_path / "work" / "trim-job",
            command_args={
                "kind": "trim_sqd",
                "input_manifest": "trim/input.json",
                "rank": 2,
            },
            expected_outputs=[Path("trimmed.json")],
        ),
        BulkJobSpec(
            job_key="generic-job",
            stage_id="stage-generic",
            wave_id="wave-1",
            target_id=None,
            work_dir=tmp_path / "work" / "generic-job",
            command_args={
                "argv": ["--alpha", "1"],
                "metadata": {"source": "unit-test"},
            },
            expected_outputs=[],
        ),
    ]
    registry.upsert_jobs(specs)
    return registry.get_submit_candidates_fifo(limit=10)


def test_create_native_bulk_group_manifests_writes_indexed_json_files(
    tmp_path: Path,
):
    records = _records_for_manifest_test(tmp_path)

    group = create_native_bulk_group_manifests(
        bulk_group_dir=tmp_path / "bulk-group-0001",
        jobs=records,
    )

    assert group.bulk_count == 3
    assert group.manifest_dir == tmp_path / "bulk-group-0001" / "manifests"
    assert [path.name for path in group.manifest_paths] == [
        "0.json",
        "1.json",
        "2.json",
    ]

    qpy_manifest = json.loads((group.manifest_dir / "0.json").read_text())
    assert qpy_manifest == {
        "job_key": "qpy-job",
        "stage_id": "stage-qpy",
        "wave_id": "wave-0",
        "target_id": "target-qpy",
        "work_dir": str(tmp_path / "work" / "qpy-job"),
        "command_args": {
            "kind": "qpy",
            "qpy_path": "inputs/circuit.qpy",
            "shots": 1024,
        },
        "expected_outputs": ["counts.json", "done.marker"],
    }

    trim_manifest = json.loads((group.manifest_dir / "1.json").read_text())
    assert trim_manifest["command_args"] == {
        "kind": "trim_sqd",
        "input_manifest": "trim/input.json",
        "rank": 2,
    }
    assert trim_manifest["expected_outputs"] == ["trimmed.json"]

    generic_manifest = json.loads((group.manifest_dir / "2.json").read_text())
    assert generic_manifest["command_args"] == {
        "argv": ["--alpha", "1"],
        "metadata": {"source": "unit-test"},
    }
    assert generic_manifest["expected_outputs"] == []


def test_create_native_bulk_group_manifests_requires_stage_id(tmp_path: Path):
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")
    registry.upsert_jobs(
        [
            BulkJobSpec(
                job_key="missing-stage",
                work_dir=tmp_path / "work" / "missing-stage",
            )
        ]
    )
    records = registry.get_submit_candidates_fifo(limit=10)

    with pytest.raises(ValueError, match="requires stage_id"):
        create_native_bulk_group_manifests(
            bulk_group_dir=tmp_path / "bulk-group-0001",
            jobs=records,
        )
