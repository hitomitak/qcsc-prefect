from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from qcsc_prefect.integrations.qiskit import artifacts as artifacts_mod
from qcsc_prefect.integrations.qiskit import result_summaries as summaries_mod
from qcsc_prefect.integrations.qiskit import serializers as serializers_mod
from qcsc_prefect.integrations.qiskit.result_summaries import (
    extract_estimator_result_summary,
    extract_sampler_result_summary,
)
from qcsc_prefect.integrations.qiskit.serializers import (
    load_json,
    make_json_serializable,
    save_estimator_result_summary,
    save_execution_metadata,
    save_json,
    save_sampler_result_summary,
    serialize_execution_metadata,
)


@dataclass
class _DataclassValue:
    timestamp: datetime
    labels: set[str]


class _BitArray:
    def get_counts(self):
        return {"00": 7, "11": 3, "10": 1}


class _SamplerData:
    meas = _BitArray()


class _SamplerPubResult:
    data = _SamplerData()


class _Array:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class _EstimatorData:
    evs = _Array([0.75])
    stds = _Array([0.125])
    ensemble_standard_error = _Array([0.0625])


class _EstimatorPubResult:
    data = _EstimatorData()
    metadata = {"shots": 128, "target_precision": 0.1}


class _Metadata:
    def model_dump(self, **_kwargs):
        return {
            "resource": "ibm_kawasaki",
            "created": datetime(2026, 5, 8, 9, 30, 0),
            "nested": {"date": date(2026, 5, 8)},
        }


class _UnknownResult:
    def __str__(self):
        return "unknown-result"


def test_make_json_serializable_handles_nested_values():
    value = {
        "when": datetime(2026, 5, 8, 9, 30, 0),
        "items": [_DataclassValue(timestamp=datetime(2026, 5, 8), labels={"b", "a"})],
        "date": date(2026, 5, 8),
    }

    assert make_json_serializable(value) == {
        "when": "2026-05-08T09:30:00",
        "items": [{"timestamp": "2026-05-08T00:00:00", "labels": ["a", "b"]}],
        "date": "2026-05-08",
    }


def test_save_json_and_load_json_round_trip(tmp_path):
    path = save_json({"b": 2, "a": [1]}, tmp_path / "nested" / "data.json")

    assert path == tmp_path / "nested" / "data.json"
    assert load_json(path) == {"a": [1], "b": 2}


def test_serialize_execution_metadata_works_with_mocked_metadata():
    serialized = serialize_execution_metadata(_Metadata())

    assert serialized["resource"] == "ibm_kawasaki"
    assert serialized["created"] == "2026-05-08T09:30:00"
    assert serialized["nested"]["date"] == "2026-05-08"


def test_save_execution_metadata(tmp_path):
    path = save_execution_metadata(_Metadata(), tmp_path / "metadata.json")

    assert load_json(path)["resource"] == "ibm_kawasaki"


def test_sampler_summary_works_with_mocked_sampler_result():
    summary = extract_sampler_result_summary([_SamplerPubResult()], max_counts=2)

    assert summary["result_type"] == "list"
    assert summary["num_pubs"] == 1
    assert summary["truncated"] is True
    assert summary["counts"] == [
        {
            "pub_index": 0,
            "register": "meas",
            "counts": {"00": 7, "11": 3},
            "truncated": True,
        }
    ]


def test_estimator_summary_works_with_mocked_estimator_result():
    summary = extract_estimator_result_summary([_EstimatorPubResult()])

    assert summary["result_type"] == "list"
    assert summary["num_pubs"] == 1
    assert summary["values"] == [
        {
            "pub_index": 0,
            "evs": [0.75],
            "stds": [0.125],
            "ensemble_standard_error": [0.0625],
            "shots": 128,
            "target_precision": 0.1,
        }
    ]


def test_unknown_result_object_returns_minimal_summary():
    sampler_summary = extract_sampler_result_summary(_UnknownResult())
    estimator_summary = extract_estimator_result_summary(_UnknownResult())

    assert sampler_summary["result_type"] == "_UnknownResult"
    assert sampler_summary["num_pubs"] == 0
    assert sampler_summary["counts"] == []
    assert estimator_summary["result_type"] == "_UnknownResult"
    assert estimator_summary["num_pubs"] == 0
    assert estimator_summary["values"] == []


def test_save_sampler_and_estimator_summaries(tmp_path):
    sampler_path = save_sampler_result_summary(
        [_SamplerPubResult()],
        tmp_path / "sampler_summary.json",
        max_counts=1,
    )
    estimator_path = save_estimator_result_summary(
        [_EstimatorPubResult()],
        tmp_path / "estimator_summary.json",
    )

    assert load_json(sampler_path)["counts"][0]["counts"] == {"00": 7}
    assert load_json(estimator_path)["values"][0]["evs"] == [0.75]


def test_artifacts_and_serializers_use_shared_summary_extraction():
    assert (
        artifacts_mod.extract_sampler_result_summary
        is summaries_mod.extract_sampler_result_summary
    )
    assert (
        artifacts_mod.extract_estimator_result_summary
        is summaries_mod.extract_estimator_result_summary
    )
    assert (
        artifacts_mod.collect_sampler_result_counts
        is summaries_mod.collect_sampler_result_counts
    )
    assert (
        artifacts_mod.collect_estimator_result_values
        is summaries_mod.collect_estimator_result_values
    )
    assert (
        serializers_mod.extract_sampler_result_summary
        is summaries_mod.extract_sampler_result_summary
    )
    assert (
        serializers_mod.extract_estimator_result_summary
        is summaries_mod.extract_estimator_result_summary
    )
