from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from qcsc_prefect.integrations.qiskit import metadata as metadata_mod
from qcsc_prefect.integrations.qiskit.metadata import (
    collect_qiskit_execution_metadata,
    flatten_qiskit_execution_metadata,
)


class _Circuit:
    def __init__(self, *, depth: int, size: int) -> None:
        self._depth = depth
        self._size = size

    def depth(self) -> int:
        return self._depth

    def size(self) -> int:
        return self._size


class _Job:
    def __init__(self, *, usage_estimation=None, metrics_usage=None) -> None:
        self.tags = ["chemistry", "test"]
        self.primitive_id = "sampler"
        self._usage_estimation = usage_estimation
        self._metrics_usage = (
            metrics_usage if metrics_usage is not None else {"quantum_seconds": 3.0}
        )

    def job_id(self) -> str:
        return "job-123"

    def backend(self):
        return SimpleNamespace(name="ibm_kawasaki")

    def metrics(self):
        return {
            "timestamps": {
                "created": "2026-04-27T00:00:00+00:00",
                "running": "2026-04-27T00:00:10+00:00",
                "finished": "2026-04-27T00:00:40+00:00",
            },
            "usage": self._metrics_usage,
        }

    @property
    def usage_estimation(self):
        return self._usage_estimation


class _MinimalJob:
    def metrics(self):
        raise RuntimeError("metrics unavailable")


class _BitArray:
    shape = ()


class _DataBin:
    meas = _BitArray()


class _SamplerPubResult:
    data = _DataBin()
    metadata = {"circuit_metadata": {}}


class _DoubleSliceSpan:
    start = "2026-04-28 04:20:15"
    stop = "2026-04-28 04:20:16"
    size = 100


class _ExecutionSpans(list):
    pass


class _PrimitiveResult(list):
    def __init__(self, pub_results, metadata) -> None:
        super().__init__(pub_results)
        self.metadata = metadata


def test_collects_job_level_metadata_from_native_job_shape():
    metadata = collect_qiskit_execution_metadata(
        job=_Job(),
        pubs=[(_Circuit(depth=12, size=34),)],
        result=[
            SimpleNamespace(
                metadata={
                    "shape": [2, 3],
                    "timestamps": {
                        "started": "2026-04-27T00:00:11+00:00",
                        "completed": "2026-04-27T00:00:31+00:00",
                    },
                }
            )
        ],
        options={"params": {"shots": 4096}},
    )

    assert metadata.resource == "ibm_kawasaki"
    assert metadata.program_type == "sampler"
    assert metadata.num_pubs == 1
    assert metadata.job_id == "job-123"
    assert metadata.tags == ["chemistry", "test"]
    assert metadata.options.params["shots"] == 4096


def test_collects_circuit_depth_size_and_pub_timing():
    metadata = collect_qiskit_execution_metadata(
        job=_Job(),
        pubs=[{"circuit": _Circuit(depth=5, size=8)}],
        result=[
            {
                "metadata": {
                    "shape": (1,),
                    "timestamp": {
                        "started": "2026-04-27T00:00:12+00:00",
                        "completed": "2026-04-27T00:00:15+00:00",
                    },
                }
            }
        ],
    )

    pub = metadata.pubs[0]
    assert pub.circuit.depth == 5
    assert pub.circuit.size == 8
    assert pub.shape == (1,)
    assert pub.duration == 3.0


def test_collects_circuit_metadata_when_pub_is_bare_circuit():
    metadata = collect_qiskit_execution_metadata(
        job=_Job(),
        pubs=[_Circuit(depth=9, size=13)],
    )

    pub = metadata.pubs[0]
    assert pub.circuit.depth == 9
    assert pub.circuit.size == 13


def test_collects_pub_shape_and_timing_from_primitive_result_execution_spans():
    result = _PrimitiveResult(
        [_SamplerPubResult()],
        metadata={
            "execution": {
                "execution_spans": _ExecutionSpans([_DoubleSliceSpan()]),
            },
            "version": 2,
        },
    )

    metadata = collect_qiskit_execution_metadata(
        job=_Job(),
        pubs=[_Circuit(depth=9, size=13)],
        result=result,
    )
    flattened = flatten_qiskit_execution_metadata(metadata)

    assert metadata.pubs[0].shape == ()
    assert metadata.pubs[0].timestamp.started == "2026-04-28 04:20:15"
    assert metadata.pubs[0].timestamp.completed == "2026-04-28 04:20:16"
    assert metadata.pubs[0].duration == 1.0
    assert flattened["pub[0].shape"] == []
    assert flattened["pub[0].timestamp.started"] == "2026-04-28 04:20:15"
    assert flattened["pub[0].timestamp.completed"] == "2026-04-28 04:20:16"
    assert flattened["pub[0].duration"] == 1.0


def test_omits_pub_timing_when_result_has_no_pub_timing():
    result = _PrimitiveResult(
        [SimpleNamespace(metadata={"shots": 128, "target_precision": 0.1})],
        metadata={"version": 2},
    )

    metadata = collect_qiskit_execution_metadata(
        job=_Job(),
        pubs=[_Circuit(depth=9, size=13)],
        result=result,
    )
    flattened = flatten_qiskit_execution_metadata(metadata)

    assert metadata.pubs[0].timestamp.started is None
    assert metadata.pubs[0].timestamp.completed is None
    assert metadata.pubs[0].duration is None
    assert "pub[0].timestamp.started" not in flattened
    assert "pub[0].timestamp.completed" not in flattened
    assert "pub[0].duration" not in flattened


def test_calculates_timing_spans_from_job_metrics():
    metadata = collect_qiskit_execution_metadata(job=_Job())

    assert metadata.span.queue == 10.0
    assert metadata.span.work == 30.0
    assert metadata.span.qpu == 3.0
    assert metadata.work_efficiency == 0.1


def test_extracts_qpu_span_from_usage_estimation_when_metrics_usage_missing():
    metadata = collect_qiskit_execution_metadata(
        job=_Job(
            metrics_usage={},
            usage_estimation={"quantum_seconds": 7.5},
        )
    )

    assert metadata.span.qpu == 7.5


def test_flattening_uses_expected_target_keys():
    created = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    metadata = collect_qiskit_execution_metadata(
        job=_Job(),
        pubs=[SimpleNamespace(circuit=_Circuit(depth=1, size=2), shape=[4])],
        result=[
            SimpleNamespace(
                metadata={
                    "timestamps": {
                        "started": created,
                        "completed": "2026-04-27T00:00:05+00:00",
                    },
                    "duration": 5,
                }
            )
        ],
        options={"params": {"shots": 128}},
    )

    flattened = flatten_qiskit_execution_metadata(metadata)

    expected_keys = {
        "resource",
        "program_type",
        "num_pubs",
        "job_id",
        "tags",
        "timestamp.created",
        "timestamp.started",
        "timestamp.completed",
        "span.queue",
        "span.work",
        "span.qpu",
        "work_efficiency",
        "pub[0].circuit.depth",
        "pub[0].circuit.size",
        "pub[0].shape",
        "pub[0].timestamp.started",
        "pub[0].timestamp.completed",
        "pub[0].duration",
        "options.params.shots",
    }
    assert expected_keys <= set(flattened)
    assert flattened["pub[0].circuit.depth"] == 1
    assert flattened["pub[0].circuit.size"] == 2
    assert flattened["pub[0].shape"] == [4]
    assert flattened["options.params.shots"] == 128


def test_missing_fields_and_metric_errors_do_not_raise():
    metadata = collect_qiskit_execution_metadata(
        job=_MinimalJob(),
        pubs=[object()],
        options=None,
    )
    flattened = flatten_qiskit_execution_metadata(metadata)

    assert metadata.resource is None
    assert metadata.job_id is None
    assert metadata.pubs[0].circuit.depth is None
    assert "options.params.shots" not in flattened
    assert metadata.collection_errors == ["metrics: RuntimeError"]


def test_unexpected_collection_failures_return_partial_metadata(monkeypatch):
    def fail_collection(*_args, **_kwargs):
        raise RuntimeError("unexpected metadata shape")

    monkeypatch.setattr(metadata_mod, "_pubs_metadata", fail_collection)

    metadata = collect_qiskit_execution_metadata(
        job=_Job(),
        pubs=[(_Circuit(depth=1, size=1),)],
        resource="ibm_kawasaki",
        program_type="sampler",
    )

    assert metadata.resource == "ibm_kawasaki"
    assert metadata.program_type == "sampler"
    assert metadata.num_pubs == 1
    assert "collect: RuntimeError" in metadata.collection_errors
