"""Metadata models for native Qiskit execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class QiskitJobTimestamps(BaseModel):
    """Job-level timestamps reported by Qiskit Runtime."""

    created: datetime | str | None = None
    started: datetime | str | None = None
    completed: datetime | str | None = None


class QiskitExecutionSpans(BaseModel):
    """Execution duration spans in seconds."""

    queue: float | None = None
    work: float | None = None
    qpu: float | None = None


class QiskitCircuitMetadata(BaseModel):
    """Circuit shape information useful for artifacts."""

    depth: int | None = None
    size: int | None = None


class QiskitPubTimestamps(BaseModel):
    """Per-pub timestamps reported by primitive results when available."""

    started: datetime | str | None = None
    completed: datetime | str | None = None


class QiskitPubMetadata(BaseModel):
    """Metadata for one primitive pub."""

    index: int
    circuit: QiskitCircuitMetadata = Field(default_factory=QiskitCircuitMetadata)
    shape: Any | None = None
    timestamp: QiskitPubTimestamps = Field(default_factory=QiskitPubTimestamps)
    duration: float | None = None


class QiskitOptionsMetadata(BaseModel):
    """Qiskit primitive options normalized for artifact metadata."""

    params: dict[str, Any] = Field(default_factory=dict)


class QiskitExecutionMetadata(BaseModel):
    """Common metadata for native Qiskit Runtime execution."""

    resource: str | None = None
    program_type: str | None = None
    num_pubs: int | None = None
    job_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    timestamp: QiskitJobTimestamps = Field(default_factory=QiskitJobTimestamps)
    span: QiskitExecutionSpans = Field(default_factory=QiskitExecutionSpans)
    work_efficiency: float | None = None
    pubs: list[QiskitPubMetadata] = Field(default_factory=list)
    options: QiskitOptionsMetadata = Field(default_factory=QiskitOptionsMetadata)
    collection_errors: list[str] = Field(default_factory=list)


def collect_qiskit_execution_metadata(
    *,
    job: Any | None = None,
    pubs: Sequence[Any] | None = None,
    result: Any | None = None,
    options: Any | None = None,
    resource: str | None = None,
    program_type: str | None = None,
) -> QiskitExecutionMetadata:
    """Collect native Qiskit execution metadata on a best-effort basis."""

    errors: list[str] = []
    try:
        return _collect_qiskit_execution_metadata(
            job=job,
            pubs=pubs,
            result=result,
            options=options,
            resource=resource,
            program_type=program_type,
            errors=errors,
        )
    except Exception as exc:
        errors.append(f"collect: {type(exc).__name__}")
        return QiskitExecutionMetadata(
            resource=resource,
            program_type=program_type,
            num_pubs=_safe_len(pubs) if pubs is not None else _safe_len(result),
            collection_errors=errors,
        )


def _collect_qiskit_execution_metadata(
    *,
    job: Any | None,
    pubs: Sequence[Any] | None,
    result: Any | None,
    options: Any | None,
    resource: str | None,
    program_type: str | None,
    errors: list[str],
) -> QiskitExecutionMetadata:
    """Collect native Qiskit execution metadata from known object shapes."""

    metrics = _safe_get(job, "metrics", errors=errors)
    if not isinstance(metrics, Mapping):
        metrics = {}

    timestamps = _job_timestamps(job=job, metrics=metrics, errors=errors)
    spans = _execution_spans(job=job, timestamps=timestamps, metrics=metrics, errors=errors)
    work_efficiency = _work_efficiency(spans)

    pub_items = list(pubs or [])
    num_pubs = len(pub_items) if pubs is not None else _safe_len(result)

    return QiskitExecutionMetadata(
        resource=resource or _resource_name(job, errors=errors),
        program_type=program_type or _program_type(job, errors=errors),
        num_pubs=num_pubs,
        job_id=_coerce_str(_safe_get(job, "job_id", errors=errors)),
        tags=_tags(job, errors=errors),
        timestamp=timestamps,
        span=spans,
        work_efficiency=work_efficiency,
        pubs=_pubs_metadata(
            pub_items,
            result=result,
            job_timestamps=timestamps,
            errors=errors,
        ),
        options=_options_metadata(options),
        collection_errors=errors,
    )


def flatten_qiskit_execution_metadata(
    metadata: QiskitExecutionMetadata,
) -> dict[str, Any]:
    """Flatten execution metadata into Prefect artifact-friendly keys."""

    flattened: dict[str, Any] = {
        "resource": metadata.resource,
        "program_type": metadata.program_type,
        "num_pubs": metadata.num_pubs,
        "job_id": metadata.job_id,
        "tags": metadata.tags,
        "timestamp.created": _artifact_value(metadata.timestamp.created),
        "timestamp.started": _artifact_value(metadata.timestamp.started),
        "timestamp.completed": _artifact_value(metadata.timestamp.completed),
        "span.queue": metadata.span.queue,
        "span.work": metadata.span.work,
        "span.qpu": metadata.span.qpu,
        "work_efficiency": metadata.work_efficiency,
        "options.params.shots": _mapping_get(metadata.options.params, "shots"),
    }

    for pub in metadata.pubs:
        prefix = f"pub[{pub.index}]"
        flattened.update(
            {
                f"{prefix}.circuit.depth": pub.circuit.depth,
                f"{prefix}.circuit.size": pub.circuit.size,
                f"{prefix}.shape": _artifact_value(pub.shape),
                f"{prefix}.timestamp.started": _artifact_value(pub.timestamp.started),
                f"{prefix}.timestamp.completed": _artifact_value(pub.timestamp.completed),
                f"{prefix}.duration": pub.duration,
            }
        )

    return flattened


def _safe_get(obj: Any, name: str, *, errors: list[str] | None = None) -> Any | None:
    if obj is None:
        return None

    try:
        value = _mapping_get(obj, name) if isinstance(obj, Mapping) else getattr(obj, name)
    except AttributeError:
        return None
    except Exception as exc:
        _record_error(errors, f"{name}: {type(exc).__name__}")
        return None

    if callable(value):
        try:
            return value()
        except Exception as exc:
            _record_error(errors, f"{name}: {type(exc).__name__}")
            return None
    return value


def _mapping_get(obj: Any, key: str) -> Any | None:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return None


def _record_error(errors: list[str] | None, message: str) -> None:
    if errors is not None:
        errors.append(message)


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _resource_name(job: Any | None, *, errors: list[str]) -> str | None:
    backend = _safe_get(job, "backend", errors=errors)
    for name in ("name", "backend_name"):
        value = _safe_get(backend, name, errors=errors)
        if value is not None:
            return str(value)
    if isinstance(backend, str):
        return backend
    return None


def _program_type(job: Any | None, *, errors: list[str]) -> str | None:
    for name in ("primitive_id", "program_id"):
        value = _safe_get(job, name, errors=errors)
        if value is not None:
            return str(value)
    return None


def _tags(job: Any | None, *, errors: list[str]) -> list[str]:
    tags = _safe_get(job, "tags", errors=errors)
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags]
    try:
        return [str(tag) for tag in tags]
    except TypeError:
        return [str(tags)]


def _job_timestamps(
    *,
    job: Any | None,
    metrics: Mapping[str, Any],
    errors: list[str],
) -> QiskitJobTimestamps:
    timestamps = _mapping_get(metrics, "timestamps")
    if not isinstance(timestamps, Mapping):
        timestamps = {}

    return QiskitJobTimestamps(
        created=_first_present(
            timestamps,
            "created",
            "creation",
            "creation_date",
            "created_at",
        )
        or _safe_get(job, "creation_date", errors=errors),
        started=_first_present(
            timestamps,
            "started",
            "running",
            "start",
            "job_started",
            "execution_started",
        ),
        completed=_first_present(
            timestamps,
            "completed",
            "finished",
            "end",
            "job_finished",
            "execution_completed",
        ),
    )


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _execution_spans(
    *,
    job: Any | None,
    timestamps: QiskitJobTimestamps,
    metrics: Mapping[str, Any],
    errors: list[str],
) -> QiskitExecutionSpans:
    spans = _mapping_get(metrics, "span")
    if not isinstance(spans, Mapping):
        spans = {}

    return QiskitExecutionSpans(
        queue=_coerce_float(_first_present(spans, "queue", "queue_time"))
        or _duration_seconds(timestamps.created, timestamps.started),
        work=_coerce_float(_first_present(spans, "work", "work_time"))
        or _duration_seconds(timestamps.started, timestamps.completed),
        qpu=_qpu_span(job=job, metrics=metrics, errors=errors),
    )


def _qpu_span(*, job: Any | None, metrics: Mapping[str, Any], errors: list[str]) -> float | None:
    usage = _mapping_get(metrics, "usage")
    if isinstance(usage, Mapping):
        value = _first_present(usage, "quantum_seconds", "qpu_seconds", "seconds")
        if value is not None:
            return _coerce_float(value)

    usage_estimation = _safe_get(job, "usage_estimation", errors=errors)
    if isinstance(usage_estimation, Mapping):
        value = _first_present(usage_estimation, "quantum_seconds", "qpu_seconds", "seconds")
        if value is not None:
            return _coerce_float(value)

    usage_seconds = _safe_get(job, "usage", errors=errors)
    return _coerce_float(usage_seconds)


def _work_efficiency(spans: QiskitExecutionSpans) -> float | None:
    if spans.qpu is None or not spans.work:
        return None
    return spans.qpu / spans.work


def _pubs_metadata(
    pubs: Sequence[Any],
    *,
    result: Any | None,
    job_timestamps: QiskitJobTimestamps,
    errors: list[str],
) -> list[QiskitPubMetadata]:
    execution_spans = _result_execution_spans(result)
    return [
        _pub_metadata(
            index=index,
            pub=pub,
            result_pub=_sequence_get(result, index),
            fallback_timestamps=job_timestamps if len(pubs) == 1 else None,
            execution_span=_execution_span_for_pub(
                execution_spans,
                pub_index=index,
                num_pubs=len(pubs),
            ),
            errors=errors,
        )
        for index, pub in enumerate(pubs)
    ]


def _pub_metadata(
    *,
    index: int,
    pub: Any,
    result_pub: Any | None,
    fallback_timestamps: QiskitJobTimestamps | None,
    execution_span: Any | None,
    errors: list[str],
) -> QiskitPubMetadata:
    circuit = _pub_circuit(pub)
    timestamp_source = result_pub if result_pub is not None else pub
    timestamps = _pub_timestamps(
        timestamp_source,
        execution_span=execution_span,
        fallback_timestamps=fallback_timestamps,
    )
    return QiskitPubMetadata(
        index=index,
        circuit=QiskitCircuitMetadata(
            depth=_coerce_int(_safe_get(circuit, "depth", errors=errors)),
            size=_coerce_int(_safe_get(circuit, "size", errors=errors)),
        ),
        shape=_pub_shape(pub=pub, result_pub=result_pub, errors=errors),
        timestamp=timestamps,
        duration=_pub_duration(timestamp_source, timestamps),
    )


def _pub_circuit(pub: Any) -> Any | None:
    if isinstance(pub, Mapping):
        return pub.get("circuit")
    if isinstance(pub, (tuple, list)) and pub:
        return pub[0]
    if _looks_like_circuit(pub):
        return pub
    return getattr(pub, "circuit", None)


def _looks_like_circuit(value: Any) -> bool:
    return callable(getattr(value, "depth", None)) and callable(getattr(value, "size", None))


def _pub_shape(*, pub: Any, result_pub: Any | None, errors: list[str]) -> Any | None:
    for source in (result_pub, pub):
        shape = _safe_get(source, "shape", errors=errors)
        if shape is not None:
            return shape
        metadata = _metadata_mapping(source)
        shape = _mapping_get(metadata, "shape")
        if shape is not None:
            return shape
        shape = _data_shape(_safe_get(source, "data", errors=errors), errors=errors)
        if shape is not None:
            return shape
    return None


def _data_shape(data: Any | None, *, errors: list[str]) -> Any | None:
    shape = _safe_get(data, "shape", errors=errors)
    if shape is not None:
        return shape

    for register in _register_names(data):
        register_data = _register_data(data, register)
        shape = _safe_get(register_data, "shape", errors=errors)
        if shape is not None:
            return shape
    return None


def _register_names(data: Any | None) -> list[str]:
    if data is None:
        return []
    if isinstance(data, Mapping):
        return [str(name) for name in data]

    keys = _safe_get(data, "keys")
    if callable(keys):
        try:
            return [str(name) for name in keys()]
        except Exception:
            pass

    if _safe_get(data, "meas") is not None:
        return ["meas"]

    names = getattr(data, "__dict__", {})
    if isinstance(names, dict):
        return [name for name in names if not name.startswith("_")]
    return []


def _register_data(data: Any | None, register: str) -> Any | None:
    if data is None:
        return None
    if isinstance(data, Mapping):
        return data.get(register)
    try:
        return data[register]
    except Exception:
        return _safe_get(data, register)


def _pub_timestamps(
    source: Any | None,
    *,
    execution_span: Any | None = None,
    fallback_timestamps: QiskitJobTimestamps | None = None,
) -> QiskitPubTimestamps:
    metadata = _metadata_mapping(source)
    timestamps = _mapping_get(metadata, "timestamps") or _mapping_get(metadata, "timestamp")
    if not isinstance(timestamps, Mapping):
        timestamps = metadata
    span_started, span_completed = _span_timestamps(execution_span)
    fallback_started = fallback_timestamps.started if fallback_timestamps is not None else None
    fallback_completed = (
        fallback_timestamps.completed if fallback_timestamps is not None else None
    )
    return QiskitPubTimestamps(
        started=(
            _first_present(timestamps, "started", "start", "running")
            or span_started
            or fallback_started
        ),
        completed=(
            _first_present(timestamps, "completed", "finished", "end")
            or span_completed
            or fallback_completed
        ),
    )


def _pub_duration(source: Any | None, timestamps: QiskitPubTimestamps) -> float | None:
    metadata = _metadata_mapping(source)
    return _coerce_float(_mapping_get(metadata, "duration")) or _duration_seconds(
        timestamps.started,
        timestamps.completed,
    )


def _metadata_mapping(source: Any | None) -> Mapping[str, Any]:
    metadata = _safe_get(source, "metadata")
    if isinstance(metadata, Mapping):
        return metadata
    if isinstance(source, Mapping):
        value = source.get("metadata")
        if isinstance(value, Mapping):
            return value
    return {}


def _result_execution_spans(result: Any | None) -> list[Any]:
    metadata = _metadata_mapping(result)
    execution = _mapping_get(metadata, "execution")
    if execution is None:
        execution = _safe_get(result, "execution")

    spans = _mapping_get(execution, "execution_spans")
    if spans is None:
        spans = _safe_get(execution, "execution_spans")
    if spans is None:
        return []

    try:
        return list(spans)
    except TypeError:
        inner_spans = _safe_get(spans, "spans")
        try:
            return list(inner_spans)
        except TypeError:
            return []


def _execution_span_for_pub(
    execution_spans: Sequence[Any],
    *,
    pub_index: int,
    num_pubs: int,
) -> Any | None:
    for span in execution_spans:
        pub_indices = _span_pub_indices(span)
        if pub_indices is not None and pub_index in pub_indices:
            return span

    if len(execution_spans) == num_pubs:
        return execution_spans[pub_index]
    if num_pubs == 1 and execution_spans:
        return execution_spans[0]
    return None


def _span_pub_indices(span: Any | None) -> set[int] | None:
    for name in ("pub_idxs", "pub_indices", "pub_index"):
        value = _safe_get(span, name)
        if value is None:
            continue
        if isinstance(value, int):
            return {value}
        try:
            return {int(item) for item in value}
        except (TypeError, ValueError):
            return None
    return None


def _span_timestamps(span: Any | None) -> tuple[Any | None, Any | None]:
    if span is None:
        return None, None
    return (
        _safe_get(span, "start")
        or _safe_get(span, "started")
        or _safe_get(span, "running"),
        _safe_get(span, "stop")
        or _safe_get(span, "completed")
        or _safe_get(span, "finished")
        or _safe_get(span, "end"),
    )


def _sequence_get(value: Any | None, index: int) -> Any | None:
    if value is None:
        return None
    try:
        return value[index]
    except Exception:
        return None


def _safe_len(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except Exception:
        return None


def _options_metadata(options: Any | None) -> QiskitOptionsMetadata:
    params: dict[str, Any] = {}
    raw_params = None
    if isinstance(options, Mapping):
        raw_params = options.get("params")
    else:
        raw_params = getattr(options, "params", None)

    if isinstance(raw_params, Mapping):
        params = dict(raw_params)
    elif options is not None:
        shots = _safe_get(options, "shots")
        if shots is not None:
            params["shots"] = shots

    return QiskitOptionsMetadata(params=params)


def _duration_seconds(start: Any, end: Any) -> float | None:
    start_dt = _to_datetime(start)
    end_dt = _to_datetime(end)
    if start_dt is None or end_dt is None:
        return None
    return (end_dt - start_dt).total_seconds()


def _to_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _artifact_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    return value
