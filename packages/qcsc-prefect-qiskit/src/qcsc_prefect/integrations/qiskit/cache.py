"""Cache key helpers for native Qiskit Prefect tasks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any


def build_qiskit_cache_payload(
    *,
    program_type: str,
    runtime_block_name: str | None = None,
    backend_name: str | None = None,
    shots: int | None = None,
    precision: float | None = None,
    options: Mapping[str, Any] | None = None,
    input_digest: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Build a canonical JSON-safe payload for native Qiskit cache keys.

    The payload intentionally contains only stable scalar/configuration values.
    Runtime objects, backend/service instances, pubs, circuits, and results
    should be represented by an explicit digest or job ID outside this helper.
    """

    program_type_value = str(program_type)
    if (
        program_type_value.strip().lower() in {"sampler", "estimator"}
        and not input_digest
        and job_id is None
    ):
        raise ValueError("input_digest is required for native Qiskit submit cache payloads.")

    payload: dict[str, Any] = {"program_type": program_type_value}
    _add_if_present(payload, "runtime_block_name", runtime_block_name)
    _add_if_present(payload, "backend_name", backend_name)
    _add_if_present(payload, "shots", shots)
    _add_if_present(payload, "precision", precision)
    if options is not None:
        payload["options"] = _canonical_json_value(options)
    _add_if_present(payload, "input_digest", input_digest)
    _add_if_present(payload, "job_id", job_id)
    return payload


def qiskit_cache_key_from_payload(payload: Mapping[str, Any]) -> str:
    """Return a readable SHA-256 cache key for a canonical Qiskit payload."""

    canonical_payload = _canonical_json_value(payload)
    serialized = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    key_type = _cache_key_type(canonical_payload)
    return f"qiskit-{key_type}-{digest}"


def qiskit_sampler_submit_cache_key(_context: Any, parameters: Mapping[str, Any]) -> str | None:
    """Prefect ``cache_key_fn`` helper for native Sampler submit tasks."""

    if _durable_submission_parameters_present(parameters):
        return None
    input_digest = _parameter(parameters, "input_digest")
    if not input_digest:
        return None

    try:
        payload = build_qiskit_cache_payload(
            program_type="sampler",
            runtime_block_name=_parameter(parameters, "runtime_block_name"),
            backend_name=_parameter(parameters, "backend_name"),
            shots=_parameter(parameters, "shots"),
            options=_parameter(parameters, "options"),
            input_digest=str(input_digest),
        )
        return qiskit_cache_key_from_payload(payload)
    except (TypeError, ValueError):
        return None


def qiskit_estimator_submit_cache_key(_context: Any, parameters: Mapping[str, Any]) -> str | None:
    """Prefect ``cache_key_fn`` helper for native Estimator submit tasks."""

    if _durable_submission_parameters_present(parameters):
        return None
    input_digest = _parameter(parameters, "input_digest")
    if not input_digest:
        return None

    try:
        payload = build_qiskit_cache_payload(
            program_type="estimator",
            runtime_block_name=_parameter(parameters, "runtime_block_name"),
            backend_name=_parameter(parameters, "backend_name"),
            precision=_parameter(parameters, "precision"),
            options=_parameter(parameters, "options"),
            input_digest=str(input_digest),
        )
        return qiskit_cache_key_from_payload(payload)
    except (TypeError, ValueError):
        return None


def qiskit_result_fetch_cache_key(_context: Any, parameters: Mapping[str, Any]) -> str | None:
    """Prefect ``cache_key_fn`` helper for optional future raw result caching."""

    job_reference = _parameter(parameters, "job_ref") or _parameter(parameters, "job_reference")
    job_id = _parameter(parameters, "job_id") or _reference_parameter(job_reference, "job_id")
    if job_id is None:
        return None

    program_type = (
        _parameter(parameters, "program_type")
        or _parameter(parameters, "primitive")
        or _reference_parameter(job_reference, "program_type")
        or _reference_parameter(job_reference, "primitive")
    )
    runtime_block_name = _parameter(parameters, "runtime_block_name") or _reference_parameter(
        job_reference,
        "runtime_block_name",
    )

    try:
        payload = build_qiskit_cache_payload(
            program_type=str(program_type) if program_type is not None else "result",
            runtime_block_name=runtime_block_name,
            job_id=str(job_id),
        )
        return qiskit_cache_key_from_payload(payload)
    except (TypeError, ValueError):
        return None


def _add_if_present(payload: dict[str, Any], key: str, value: Any | None) -> None:
    if value is not None:
        payload[key] = _canonical_json_value(value)


def _canonical_json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Qiskit cache payload cannot contain non-finite floats.")
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple | list):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [_canonical_json_value(item) for item in value]
        return sorted(items, key=_canonical_sort_key)
    raise TypeError(f"Qiskit cache payload contains unsupported value {type(value).__name__}.")


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _cache_key_type(payload: Mapping[str, Any]) -> str:
    if payload.get("job_id") is not None and payload.get("input_digest") is None:
        return "result"

    raw_type = payload.get("cache_kind") or payload.get("program_type") or "runtime"
    normalized = str(raw_type).strip().lower().replace("_", "-")
    if normalized in {"sampler", "estimator", "result"}:
        return normalized
    return "runtime"


def _parameter(parameters: Mapping[str, Any], key: str) -> Any | None:
    if isinstance(parameters, Mapping):
        return parameters.get(key)
    return None


def _reference_parameter(reference: Any, key: str) -> Any | None:
    if isinstance(reference, Mapping):
        return reference.get(key)
    return None


def _durable_submission_parameters_present(parameters: Mapping[str, Any]) -> bool:
    return any(
        _parameter(parameters, key) is not None
        for key in ("submission_key", "spec_hash", "journal_path")
    )
