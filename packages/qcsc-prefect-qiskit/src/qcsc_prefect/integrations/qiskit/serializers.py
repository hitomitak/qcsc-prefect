"""Summary-only JSON serialization helpers for native Qiskit results.

Example:
    summary_path = save_sampler_result_summary(
        result,
        "runs/qiskit/job-abc123/sampler_summary.json",
    )

These helpers serialize compact JSON summaries only. They are not intended to
replace Prefect result persistence and are not full Qiskit PrimitiveResult
serializers.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from qcsc_prefect.integrations.qiskit.result_summaries import (
    extract_estimator_result_summary,
    extract_sampler_result_summary,
)


def make_json_serializable(value: Any) -> Any:
    """Convert a value to a defensive JSON-safe representation."""

    return _make_json_serializable(value, seen=set())


def save_json(data: Any, output_path: str | Path) -> Path:
    """Save data as UTF-8 JSON after converting it to JSON-safe values."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = make_json_serializable(data)
    with path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
    return path


def load_json(input_path: str | Path) -> Any:
    """Load UTF-8 JSON from a path."""

    with Path(input_path).open(encoding="utf-8") as file:
        return json.load(file)


def serialize_execution_metadata(metadata: Any) -> dict[str, Any]:
    """Serialize Qiskit execution metadata to a JSON-safe dict."""

    if metadata is None:
        return {}

    raw = _metadata_as_mapping(metadata)
    serializable = make_json_serializable(raw)
    if isinstance(serializable, dict):
        return serializable
    return {"value": serializable}


def save_execution_metadata(metadata: Any, output_path: str | Path) -> Path:
    """Save Qiskit execution metadata as JSON."""

    return save_json(serialize_execution_metadata(metadata), output_path)


def save_sampler_result_summary(
    result: Any,
    output_path: str | Path,
    max_counts: int = 20,
) -> Path:
    """Save a summary-only JSON representation of a Sampler result."""

    return save_json(
        extract_sampler_result_summary(result, max_counts=max_counts),
        output_path,
    )


def save_estimator_result_summary(
    result: Any,
    output_path: str | Path,
) -> Path:
    """Save a summary-only JSON representation of an Estimator result."""

    return save_json(extract_estimator_result_summary(result), output_path)


def _make_json_serializable(value: Any, *, seen: set[int]) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()

    value_id = id(value)
    if value_id in seen:
        return "<recursive>"

    if isinstance(value, Mapping):
        seen.add(value_id)
        try:
            return {
                str(key): _make_json_serializable(item, seen=seen)
                for key, item in value.items()
            }
        finally:
            seen.discard(value_id)

    if isinstance(value, tuple | list | set | frozenset):
        seen.add(value_id)
        try:
            items = [_make_json_serializable(item, seen=seen) for item in value]
            if isinstance(value, set | frozenset):
                return sorted(items, key=_json_sort_key)
            return items
        finally:
            seen.discard(value_id)

    seen.add(value_id)
    try:
        converted = _object_to_data(value)
        if converted is not _UNHANDLED:
            return _make_json_serializable(converted, seen=seen)
        return _string_fallback(value)
    finally:
        seen.discard(value_id)


def _metadata_as_mapping(metadata: Any) -> Any:
    for converter in (
        _model_dump_data,
        _dict_method_data,
        _to_dict_data,
    ):
        data = converter(metadata)
        if data is not _UNHANDLED:
            return data
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return _object_dict_data(metadata)


def _object_to_data(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return dataclasses.asdict(value)
        except Exception:
            pass

    for converter in (
        _model_dump_data,
        _to_dict_data,
        _dict_method_data,
        _object_dict_data,
    ):
        data = converter(value)
        if data is not _UNHANDLED:
            return data
    return _UNHANDLED


def _model_dump_data(value: Any) -> Any:
    try:
        model_dump = getattr(value, "model_dump", None)
    except Exception:
        return _UNHANDLED
    if not callable(model_dump):
        return _UNHANDLED
    for kwargs in ({"mode": "json"}, {}):
        try:
            return model_dump(**kwargs)
        except Exception:
            continue
    return _UNHANDLED


def _dict_method_data(value: Any) -> Any:
    try:
        dict_method = getattr(value, "dict", None)
    except Exception:
        return _UNHANDLED
    if not callable(dict_method):
        return _UNHANDLED
    try:
        return dict_method()
    except Exception:
        return _UNHANDLED


def _to_dict_data(value: Any) -> Any:
    try:
        to_dict = getattr(value, "to_dict", None)
    except Exception:
        return _UNHANDLED
    if not callable(to_dict):
        return _UNHANDLED
    try:
        return to_dict()
    except Exception:
        return _UNHANDLED


def _object_dict_data(value: Any) -> Any:
    try:
        data = getattr(value, "__dict__", None)
    except Exception:
        return _UNHANDLED
    if not isinstance(data, Mapping):
        return _UNHANDLED
    return {str(key): item for key, item in data.items() if not str(key).startswith("_")}


def _json_sort_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(value)


def _string_fallback(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


class _Unhandled:
    pass


_UNHANDLED = _Unhandled()
