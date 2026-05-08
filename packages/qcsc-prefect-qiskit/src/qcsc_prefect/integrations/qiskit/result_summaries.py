"""Pure result summary helpers for native Qiskit result objects."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any


def extract_sampler_result_summary(result: Any, max_counts: int = 20) -> dict[str, Any]:
    """Extract a compact JSON-safe summary from a native Qiskit Sampler result."""

    pub_items = list(iter_result_pubs(result))
    counts = collect_sampler_result_counts(pub_items, max_counts=max_counts)
    return {
        "result_type": _result_type(result),
        "num_pubs": len(pub_items) if result is not None else None,
        "max_counts": max_counts,
        "counts": counts,
        "num_count_summaries": len(counts),
        "truncated": any(summary.get("truncated", False) for summary in counts),
    }


def extract_estimator_result_summary(result: Any) -> dict[str, Any]:
    """Extract a compact JSON-safe summary from a native Qiskit Estimator result."""

    pub_items = list(iter_result_pubs(result))
    values = collect_estimator_result_values(pub_items)
    return {
        "result_type": _result_type(result),
        "num_pubs": len(pub_items) if result is not None else None,
        "values": values,
        "num_value_summaries": len(values),
    }


def collect_estimator_result_values(result: Any) -> list[dict[str, Any]]:
    """Collect per-pub estimator values from native Qiskit result objects."""

    summaries: list[dict[str, Any]] = []
    for pub_index, pub_result in enumerate(iter_result_pubs(result)):
        data = safe_get_attr(pub_result, "data")
        values = {
            "evs": result_json_value(safe_get_attr(data, "evs")),
            "stds": result_json_value(safe_get_attr(data, "stds")),
            "ensemble_standard_error": result_json_value(
                safe_get_attr(data, "ensemble_standard_error")
            ),
        }
        if all(value is None for value in values.values()):
            continue
        metadata = safe_get_attr(pub_result, "metadata")
        summaries.append(
            {
                "pub_index": pub_index,
                **values,
                "shots": _metadata_value(metadata, "shots"),
                "target_precision": _metadata_value(metadata, "target_precision"),
            }
        )
    return summaries


def collect_sampler_result_counts(
    result: Any,
    *,
    max_counts: int = 20,
) -> list[dict[str, Any]]:
    """Collect per-pub sampler counts from native Qiskit result objects."""

    summaries: list[dict[str, Any]] = []
    for pub_index, pub_result in enumerate(iter_result_pubs(result)):
        data = safe_get_attr(pub_result, "data")
        for register in _register_names(data):
            register_data = _register_data(data, register)
            counts = _counts(register_data)
            if counts is None:
                continue
            limited_counts, truncated = _limit_mapping(counts, max_items=max_counts)
            summaries.append(
                {
                    "pub_index": pub_index,
                    "register": register,
                    "counts": limited_counts,
                    "truncated": truncated,
                }
            )
    return summaries


def result_json_value(value: Any) -> Any:
    """Return a compact JSON-safe representation useful in summaries."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): result_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [result_json_value(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [result_json_value(item) for item in value]
        return sorted(items, key=lambda item: str(item))

    tolist = safe_get_attr(value, "tolist")
    if callable(tolist):
        try:
            return result_json_value(tolist())
        except Exception:
            pass

    item = safe_get_attr(value, "item")
    if callable(item):
        try:
            return result_json_value(item())
        except Exception:
            pass

    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def iter_result_pubs(result: Any):
    """Yield pub result entries from known native Qiskit result shapes."""

    if result is None or isinstance(result, str | bytes):
        return
    try:
        yield from result
        return
    except TypeError:
        pass
    try:
        yield result[0]
    except Exception:
        return


def safe_get_attr(obj: Any, name: str) -> Any | None:
    """Return a mapping key or object attribute without propagating lookup errors."""

    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _result_type(result: Any) -> str | None:
    if result is None:
        return None
    return type(result).__name__


def _metadata_value(metadata: Any, key: str) -> Any:
    if isinstance(metadata, Mapping):
        return result_json_value(metadata.get(key))
    return result_json_value(safe_get_attr(metadata, key))


def _register_names(data: Any) -> list[str]:
    if data is None:
        return []

    keys = safe_get_attr(data, "keys")
    if callable(keys):
        try:
            return [str(name) for name in keys()]
        except Exception:
            pass

    if safe_get_attr(data, "meas") is not None:
        return ["meas"]

    names = getattr(data, "__dict__", {})
    if isinstance(names, dict):
        return [name for name in names if not name.startswith("_")]
    return []


def _register_data(data: Any, register: str) -> Any | None:
    if data is None:
        return None
    if isinstance(data, Mapping):
        return data.get(register)
    try:
        return data[register]
    except Exception:
        return safe_get_attr(data, register)


def _counts(register_data: Any) -> dict[str, int] | None:
    get_counts = safe_get_attr(register_data, "get_counts")
    if callable(get_counts):
        try:
            counts = get_counts()
            if isinstance(counts, Mapping):
                return {str(key): int(value) for key, value in counts.items()}
        except Exception:
            return None
    return None


def _limit_mapping(mapping: Mapping[str, int], *, max_items: int) -> tuple[dict[str, int], bool]:
    items = sorted(mapping.items(), key=lambda item: (-item[1], item[0]))
    if max_items < 1:
        return {}, bool(items)
    limited = dict(items[:max_items])
    return limited, len(items) > max_items
