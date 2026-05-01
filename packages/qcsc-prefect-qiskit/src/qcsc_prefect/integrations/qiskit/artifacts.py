"""Prefect artifact helpers for native Qiskit execution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from prefect.artifacts import create_markdown_artifact, create_table_artifact
from qcsc_prefect.integrations.qiskit.metadata import (
    QiskitExecutionMetadata,
    flatten_qiskit_execution_metadata,
)


def build_qiskit_execution_markdown(metadata: QiskitExecutionMetadata) -> str:
    """Build a readable Markdown summary for a Qiskit Runtime job."""

    flattened = flatten_qiskit_execution_metadata(metadata)
    rows = [
        "| Key | Value |",
        "| --- | --- |",
    ]
    for key, value in flattened.items():
        rows.append(f"| `{key}` | {_markdown_value(value)} |")
    if metadata.collection_errors:
        rows.extend(
            [
                "",
                "### Metadata Collection Notes",
                "",
                *[f"- `{error}`" for error in metadata.collection_errors],
            ]
        )

    return "\n".join(
        [
            "# Qiskit Runtime Job Summary",
            "",
            *rows,
        ]
    )


def build_qiskit_execution_table(metadata: QiskitExecutionMetadata) -> list[list[Any]]:
    """Build a table artifact payload from flattened Qiskit metadata."""

    flattened = flatten_qiskit_execution_metadata(metadata)
    return [["Key", "Value"], *[[key, value] for key, value in flattened.items()]]


async def create_qiskit_execution_markdown_artifact(
    metadata: QiskitExecutionMetadata,
    *,
    key: str = "qiskit-runtime-summary",
) -> None:
    """Create a Prefect Markdown artifact for a Qiskit Runtime job."""

    await create_markdown_artifact(
        markdown=build_qiskit_execution_markdown(metadata),
        key=key,
    )


async def create_qiskit_execution_table_artifact(
    metadata: QiskitExecutionMetadata,
    *,
    key: str = "qiskit-runtime-metadata",
) -> None:
    """Create a Prefect table artifact for flattened Qiskit metadata."""

    await create_table_artifact(
        table=build_qiskit_execution_table(metadata),
        key=key,
    )


def build_qiskit_sampler_result_markdown(
    result: Any,
    *,
    max_counts: int = 20,
) -> str:
    """Build a compact Markdown summary of Sampler result counts."""

    rows = [
        "| Pub | Register | Counts |",
        "| --- | --- | --- |",
    ]
    summaries = collect_sampler_result_counts(result, max_counts=max_counts)
    if not summaries:
        rows.append("|  |  | No sampler counts found. |")

    for summary in summaries:
        counts_json = json.dumps(summary["counts"], sort_keys=True)
        if summary.get("truncated"):
            counts_json = f"{counts_json} ... truncated"
        rows.append(
            "| "
            f"{summary['pub_index']} | "
            f"`{summary['register']}` | "
            f"`{_markdown_value(counts_json)}` |"
        )

    return "\n".join(
        [
            "# Qiskit Sampler Result Summary",
            "",
            *rows,
        ]
    )


async def create_qiskit_sampler_result_artifact(
    result: Any,
    *,
    key: str = "qiskit-sampler-result",
    max_counts: int = 20,
) -> None:
    """Create a Prefect Markdown artifact for Sampler result counts."""

    await create_markdown_artifact(
        markdown=build_qiskit_sampler_result_markdown(result, max_counts=max_counts),
        key=key,
    )


def build_qiskit_estimator_result_markdown(result: Any) -> str:
    """Build a compact Markdown summary of Estimator result values."""

    rows = [
        "| Pub | EVs | STDs | Ensemble Standard Error | Shots | Target Precision |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    summaries = collect_estimator_result_values(result)
    if not summaries:
        rows.append("|  |  |  |  |  | No estimator values found. |")

    for summary in summaries:
        rows.append(
            "| "
            f"{summary['pub_index']} | "
            f"`{_json_dumps(summary['evs'])}` | "
            f"`{_json_dumps(summary['stds'])}` | "
            f"`{_json_dumps(summary['ensemble_standard_error'])}` | "
            f"`{_json_dumps(summary['shots'])}` | "
            f"`{_json_dumps(summary['target_precision'])}` |"
        )

    return "\n".join(
        [
            "# Qiskit Estimator Result Summary",
            "",
            *rows,
        ]
    )


async def create_qiskit_estimator_result_artifact(
    result: Any,
    *,
    key: str = "qiskit-estimator-result",
) -> None:
    """Create a Prefect Markdown artifact for Estimator result values."""

    await create_markdown_artifact(
        markdown=build_qiskit_estimator_result_markdown(result),
        key=key,
    )


def collect_estimator_result_values(result: Any) -> list[dict[str, Any]]:
    """Collect per-pub estimator values from native Qiskit result objects."""

    summaries: list[dict[str, Any]] = []
    for pub_index, pub_result in enumerate(_iter_result_pubs(result)):
        data = _safe_get_attr(pub_result, "data")
        values = {
            "evs": _json_value(_safe_get_attr(data, "evs")),
            "stds": _json_value(_safe_get_attr(data, "stds")),
            "ensemble_standard_error": _json_value(
                _safe_get_attr(data, "ensemble_standard_error")
            ),
        }
        if all(value is None for value in values.values()):
            continue
        metadata = _safe_get_attr(pub_result, "metadata")
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
    for pub_index, pub_result in enumerate(_iter_result_pubs(result)):
        data = _safe_get_attr(pub_result, "data")
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


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    escaped = str(value).replace("|", "\\|").replace("\n", "<br>")
    return escaped


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]

    tolist = _safe_get_attr(value, "tolist")
    if callable(tolist):
        try:
            return _json_value(tolist())
        except Exception:
            pass

    item = _safe_get_attr(value, "item")
    if callable(item):
        try:
            return _json_value(item())
        except Exception:
            pass

    return str(value)


def _metadata_value(metadata: Any, key: str) -> Any:
    if isinstance(metadata, Mapping):
        return _json_value(metadata.get(key))
    return _json_value(_safe_get_attr(metadata, key))


def _iter_result_pubs(result: Any):
    if result is None:
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


def _safe_get_attr(obj: Any, name: str) -> Any | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _register_names(data: Any) -> list[str]:
    if data is None:
        return []

    keys = _safe_get_attr(data, "keys")
    if callable(keys):
        try:
            return [str(name) for name in keys()]
        except Exception:
            pass

    if hasattr(data, "meas"):
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
        return _safe_get_attr(data, register)


def _counts(register_data: Any) -> dict[str, int] | None:
    get_counts = _safe_get_attr(register_data, "get_counts")
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
