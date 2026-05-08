"""Prefect artifact helpers for native Qiskit execution."""

from __future__ import annotations

import json
from typing import Any

from prefect.artifacts import create_markdown_artifact, create_table_artifact
from qcsc_prefect.integrations.qiskit.metadata import (
    QiskitExecutionMetadata,
    flatten_qiskit_execution_metadata,
)
from qcsc_prefect.integrations.qiskit.result_summaries import (
    collect_estimator_result_values as collect_estimator_result_values,
)
from qcsc_prefect.integrations.qiskit.result_summaries import (
    collect_sampler_result_counts as collect_sampler_result_counts,
)
from qcsc_prefect.integrations.qiskit.result_summaries import (
    extract_estimator_result_summary,
    extract_sampler_result_summary,
    iter_result_pubs,
    result_json_value,
    safe_get_attr,
)


def build_qiskit_execution_markdown(metadata: QiskitExecutionMetadata) -> str:
    """Build a readable Markdown summary for a Qiskit Runtime job."""

    return _build_qiskit_metadata_markdown(
        metadata,
        title="Qiskit Runtime Job Summary",
    )


def build_qiskit_sampler_metadata_markdown(metadata: QiskitExecutionMetadata) -> str:
    """Build a readable Markdown metadata summary for a Sampler job."""

    return _build_qiskit_metadata_markdown(
        metadata,
        title="Qiskit Sampler Metadata Summary",
    )


def build_qiskit_estimator_metadata_markdown(
    metadata: QiskitExecutionMetadata,
    *,
    result: Any | None = None,
) -> str:
    """Build a readable Markdown metadata summary for an Estimator job."""

    sections = [
        _build_qiskit_metadata_markdown(
            metadata,
            title="Qiskit Estimator Metadata Summary",
        )
    ]

    result_metadata = safe_get_attr(result, "metadata")
    if result_metadata is not None:
        sections.extend(
            [
                "",
                "### Estimator Primitive Result Metadata",
                "",
                "```json",
                _json_dumps(result_metadata),
                "```",
            ]
        )

    pub_metadata_rows = [
        "| Pub | Metadata |",
        "| --- | --- |",
    ]
    has_pub_metadata = False
    for pub_index, pub_result in enumerate(iter_result_pubs(result)):
        pub_metadata = safe_get_attr(pub_result, "metadata")
        if pub_metadata is None:
            continue
        has_pub_metadata = True
        pub_metadata_rows.append(
            f"| {pub_index} | `{_markdown_value(_json_dumps(pub_metadata))}` |"
        )
    if has_pub_metadata:
        sections.extend(
            [
                "",
                "### Estimator Pub Metadata",
                "",
                *pub_metadata_rows,
            ]
        )

    return "\n".join(sections)


def _build_qiskit_metadata_markdown(
    metadata: QiskitExecutionMetadata,
    *,
    title: str,
) -> str:
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
            f"# {title}",
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


async def create_qiskit_sampler_metadata_artifact(
    metadata: QiskitExecutionMetadata,
    *,
    key: str = "qiskit-sampler-summary",
) -> None:
    """Create a Prefect Markdown metadata artifact for a Sampler job."""

    await create_markdown_artifact(
        markdown=build_qiskit_sampler_metadata_markdown(metadata),
        key=key,
    )


async def create_qiskit_estimator_metadata_artifact(
    metadata: QiskitExecutionMetadata,
    *,
    result: Any | None = None,
    key: str = "qiskit-estimator-summary",
) -> None:
    """Create a Prefect Markdown metadata artifact for an Estimator job."""

    await create_markdown_artifact(
        markdown=build_qiskit_estimator_metadata_markdown(metadata, result=result),
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
    summary = extract_sampler_result_summary(result, max_counts=max_counts)
    summaries = summary["counts"]
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
    summary = extract_estimator_result_summary(result)
    summaries = summary["values"]
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


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    escaped = str(value).replace("|", "\\|").replace("\n", "<br>")
    return escaped


def _json_dumps(value: Any) -> str:
    return json.dumps(result_json_value(value), sort_keys=True)
