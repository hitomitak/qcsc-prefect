"""Prefect artifact helpers for native Qiskit execution."""

from __future__ import annotations

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


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    escaped = str(value).replace("|", "\\|").replace("\n", "<br>")
    return escaped
