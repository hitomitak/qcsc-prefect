from __future__ import annotations

import asyncio

from qcsc_prefect.integrations.qiskit import artifacts as artifacts_mod
from qcsc_prefect.integrations.qiskit.artifacts import (
    build_qiskit_execution_markdown,
    build_qiskit_execution_table,
    create_qiskit_execution_markdown_artifact,
    create_qiskit_execution_table_artifact,
)
from qcsc_prefect.integrations.qiskit.metadata import QiskitExecutionMetadata


def test_builds_readable_markdown_summary():
    metadata = QiskitExecutionMetadata(
        resource="ibm_kawasaki",
        program_type="sampler",
        num_pubs=1,
        job_id="job-123",
    )

    markdown = build_qiskit_execution_markdown(metadata)

    assert "# Qiskit Runtime Job Summary" in markdown
    assert "`resource`" in markdown
    assert "ibm_kawasaki" in markdown
    assert "`job_id`" in markdown
    assert "job-123" in markdown


def test_builds_table_artifact_payload():
    metadata = QiskitExecutionMetadata(resource="ibm_kawasaki")

    table = build_qiskit_execution_table(metadata)

    assert table[0] == ["Key", "Value"]
    assert ["resource", "ibm_kawasaki"] in table


def test_create_artifact_helpers_call_prefect_artifacts(monkeypatch):
    calls = {}

    async def fake_create_markdown_artifact(*, markdown: str, key: str) -> None:
        calls["markdown"] = markdown
        calls["markdown_key"] = key

    async def fake_create_table_artifact(*, table, key: str) -> None:
        calls["table"] = table
        calls["table_key"] = key

    monkeypatch.setattr(
        artifacts_mod,
        "create_markdown_artifact",
        fake_create_markdown_artifact,
    )
    monkeypatch.setattr(
        artifacts_mod,
        "create_table_artifact",
        fake_create_table_artifact,
    )

    metadata = QiskitExecutionMetadata(resource="ibm_kawasaki")
    asyncio.run(create_qiskit_execution_markdown_artifact(metadata, key="summary"))
    asyncio.run(create_qiskit_execution_table_artifact(metadata, key="table"))

    assert calls["markdown_key"] == "summary"
    assert "ibm_kawasaki" in calls["markdown"]
    assert calls["table_key"] == "table"
    assert ["resource", "ibm_kawasaki"] in calls["table"]
