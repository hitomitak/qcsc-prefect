from __future__ import annotations

import asyncio

from qcsc_prefect.integrations.qiskit import artifacts as artifacts_mod
from qcsc_prefect.integrations.qiskit.artifacts import (
    build_qiskit_execution_markdown,
    build_qiskit_execution_table,
    build_qiskit_sampler_result_markdown,
    collect_sampler_result_counts,
    create_qiskit_execution_markdown_artifact,
    create_qiskit_execution_table_artifact,
    create_qiskit_sampler_result_artifact,
)
from qcsc_prefect.integrations.qiskit.metadata import QiskitExecutionMetadata


class _BitArray:
    def get_counts(self):
        return {"00": 7, "11": 3}


class _Data:
    meas = _BitArray()


class _PubResult:
    data = _Data()


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


def test_collects_sampler_result_counts_from_meas_register():
    summaries = collect_sampler_result_counts([_PubResult()])

    assert summaries == [
        {
            "pub_index": 0,
            "register": "meas",
            "counts": {"00": 7, "11": 3},
            "truncated": False,
        }
    ]


def test_builds_sampler_result_markdown():
    markdown = build_qiskit_sampler_result_markdown([_PubResult()])

    assert "# Qiskit Sampler Result Summary" in markdown
    assert "`meas`" in markdown
    assert '"00": 7' in markdown


def test_create_artifact_helpers_call_prefect_artifacts(monkeypatch):
    calls = {"markdown": [], "table": []}

    async def fake_create_markdown_artifact(*, markdown: str, key: str) -> None:
        calls["markdown"].append({"markdown": markdown, "key": key})

    async def fake_create_table_artifact(*, table, key: str) -> None:
        calls["table"].append({"table": table, "key": key})

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
    asyncio.run(create_qiskit_sampler_result_artifact([_PubResult()], key="result"))
    asyncio.run(create_qiskit_execution_table_artifact(metadata, key="table"))

    assert calls["markdown"][0]["key"] == "summary"
    assert "ibm_kawasaki" in calls["markdown"][0]["markdown"]
    assert calls["markdown"][1]["key"] == "result"
    assert "Qiskit Sampler Result Summary" in calls["markdown"][1]["markdown"]
    assert calls["table"][0]["key"] == "table"
    assert ["resource", "ibm_kawasaki"] in calls["table"][0]["table"]
