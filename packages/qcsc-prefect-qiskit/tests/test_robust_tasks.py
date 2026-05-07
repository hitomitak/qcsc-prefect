from __future__ import annotations

import asyncio
from typing import Any

import pytest
from qcsc_prefect.integrations.qiskit import tasks as tasks_mod
from qcsc_prefect.integrations.qiskit.tasks import (
    QiskitJobFetchTaskError,
    fetch_qiskit_job_result_task,
    submit_estimator_job_task,
    submit_sampler_job_task,
)


class _Backend:
    name = "ibm_kawasaki"


class _Result:
    metadata = {"shape": [1]}


class _Job:
    def __init__(self, job_id: str) -> None:
        self._job_id = job_id
        self.result_called = False

    def job_id(self) -> str:
        return self._job_id

    def backend(self):
        return _Backend()

    def metrics(self):
        return {}

    def result(self):
        self.result_called = True
        return [_Result()]


class _Service:
    fetched_job_ids: list[str] = []
    jobs: dict[str, _Job] = {}

    def job(self, job_id: str):
        self.fetched_job_ids.append(job_id)
        return self.jobs[job_id]


class _RuntimeConfig:
    backend_name = "ibm_kawasaki"

    def __init__(self) -> None:
        self.backend = _Backend()
        self.service = _Service()

    def get_backend(self):
        return self.backend

    def get_service(self):
        return self.service


class _ConfigAPI:
    loaded_names: list[str] = []
    config = _RuntimeConfig()

    @staticmethod
    async def load(block_name: str):
        _ConfigAPI.loaded_names.append(block_name)
        return _ConfigAPI.config


class _Sampler:
    instances: list["_Sampler"] = []

    def __init__(self, *, mode, options=None) -> None:
        self.mode = mode
        self.options = options
        self.run_calls: list[dict[str, Any]] = []
        self.job = _Job("sampler-job-123")
        self.instances.append(self)

    def run(self, pubs, *, shots=None):
        self.run_calls.append({"pubs": pubs, "shots": shots})
        return self.job


class _Estimator:
    instances: list["_Estimator"] = []

    def __init__(self, *, mode, options=None) -> None:
        self.mode = mode
        self.options = options
        self.run_calls: list[dict[str, Any]] = []
        self.job = _Job("estimator-job-456")
        self.instances.append(self)

    def run(self, pubs, *, precision=None):
        self.run_calls.append({"pubs": pubs, "precision": precision})
        return self.job


def _patch_runtime(monkeypatch):
    _ConfigAPI.loaded_names = []
    _ConfigAPI.config = _RuntimeConfig()
    _Service.fetched_job_ids = []
    _Service.jobs = {}
    _Sampler.instances = []
    _Estimator.instances = []
    monkeypatch.setattr(tasks_mod, "QiskitRuntimeConfig", _ConfigAPI)
    monkeypatch.setattr(tasks_mod, "_sampler_class", lambda: _Sampler)
    monkeypatch.setattr(tasks_mod, "_estimator_class", lambda: _Estimator)


def _patch_fetch_artifacts(monkeypatch):
    calls = {"metadata": [], "result": []}

    async def fake_sampler_metadata(metadata, *, key: str) -> None:
        calls["metadata"].append({"primitive": "sampler", "metadata": metadata, "key": key})

    async def fake_sampler_result(result, *, key: str) -> None:
        calls["result"].append({"primitive": "sampler", "result": result, "key": key})

    async def fake_estimator_metadata(metadata, *, result=None, key: str) -> None:
        calls["metadata"].append(
            {"primitive": "estimator", "metadata": metadata, "result": result, "key": key}
        )

    async def fake_estimator_result(result, *, key: str) -> None:
        calls["result"].append({"primitive": "estimator", "result": result, "key": key})

    monkeypatch.setattr(tasks_mod, "create_qiskit_sampler_metadata_artifact", fake_sampler_metadata)
    monkeypatch.setattr(tasks_mod, "create_qiskit_sampler_result_artifact", fake_sampler_result)
    monkeypatch.setattr(
        tasks_mod,
        "create_qiskit_estimator_metadata_artifact",
        fake_estimator_metadata,
    )
    monkeypatch.setattr(
        tasks_mod,
        "create_qiskit_estimator_result_artifact",
        fake_estimator_result,
    )
    return calls


def test_submit_sampler_job_task_returns_job_reference(monkeypatch):
    _patch_runtime(monkeypatch)
    pubs = ["pub-0"]

    reference = asyncio.run(
        submit_sampler_job_task.fn(
            pubs,
            runtime_block_name="ibm-runtime",
            shots=1024,
            options={"resilience_level": 1},
        )
    )

    sampler = _Sampler.instances[0]
    assert sampler.mode is _ConfigAPI.config.backend
    assert sampler.options == {"resilience_level": 1}
    assert sampler.run_calls == [{"pubs": pubs, "shots": 1024}]
    assert sampler.job.result_called is False
    assert reference == {
        "primitive": "sampler",
        "backend_name": "ibm_kawasaki",
        "job_id": "sampler-job-123",
        "shots": 1024,
    }


def test_submit_estimator_job_task_returns_job_reference(monkeypatch):
    _patch_runtime(monkeypatch)
    pubs = ["pub-0"]

    reference = asyncio.run(
        submit_estimator_job_task.fn(
            pubs,
            runtime_block_name="ibm-runtime",
            precision=0.01,
            options={"default_precision": 0.02},
        )
    )

    estimator = _Estimator.instances[0]
    assert estimator.mode is _ConfigAPI.config.backend
    assert estimator.options == {"default_precision": 0.02}
    assert estimator.run_calls == [{"pubs": pubs, "precision": 0.01}]
    assert estimator.job.result_called is False
    assert reference == {
        "primitive": "estimator",
        "backend_name": "ibm_kawasaki",
        "job_id": "estimator-job-456",
        "precision": 0.01,
    }


def test_fetch_qiskit_job_result_task_uses_existing_sampler_job_id(monkeypatch):
    _patch_runtime(monkeypatch)
    artifact_calls = _patch_fetch_artifacts(monkeypatch)
    job = _Job("sampler-job-123")
    _Service.jobs = {"sampler-job-123": job}

    def fail_if_sampler_created():
        raise AssertionError("fetch must not create SamplerV2")

    monkeypatch.setattr(tasks_mod, "_sampler_class", fail_if_sampler_created)

    result = asyncio.run(
        fetch_qiskit_job_result_task.fn(
            runtime_block_name="ibm-runtime",
            job_reference={
                "primitive": "sampler",
                "backend_name": "ibm_kawasaki",
                "job_id": "sampler-job-123",
                "shots": 1024,
            },
            pubs=["pub-0"],
            artifact_key="sampler-fetch",
        )
    )

    assert _Service.fetched_job_ids == ["sampler-job-123"]
    assert job.result_called is True
    assert result["primitive"] == "sampler"
    assert result["job_id"] == "sampler-job-123"
    assert result["shots"] == 1024
    assert artifact_calls["metadata"][0]["key"] == "sampler-fetch"
    assert artifact_calls["result"][0]["key"] == "sampler-fetch-result"


def test_fetch_qiskit_job_result_task_uses_existing_estimator_job_id(monkeypatch):
    _patch_runtime(monkeypatch)
    artifact_calls = _patch_fetch_artifacts(monkeypatch)
    job = _Job("estimator-job-456")
    _Service.jobs = {"estimator-job-456": job}

    def fail_if_estimator_created():
        raise AssertionError("fetch must not create EstimatorV2")

    monkeypatch.setattr(tasks_mod, "_estimator_class", fail_if_estimator_created)

    result = asyncio.run(
        fetch_qiskit_job_result_task.fn(
            runtime_block_name="ibm-runtime",
            job_reference={
                "primitive": "estimator",
                "backend_name": "ibm_kawasaki",
                "job_id": "estimator-job-456",
                "precision": 0.01,
            },
            pubs=["pub-0"],
            artifact_key="estimator-fetch",
        )
    )

    assert _Service.fetched_job_ids == ["estimator-job-456"]
    assert job.result_called is True
    assert result["primitive"] == "estimator"
    assert result["job_id"] == "estimator-job-456"
    assert result["precision"] == 0.01
    assert artifact_calls["metadata"][0]["key"] == "estimator-fetch"
    assert artifact_calls["metadata"][0]["result"] is result["result"]
    assert artifact_calls["result"][0]["key"] == "estimator-fetch-result"


def test_fetch_qiskit_job_result_task_requires_job_id(monkeypatch):
    _patch_runtime(monkeypatch)

    with pytest.raises(QiskitJobFetchTaskError) as exc_info:
        asyncio.run(fetch_qiskit_job_result_task.fn(runtime_block_name="ibm-runtime"))

    assert "job ID is required" in str(exc_info.value)
