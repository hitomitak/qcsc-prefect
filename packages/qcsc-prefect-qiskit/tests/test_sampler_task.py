from __future__ import annotations

import asyncio
from typing import Any

import pytest
from qcsc_prefect.integrations.qiskit import tasks as tasks_mod
from qcsc_prefect.integrations.qiskit.tasks import QiskitSamplerTaskError, run_sampler_task


class _Backend:
    name = "ibm_kawasaki"


class _RuntimeConfig:
    backend_name = "ibm_kawasaki"

    def __init__(self) -> None:
        self.backend = _Backend()

    def get_backend(self):
        return self.backend


class _ConfigAPI:
    loaded_names: list[str] = []
    config = _RuntimeConfig()

    @staticmethod
    async def load(block_name: str):
        _ConfigAPI.loaded_names.append(block_name)
        return _ConfigAPI.config


class _FailingConfigAPI:
    @staticmethod
    async def load(_block_name: str):
        raise ValueError("missing block with hidden-token")


class _Result:
    def __init__(self) -> None:
        self.metadata = {"shape": [1]}


class _Job:
    def __init__(self) -> None:
        self.result_called = False
        self.tags = ["prefect"]
        self.fail_result = False

    def job_id(self) -> str:
        return "job-123"

    def backend(self):
        return _Backend()

    def metrics(self):
        return {}

    def result(self):
        if self.fail_result:
            raise RuntimeError("result failed with hidden-token")
        self.result_called = True
        return [_Result()]


class _Sampler:
    instances: list["_Sampler"] = []

    def __init__(self, *, mode, options=None) -> None:
        self.mode = mode
        self.options = options
        self.run_calls: list[dict[str, Any]] = []
        self.job = _Job()
        self.instances.append(self)

    def run(self, pubs, *, shots=None):
        self.run_calls.append({"pubs": pubs, "shots": shots})
        return self.job


class _FailingRunSampler(_Sampler):
    def run(self, pubs, *, shots=None):
        self.run_calls.append({"pubs": pubs, "shots": shots})
        raise RuntimeError("submit failed with hidden-token")


class _FailingResultSampler(_Sampler):
    def __init__(self, *, mode, options=None) -> None:
        super().__init__(mode=mode, options=options)
        self.job.fail_result = True


def _patch_sampler_task(monkeypatch):
    _ConfigAPI.loaded_names = []
    _ConfigAPI.config = _RuntimeConfig()
    _Sampler.instances = []
    artifact_calls = {"metadata": [], "result": []}

    async def fake_create_artifact(metadata, *, key: str) -> None:
        artifact_calls["metadata"].append({"metadata": metadata, "key": key})

    async def fake_create_result_artifact(result, *, key: str) -> None:
        artifact_calls["result"].append({"result": result, "key": key})

    monkeypatch.setattr(tasks_mod, "QiskitRuntimeConfig", _ConfigAPI)
    monkeypatch.setattr(tasks_mod, "_sampler_class", lambda: _Sampler)
    monkeypatch.setattr(
        tasks_mod,
        "create_qiskit_sampler_metadata_artifact",
        fake_create_artifact,
    )
    monkeypatch.setattr(
        tasks_mod,
        "create_qiskit_sampler_result_artifact",
        fake_create_result_artifact,
    )
    return artifact_calls


def test_run_sampler_task_uses_native_backend_and_runs_pubs(monkeypatch):
    artifact_calls = _patch_sampler_task(monkeypatch)
    pubs = ["pub-0"]

    result = asyncio.run(
        run_sampler_task.fn(
            pubs,
            runtime_block_name="ibm-runtime",
            shots=1024,
            artifact_key="sampler-summary",
            options={"params": {"resilience_level": 1}},
        )
    )

    sampler = _Sampler.instances[0]
    assert _ConfigAPI.loaded_names == ["ibm-runtime"]
    assert sampler.mode is _ConfigAPI.config.backend
    assert sampler.options == {"params": {"resilience_level": 1}}
    assert sampler.run_calls == [{"pubs": pubs, "shots": 1024}]
    assert sampler.job.result_called is True
    assert result["primitive"] == "sampler"
    assert result["backend_name"] == "ibm_kawasaki"
    assert result["job_id"] == "job-123"
    assert result["shots"] == 1024
    assert isinstance(result["result"][0], _Result)
    assert artifact_calls["metadata"][0]["key"] == "sampler-summary"
    assert artifact_calls["metadata"][0]["metadata"].job_id == "job-123"
    assert artifact_calls["metadata"][0]["metadata"].options.params["shots"] == 1024
    assert artifact_calls["result"][0]["key"] == "sampler-summary-result"
    assert isinstance(artifact_calls["result"][0]["result"][0], _Result)


def test_run_sampler_task_uses_default_artifact_key(monkeypatch):
    artifact_calls = _patch_sampler_task(monkeypatch)

    asyncio.run(
        run_sampler_task.fn(
            ["pub-0"],
            runtime_block_name="ibm-runtime",
        )
    )

    assert artifact_calls["metadata"][0]["key"] == "qiskit-sampler-summary"
    assert artifact_calls["result"][0]["key"] == "qiskit-sampler-summary-result"


def test_run_sampler_task_wraps_block_load_errors(monkeypatch):
    monkeypatch.setattr(tasks_mod, "QiskitRuntimeConfig", _FailingConfigAPI)

    with pytest.raises(QiskitSamplerTaskError) as exc_info:
        asyncio.run(
            run_sampler_task.fn(
                ["pub-0"],
                runtime_block_name="missing-runtime",
            )
        )

    message = str(exc_info.value)
    assert "Failed to load QiskitRuntimeConfig block 'missing-runtime'" in message
    assert "ValueError" in message
    assert "hidden-token" not in message
    assert exc_info.value.__cause__ is None


def test_run_sampler_task_wraps_sampler_run_errors(monkeypatch):
    _patch_sampler_task(monkeypatch)
    monkeypatch.setattr(tasks_mod, "_sampler_class", lambda: _FailingRunSampler)

    with pytest.raises(QiskitSamplerTaskError) as exc_info:
        asyncio.run(
            run_sampler_task.fn(
                ["pub-0"],
                runtime_block_name="ibm-runtime",
                shots=100,
            )
        )

    message = str(exc_info.value)
    assert "Failed to submit native Qiskit SamplerV2 job" in message
    assert "backend 'ibm_kawasaki'" in message
    assert "RuntimeError" in message
    assert "hidden-token" not in message
    assert exc_info.value.__cause__ is None


def test_run_sampler_task_wraps_job_result_errors(monkeypatch):
    _patch_sampler_task(monkeypatch)
    monkeypatch.setattr(tasks_mod, "_sampler_class", lambda: _FailingResultSampler)

    with pytest.raises(QiskitSamplerTaskError) as exc_info:
        asyncio.run(
            run_sampler_task.fn(
                ["pub-0"],
                runtime_block_name="ibm-runtime",
            )
        )

    message = str(exc_info.value)
    assert "Failed while waiting for native Qiskit SamplerV2 job result" in message
    assert "job_id='job-123'" in message
    assert "backend='ibm_kawasaki'" in message
    assert "RuntimeError" in message
    assert "hidden-token" not in message
    assert exc_info.value.__cause__ is None
