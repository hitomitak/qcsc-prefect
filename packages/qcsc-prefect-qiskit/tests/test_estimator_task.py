from __future__ import annotations

import asyncio
from typing import Any

import pytest
from qcsc_prefect.integrations.qiskit import tasks as tasks_mod
from qcsc_prefect.integrations.qiskit.tasks import QiskitEstimatorTaskError, run_estimator_task


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
        return "job-456"

    def backend(self):
        return _Backend()

    def metrics(self):
        return {}

    def result(self):
        if self.fail_result:
            raise RuntimeError("result failed with hidden-token")
        self.result_called = True
        return [_Result()]


class _Estimator:
    instances: list["_Estimator"] = []

    def __init__(self, *, mode, options=None) -> None:
        self.mode = mode
        self.options = options
        self.run_calls: list[dict[str, Any]] = []
        self.job = _Job()
        self.instances.append(self)

    def run(self, pubs, *, precision=None):
        self.run_calls.append({"pubs": pubs, "precision": precision})
        return self.job


class _FailingRunEstimator(_Estimator):
    def run(self, pubs, *, precision=None):
        self.run_calls.append({"pubs": pubs, "precision": precision})
        raise RuntimeError("submit failed with hidden-token")


class _FailingResultEstimator(_Estimator):
    def __init__(self, *, mode, options=None) -> None:
        super().__init__(mode=mode, options=options)
        self.job.fail_result = True


def _patch_estimator_task(monkeypatch):
    _ConfigAPI.loaded_names = []
    _ConfigAPI.config = _RuntimeConfig()
    _Estimator.instances = []
    artifact_calls = {"metadata": [], "result": []}

    async def fake_create_artifact(metadata, *, result=None, key: str) -> None:
        artifact_calls["metadata"].append(
            {"metadata": metadata, "result": result, "key": key}
        )

    async def fake_create_result_artifact(result, *, key: str) -> None:
        artifact_calls["result"].append({"result": result, "key": key})

    monkeypatch.setattr(tasks_mod, "QiskitRuntimeConfig", _ConfigAPI)
    monkeypatch.setattr(tasks_mod, "_estimator_class", lambda: _Estimator)
    monkeypatch.setattr(
        tasks_mod,
        "create_qiskit_estimator_metadata_artifact",
        fake_create_artifact,
    )
    monkeypatch.setattr(
        tasks_mod,
        "create_qiskit_estimator_result_artifact",
        fake_create_result_artifact,
    )
    return artifact_calls


def test_run_estimator_task_uses_native_backend_and_runs_pubs(monkeypatch):
    artifact_calls = _patch_estimator_task(monkeypatch)
    pubs = ["pub-0"]

    result = asyncio.run(
        run_estimator_task.fn(
            pubs,
            runtime_block_name="ibm-runtime",
            precision=0.01,
            artifact_key="estimator-summary",
            options={"default_precision": 0.02},
        )
    )

    estimator = _Estimator.instances[0]
    assert _ConfigAPI.loaded_names == ["ibm-runtime"]
    assert estimator.mode is _ConfigAPI.config.backend
    assert estimator.options == {"default_precision": 0.02}
    assert estimator.run_calls == [{"pubs": pubs, "precision": 0.01}]
    assert estimator.job.result_called is True
    assert result["primitive"] == "estimator"
    assert result["backend_name"] == "ibm_kawasaki"
    assert result["job_id"] == "job-456"
    assert result["precision"] == 0.01
    assert isinstance(result["result"][0], _Result)
    assert artifact_calls["metadata"][0]["key"] == "estimator-summary"
    assert artifact_calls["metadata"][0]["metadata"].job_id == "job-456"
    assert artifact_calls["metadata"][0]["metadata"].program_type == "estimator"
    assert artifact_calls["metadata"][0]["metadata"].options.params["precision"] == 0.01
    assert isinstance(artifact_calls["metadata"][0]["result"][0], _Result)
    assert artifact_calls["result"][0]["key"] == "estimator-summary-result"
    assert isinstance(artifact_calls["result"][0]["result"][0], _Result)


def test_run_estimator_task_accepts_runtime_config_object(monkeypatch):
    artifact_calls = _patch_estimator_task(monkeypatch)
    runtime_config = _RuntimeConfig()
    pubs = ["pub-0"]

    result = asyncio.run(
        run_estimator_task.fn(
            pubs,
            runtime_config=runtime_config,
            precision=0.02,
        )
    )

    estimator = _Estimator.instances[0]
    assert _ConfigAPI.loaded_names == []
    assert estimator.mode is runtime_config.backend
    assert estimator.run_calls == [{"pubs": pubs, "precision": 0.02}]
    assert result["backend_name"] == "ibm_kawasaki"
    assert result["precision"] == 0.02
    assert artifact_calls["metadata"][0]["metadata"].job_id == "job-456"


def test_run_estimator_task_uses_default_artifact_key(monkeypatch):
    artifact_calls = _patch_estimator_task(monkeypatch)

    asyncio.run(
        run_estimator_task.fn(
            ["pub-0"],
            runtime_block_name="ibm-runtime",
        )
    )

    assert artifact_calls["metadata"][0]["key"] == "qiskit-estimator-summary"
    assert artifact_calls["result"][0]["key"] == "qiskit-estimator-summary-result"


def test_run_estimator_task_wraps_block_load_errors(monkeypatch):
    monkeypatch.setattr(tasks_mod, "QiskitRuntimeConfig", _FailingConfigAPI)

    with pytest.raises(QiskitEstimatorTaskError) as exc_info:
        asyncio.run(
            run_estimator_task.fn(
                ["pub-0"],
                runtime_block_name="missing-runtime",
            )
        )

    message = str(exc_info.value)
    assert "Failed to load QiskitRuntimeConfig block 'missing-runtime'" in message
    assert "ValueError" in message
    assert "hidden-token" not in message
    assert exc_info.value.__cause__ is None


def test_run_estimator_task_wraps_estimator_run_errors(monkeypatch):
    _patch_estimator_task(monkeypatch)
    monkeypatch.setattr(tasks_mod, "_estimator_class", lambda: _FailingRunEstimator)

    with pytest.raises(QiskitEstimatorTaskError) as exc_info:
        asyncio.run(
            run_estimator_task.fn(
                ["pub-0"],
                runtime_block_name="ibm-runtime",
                precision=0.01,
            )
        )

    message = str(exc_info.value)
    assert "Failed to submit native Qiskit EstimatorV2 job" in message
    assert "backend 'ibm_kawasaki'" in message
    assert "RuntimeError" in message
    assert "hidden-token" not in message
    assert exc_info.value.__cause__ is None


def test_run_estimator_task_wraps_job_result_errors(monkeypatch):
    _patch_estimator_task(monkeypatch)
    monkeypatch.setattr(tasks_mod, "_estimator_class", lambda: _FailingResultEstimator)

    with pytest.raises(QiskitEstimatorTaskError) as exc_info:
        asyncio.run(
            run_estimator_task.fn(
                ["pub-0"],
                runtime_block_name="ibm-runtime",
            )
        )

    message = str(exc_info.value)
    assert "Failed while waiting for native Qiskit EstimatorV2 job result" in message
    assert "job_id='job-456'" in message
    assert "backend='ibm_kawasaki'" in message
    assert "RuntimeError" in message
    assert "hidden-token" not in message
    assert exc_info.value.__cause__ is None
