from __future__ import annotations

import asyncio

import pytest
from qcsc_prefect.integrations.qiskit import wrappers as wrappers_mod
from qcsc_prefect.integrations.qiskit.wrappers import (
    QCSCEstimatorV2,
    QCSCPrimitiveJob,
    QCSCSamplerV2,
)


class _RuntimeConfig:
    backend_name = "ibm_kawasaki"


class _FakeTask:
    def __init__(self, name, handler, task_options=None):
        self.name = name
        self.handler = handler
        self.task_options = dict(task_options or {})

    def with_options(self, **options):
        return _FakeTask(
            self.name,
            self.handler,
            {**self.task_options, **options},
        )

    async def __call__(self, *args, **kwargs):
        return self.handler(self.task_options, args, kwargs)


def test_wrapper_requires_exactly_one_runtime_source():
    with pytest.raises(ValueError, match="Either runtime_block_name or runtime_config"):
        QCSCSamplerV2()

    with pytest.raises(ValueError, match="either runtime_block_name or runtime_config"):
        QCSCEstimatorV2(
            runtime_block_name="ibm-runtime",
            runtime_config=_RuntimeConfig(),
        )


def test_sampler_wrapper_simple_mode_delegates_to_run_task(monkeypatch):
    calls = []

    def run_handler(task_options, args, kwargs):
        calls.append({"task_options": task_options, "args": args, "kwargs": kwargs})
        return {"primitive": "sampler", "kwargs": kwargs}

    monkeypatch.setattr(
        wrappers_mod,
        "run_sampler_task",
        _FakeTask("run-qiskit-sampler", run_handler),
    )

    sampler = QCSCSamplerV2(
        runtime_block_name="ibm-runtime",
        options={"default_shots": 100},
    )
    result = asyncio.run(
        sampler.run(
            ["pub-0"],
            shots=100,
            artifact_key="sampler-artifact",
            robust=False,
        )
    )

    assert result["primitive"] == "sampler"
    assert calls[0]["args"] == (["pub-0"],)
    assert calls[0]["kwargs"]["runtime_block_name"] == "ibm-runtime"
    assert calls[0]["kwargs"]["shots"] == 100
    assert calls[0]["kwargs"]["artifact_key"] == "sampler-artifact"
    assert calls[0]["kwargs"]["options"] == {"default_shots": 100}


def test_sampler_wrapper_robust_mode_can_cache_submit_and_fetch(monkeypatch):
    submit_calls = []
    fetch_calls = []
    digest_calls = []

    def digest(pubs, **kwargs):
        digest_calls.append({"pubs": pubs, "kwargs": kwargs})
        return "sampler-digest"

    def submit_handler(task_options, args, kwargs):
        submit_calls.append({"task_options": task_options, "args": args, "kwargs": kwargs})
        return {
            "primitive": "sampler",
            "program_type": "sampler",
            "backend_name": "ibm_kawasaki",
            "job_id": "sampler-job-123",
            "input_digest": kwargs["input_digest"],
        }

    def fetch_handler(task_options, args, kwargs):
        fetch_calls.append({"task_options": task_options, "args": args, "kwargs": kwargs})
        return {
            "primitive": "sampler",
            "job_id": kwargs["job_reference"]["job_id"],
            "task_options": task_options,
        }

    monkeypatch.setattr(wrappers_mod, "build_qiskit_sampler_input_digest", digest)
    monkeypatch.setattr(
        wrappers_mod,
        "submit_sampler_job_task",
        _FakeTask("submit-qiskit-sampler-job", submit_handler),
    )
    monkeypatch.setattr(
        wrappers_mod,
        "fetch_qiskit_job_result_task",
        _FakeTask("fetch-qiskit-job-result", fetch_handler),
    )
    monkeypatch.setattr(wrappers_mod, "qiskit_retry_delays", lambda: [1, 2])

    sampler = QCSCSamplerV2(runtime_config=_RuntimeConfig())
    result = asyncio.run(
        sampler.run(
            ["pub-0"],
            shots=100,
            artifact_key="sampler-artifact",
            cache_submit=True,
            cache_result=True,
            retry_fetch=True,
            cache_scope="global",
        )
    )

    assert result["job_id"] == "sampler-job-123"
    assert digest_calls[0]["kwargs"]["backend_name"] == "ibm_kawasaki"
    assert digest_calls[0]["kwargs"]["shots"] == 100
    assert digest_calls[0]["kwargs"]["cache_scope"] == "global"
    assert submit_calls[0]["kwargs"]["input_digest"] == "sampler-digest"
    assert (
        submit_calls[0]["task_options"]["cache_key_fn"]
        is wrappers_mod.qiskit_sampler_submit_cache_key
    )
    assert submit_calls[0]["task_options"]["persist_result"] is True
    assert fetch_calls[0]["kwargs"]["artifact_key"] == "sampler-artifact"
    assert (
        fetch_calls[0]["task_options"]["cache_key_fn"]
        is wrappers_mod.qiskit_result_fetch_cache_key
    )
    assert fetch_calls[0]["task_options"]["persist_result"] is True
    assert fetch_calls[0]["task_options"]["result_serializer"] == "compressed/pickle"
    assert fetch_calls[0]["task_options"]["retries"] == 2


def test_sampler_submit_returns_job_like_handle_with_result(monkeypatch):
    fetch_calls = []

    def submit_handler(task_options, args, kwargs):
        return {
            "primitive": "sampler",
            "program_type": "sampler",
            "backend_name": "ibm_kawasaki",
            "job_id": "sampler-job-123",
        }

    def fetch_handler(task_options, args, kwargs):
        fetch_calls.append({"task_options": task_options, "args": args, "kwargs": kwargs})
        return {
            "primitive": "sampler",
            "job_id": kwargs["job_reference"]["job_id"],
            "result": "native-sampler-result",
        }

    monkeypatch.setattr(
        wrappers_mod,
        "submit_sampler_job_task",
        _FakeTask("submit-qiskit-sampler-job", submit_handler),
    )
    monkeypatch.setattr(
        wrappers_mod,
        "fetch_qiskit_job_result_task",
        _FakeTask("fetch-qiskit-job-result", fetch_handler),
    )

    sampler = QCSCSamplerV2(runtime_config=_RuntimeConfig())
    job = asyncio.run(
        sampler.submit(
            ["pub-0"],
            shots=100,
            cache_result=True,
        )
    )
    native_result = asyncio.run(job.result())
    output = asyncio.run(job.output())

    assert isinstance(job, QCSCPrimitiveJob)
    assert job.job_id() == "sampler-job-123"
    assert job.primitive == "sampler"
    assert job.backend_name == "ibm_kawasaki"
    assert native_result == "native-sampler-result"
    assert output["result"] == "native-sampler-result"
    assert len(fetch_calls) == 1
    assert fetch_calls[0]["kwargs"]["pubs"] == ["pub-0"]
    assert (
        fetch_calls[0]["task_options"]["cache_key_fn"]
        is wrappers_mod.qiskit_result_fetch_cache_key
    )


def test_estimator_wrapper_robust_mode_can_cache_submit_and_fetch(monkeypatch):
    submit_calls = []
    fetch_calls = []
    digest_calls = []

    def digest(pubs, **kwargs):
        digest_calls.append({"pubs": pubs, "kwargs": kwargs})
        return "estimator-digest"

    def submit_handler(task_options, args, kwargs):
        submit_calls.append({"task_options": task_options, "args": args, "kwargs": kwargs})
        return {
            "primitive": "estimator",
            "program_type": "estimator",
            "backend_name": "ibm_kawasaki",
            "job_id": "estimator-job-456",
            "input_digest": kwargs["input_digest"],
        }

    def fetch_handler(task_options, args, kwargs):
        fetch_calls.append({"task_options": task_options, "args": args, "kwargs": kwargs})
        return {
            "primitive": "estimator",
            "job_id": kwargs["job_reference"]["job_id"],
            "task_options": task_options,
        }

    monkeypatch.setattr(wrappers_mod, "build_qiskit_estimator_input_digest", digest)
    monkeypatch.setattr(
        wrappers_mod,
        "submit_estimator_job_task",
        _FakeTask("submit-qiskit-estimator-job", submit_handler),
    )
    monkeypatch.setattr(
        wrappers_mod,
        "fetch_qiskit_job_result_task",
        _FakeTask("fetch-qiskit-job-result", fetch_handler),
    )

    estimator = QCSCEstimatorV2(runtime_config=_RuntimeConfig())
    result = asyncio.run(
        estimator.run(
            ["pub-0"],
            precision=0.01,
            cache_submit=True,
            cache_result=True,
        )
    )

    assert result["job_id"] == "estimator-job-456"
    assert digest_calls[0]["kwargs"]["backend_name"] == "ibm_kawasaki"
    assert digest_calls[0]["kwargs"]["precision"] == 0.01
    assert submit_calls[0]["kwargs"]["input_digest"] == "estimator-digest"
    assert (
        submit_calls[0]["task_options"]["cache_key_fn"]
        is wrappers_mod.qiskit_estimator_submit_cache_key
    )
    assert (
        fetch_calls[0]["task_options"]["cache_key_fn"]
        is wrappers_mod.qiskit_result_fetch_cache_key
    )
    assert fetch_calls[0]["task_options"]["result_serializer"] == "compressed/pickle"


def test_wrapper_rejects_cache_options_in_simple_mode():
    sampler = QCSCSamplerV2(runtime_config=_RuntimeConfig())

    with pytest.raises(ValueError, match="require robust=True"):
        asyncio.run(sampler.run(["pub-0"], robust=False, cache_result=True))
