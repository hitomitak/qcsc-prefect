from __future__ import annotations

import asyncio

from qcsc_prefect.integrations.qiskit import tasks as tasks_mod
from qcsc_prefect.integrations.qiskit.cache import (
    build_qiskit_cache_payload,
    qiskit_cache_key_from_payload,
    qiskit_estimator_submit_cache_key,
    qiskit_result_fetch_cache_key,
    qiskit_sampler_submit_cache_key,
)
from qcsc_prefect.integrations.qiskit.tasks import (
    build_cached_fetch_qiskit_job_result_task,
    cached_fetch_qiskit_job_result_task,
    submit_sampler_job_task,
)


class _UnserializablePub:
    def __repr__(self) -> str:
        raise AssertionError("cache helpers must not serialize pubs")


class _Backend:
    name = "ibm_kawasaki"


class _Job:
    result_called = False

    def job_id(self) -> str:
        return "sampler-job-123"


class _RuntimeConfig:
    backend_name = "ibm_kawasaki"

    def __init__(self) -> None:
        self.backend = _Backend()

    def get_backend(self):
        return self.backend


class _ConfigAPI:
    config = _RuntimeConfig()

    @staticmethod
    async def load(_block_name: str):
        return _ConfigAPI.config


class _Sampler:
    instances: list["_Sampler"] = []

    def __init__(self, *, mode, options=None) -> None:
        self.mode = mode
        self.options = options
        self.run_calls = []
        self.job = _Job()
        self.instances.append(self)

    def run(self, pubs, *, shots=None):
        self.run_calls.append({"pubs": pubs, "shots": shots})
        return self.job


def test_same_payload_gives_same_cache_key():
    payload_a = build_qiskit_cache_payload(
        program_type="sampler",
        runtime_block_name="ibm-runtime",
        shots=1024,
        options={"b": 2, "a": {"z": [3, 1]}},
        input_digest="digest-abc",
    )
    payload_b = build_qiskit_cache_payload(
        program_type="sampler",
        runtime_block_name="ibm-runtime",
        shots=1024,
        options={"a": {"z": [3, 1]}, "b": 2},
        input_digest="digest-abc",
    )

    assert payload_a == payload_b
    assert qiskit_cache_key_from_payload(payload_a) == qiskit_cache_key_from_payload(payload_b)
    assert qiskit_cache_key_from_payload(payload_a).startswith("qiskit-sampler-")


def test_different_input_digest_gives_different_cache_key():
    first = qiskit_sampler_submit_cache_key(
        None,
        {
            "pubs": [_UnserializablePub()],
            "runtime_block_name": "ibm-runtime",
            "shots": 1024,
            "input_digest": "digest-abc",
        },
    )
    second = qiskit_sampler_submit_cache_key(
        None,
        {
            "pubs": [_UnserializablePub()],
            "runtime_block_name": "ibm-runtime",
            "shots": 1024,
            "input_digest": "digest-def",
        },
    )

    assert first is not None
    assert second is not None
    assert first != second


def test_different_shots_gives_different_sampler_submit_cache_key():
    base_parameters = {
        "pubs": [_UnserializablePub()],
        "runtime_block_name": "ibm-runtime",
        "input_digest": "digest-abc",
    }

    assert qiskit_sampler_submit_cache_key(
        None,
        {**base_parameters, "shots": 1024},
    ) != qiskit_sampler_submit_cache_key(
        None,
        {**base_parameters, "shots": 2048},
    )


def test_different_precision_gives_different_estimator_submit_cache_key():
    base_parameters = {
        "pubs": [_UnserializablePub()],
        "runtime_block_name": "ibm-runtime",
        "input_digest": "digest-abc",
    }

    assert qiskit_estimator_submit_cache_key(
        None,
        {**base_parameters, "precision": 0.01},
    ) != qiskit_estimator_submit_cache_key(
        None,
        {**base_parameters, "precision": 0.02},
    )


def test_different_options_gives_different_cache_key():
    base_parameters = {
        "pubs": [_UnserializablePub()],
        "runtime_block_name": "ibm-runtime",
        "shots": 1024,
        "input_digest": "digest-abc",
    }

    assert qiskit_sampler_submit_cache_key(
        None,
        {**base_parameters, "options": {"params": {"resilience_level": 1}}},
    ) != qiskit_sampler_submit_cache_key(
        None,
        {**base_parameters, "options": {"params": {"resilience_level": 2}}},
    )


def test_missing_input_digest_returns_none_for_submit_cache_helpers():
    assert qiskit_sampler_submit_cache_key(None, {"shots": 1024}) is None
    assert qiskit_estimator_submit_cache_key(None, {"precision": 0.01}) is None


def test_missing_job_id_returns_none_for_fetch_cache_helper():
    assert (
        qiskit_result_fetch_cache_key(None, {"job_reference": {"program_type": "sampler"}})
        is None
    )


def test_cache_helpers_do_not_attempt_to_serialize_pubs_or_circuits():
    key = qiskit_sampler_submit_cache_key(
        None,
        {
            "pubs": [_UnserializablePub()],
            "circuits": [_UnserializablePub()],
            "runtime_block_name": "ibm-runtime",
            "shots": 1024,
            "input_digest": "digest-abc",
        },
    )

    assert key is not None


def test_result_fetch_cache_key_uses_job_reference_and_result_prefix():
    key = qiskit_result_fetch_cache_key(
        None,
        {
            "job_reference": {
                "program_type": "sampler",
                "runtime_block_name": "ibm-runtime",
                "job_id": "sampler-job-123",
            }
        },
    )

    assert key is not None
    assert key.startswith("qiskit-result-")


def test_cached_fetch_task_uses_prefect_compressed_pickle_result_cache():
    assert cached_fetch_qiskit_job_result_task.cache_key_fn is qiskit_result_fetch_cache_key
    assert cached_fetch_qiskit_job_result_task.persist_result is True
    assert cached_fetch_qiskit_job_result_task.result_serializer == "compressed/pickle"


def test_build_cached_fetch_task_preserves_extra_task_options():
    cached_task = build_cached_fetch_qiskit_job_result_task(retries=2)

    assert cached_task.cache_key_fn is qiskit_result_fetch_cache_key
    assert cached_task.persist_result is True
    assert cached_task.result_serializer == "compressed/pickle"
    assert cached_task.retries == 2


def test_cached_fetch_task_can_be_combined_with_retry_options():
    retried_cached_task = cached_fetch_qiskit_job_result_task.with_options(retries=2)

    assert retried_cached_task.cache_key_fn is qiskit_result_fetch_cache_key
    assert retried_cached_task.persist_result is True
    assert retried_cached_task.result_serializer == "compressed/pickle"
    assert retried_cached_task.retries == 2


def test_submit_task_job_reference_includes_input_digest(monkeypatch):
    _ConfigAPI.config = _RuntimeConfig()
    _Sampler.instances = []
    monkeypatch.setattr(tasks_mod, "QiskitRuntimeConfig", _ConfigAPI)
    monkeypatch.setattr(tasks_mod, "_sampler_class", lambda: _Sampler)

    reference = asyncio.run(
        submit_sampler_job_task.fn(
            ["pub-0"],
            runtime_block_name="ibm-runtime",
            shots=1024,
            input_digest="digest-abc",
        )
    )

    assert _Sampler.instances[0].run_calls == [{"pubs": ["pub-0"], "shots": 1024}]
    assert reference["program_type"] == "sampler"
    assert reference["runtime_block_name"] == "ibm-runtime"
    assert reference["input_digest"] == "digest-abc"
