from __future__ import annotations

from importlib import import_module


def test_native_qiskit_integration_module_imports():
    module = import_module("qcsc_prefect.integrations.qiskit")

    assert module.__name__ == "qcsc_prefect.integrations.qiskit"
    assert module.QiskitRuntimeConfig.__name__ == "QiskitRuntimeConfig"
    assert module.QiskitRuntimeConfigError.__name__ == "QiskitRuntimeConfigError"
    assert module.QiskitEstimatorTaskError.__name__ == "QiskitEstimatorTaskError"
    assert module.QiskitJobFetchTaskError.__name__ == "QiskitJobFetchTaskError"
    assert module.QiskitSamplerTaskError.__name__ == "QiskitSamplerTaskError"
    assert callable(module.build_qiskit_cache_payload)
    assert callable(module.build_qiskit_estimator_metadata_markdown)
    assert callable(module.build_qiskit_estimator_result_markdown)
    assert callable(module.create_qiskit_estimator_metadata_artifact)
    assert callable(module.create_qiskit_estimator_result_artifact)
    assert callable(module.create_qiskit_sampler_metadata_artifact)
    assert module.fetch_qiskit_job_result_task.name == "fetch-qiskit-job-result"
    assert callable(module.is_transient_qiskit_error)
    assert callable(module.qiskit_cache_key_from_payload)
    assert callable(module.qiskit_estimator_submit_cache_key)
    assert callable(module.qiskit_result_fetch_cache_key)
    assert callable(module.qiskit_retry_delays)
    assert callable(module.qiskit_sampler_submit_cache_key)
    assert module.run_estimator_task.name == "run-qiskit-estimator"
    assert module.run_sampler_task.name == "run-qiskit-sampler"
    assert callable(module.should_retry_qiskit_fetch_failure)
    assert callable(module.should_retry_qiskit_submit_failure)
    assert module.submit_estimator_job_task.name == "submit-qiskit-estimator-job"
    assert module.submit_sampler_job_task.name == "submit-qiskit-sampler-job"


def test_native_qiskit_placeholder_submodules_import():
    submodules = [
        "artifacts",
        "blocks",
        "cache",
        "metadata",
        "retry",
        "serializers",
        "tasks",
    ]

    for submodule in submodules:
        module = import_module(f"qcsc_prefect.integrations.qiskit.{submodule}")
        assert module.__name__ == f"qcsc_prefect.integrations.qiskit.{submodule}"
