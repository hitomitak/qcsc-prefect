from __future__ import annotations

from importlib import import_module


def test_native_qiskit_integration_module_imports():
    module = import_module("qcsc_prefect.integrations.qiskit")

    assert module.__name__ == "qcsc_prefect.integrations.qiskit"
    assert module.QiskitRuntimeConfig.__name__ == "QiskitRuntimeConfig"
    assert module.QiskitRuntimeConfigError.__name__ == "QiskitRuntimeConfigError"
    assert module.QiskitEstimatorTaskError.__name__ == "QiskitEstimatorTaskError"
    assert module.QiskitSamplerTaskError.__name__ == "QiskitSamplerTaskError"
    assert module.run_estimator_task.name == "run-qiskit-estimator"
    assert module.run_sampler_task.name == "run-qiskit-sampler"


def test_native_qiskit_placeholder_submodules_import():
    submodules = [
        "artifacts",
        "blocks",
        "metadata",
        "retry",
        "serializers",
        "tasks",
    ]

    for submodule in submodules:
        module = import_module(f"qcsc_prefect.integrations.qiskit.{submodule}")
        assert module.__name__ == f"qcsc_prefect.integrations.qiskit.{submodule}"
