from __future__ import annotations

from typing import Any

from qcsc_prefect.integrations.qiskit import QiskitRuntimeConfig
from qcsc_prefect.integrations.qiskit import blocks as blocks_mod


class _FakeRuntimeService:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        self.backend_calls: list[str] = []

    def backend(self, backend_name: str) -> dict[str, str]:
        self.backend_calls.append(backend_name)
        return {"backend_name": backend_name}


def _patch_runtime_service(monkeypatch):
    _FakeRuntimeService.calls = []
    monkeypatch.setattr(
        blocks_mod,
        "_runtime_service_class",
        lambda: _FakeRuntimeService,
    )
    return _FakeRuntimeService


def test_get_service_uses_block_credentials(monkeypatch):
    service_cls = _patch_runtime_service(monkeypatch)

    config = QiskitRuntimeConfig(
        backend_name="ibm_kawasaki",
        channel="ibm_quantum_platform",
        instance="crn:v1:test-instance",
        token="test-token",
    )

    service = config.get_service()

    assert isinstance(service, service_cls)
    assert service_cls.calls == [
        {
            "channel": "ibm_quantum_platform",
            "instance": "crn:v1:test-instance",
            "token": "test-token",
        }
    ]


def test_get_service_can_defer_to_qiskit_saved_account_or_environment(monkeypatch):
    service_cls = _patch_runtime_service(monkeypatch)

    config = QiskitRuntimeConfig(backend_name="ibm_kawasaki")

    service = config.get_service()

    assert isinstance(service, service_cls)
    assert service_cls.calls == [{}]


def test_get_service_passes_saved_account_name_and_filename(monkeypatch):
    service_cls = _patch_runtime_service(monkeypatch)

    config = QiskitRuntimeConfig(
        backend_name="ibm_kawasaki",
        account_name="production",
        filename="/tmp/qiskit-ibm.json",
    )

    config.get_service()

    assert service_cls.calls == [
        {
            "name": "production",
            "filename": "/tmp/qiskit-ibm.json",
        }
    ]


def test_get_backend_loads_configured_backend(monkeypatch):
    _patch_runtime_service(monkeypatch)

    config = QiskitRuntimeConfig(backend_name="ibm_kawasaki")

    backend = config.get_backend()

    assert backend == {"backend_name": "ibm_kawasaki"}
