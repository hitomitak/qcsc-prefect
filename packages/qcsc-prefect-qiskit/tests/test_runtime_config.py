from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from qcsc_prefect.integrations.qiskit import QiskitRuntimeConfig, QiskitRuntimeConfigError
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


def test_backend_name_must_not_be_blank():
    with pytest.raises(ValidationError, match="backend_name must not be empty"):
        QiskitRuntimeConfig(backend_name="  ")


def test_backend_name_is_stripped():
    config = QiskitRuntimeConfig(backend_name="  ibm_kawasaki  ")

    assert config.backend_name == "ibm_kawasaki"


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


def test_get_service_wraps_errors_without_exposing_token(monkeypatch):
    class _FailingRuntimeService:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("raw failure with super-secret-token")

    monkeypatch.setattr(
        blocks_mod,
        "_runtime_service_class",
        lambda: _FailingRuntimeService,
    )
    config = QiskitRuntimeConfig(
        backend_name="ibm_kawasaki",
        channel="ibm_quantum_platform",
        token="super-secret-token",
    )

    with pytest.raises(QiskitRuntimeConfigError) as exc_info:
        config.get_service()

    message = str(exc_info.value)
    assert "Failed to create QiskitRuntimeService" in message
    assert "credential_source='block token'" in message
    assert "super-secret-token" not in message
    assert exc_info.value.__cause__ is None


def test_get_backend_wraps_errors_with_backend_context(monkeypatch):
    class _FailingBackendService:
        def backend(self, _backend_name: str) -> None:
            raise LookupError("backend does not exist")

    monkeypatch.setattr(
        QiskitRuntimeConfig,
        "get_service",
        lambda _self: _FailingBackendService(),
    )
    config = QiskitRuntimeConfig(backend_name="missing_backend")

    with pytest.raises(QiskitRuntimeConfigError) as exc_info:
        config.get_backend()

    message = str(exc_info.value)
    assert "Failed to load Qiskit backend 'missing_backend'" in message
    assert "backend_name='missing_backend'" in message
    assert exc_info.value.__cause__ is None
