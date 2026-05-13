"""Prefect blocks for native Qiskit Runtime configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prefect.blocks.core import Block
from pydantic import Field, SecretStr, field_validator

if TYPE_CHECKING:
    from qiskit_ibm_runtime import QiskitRuntimeService


def _runtime_service_class():
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService


class QiskitRuntimeConfigError(RuntimeError):
    """Raised when native Qiskit Runtime configuration cannot be resolved."""


class QiskitRuntimeConfig(Block):
    """Configuration needed to create native Qiskit Runtime objects.

    The block stores only the configuration needed to construct native
    ``qiskit_ibm_runtime`` objects. It does not reimplement Runtime behavior.

    If ``token`` is not set, Qiskit's normal saved account file and environment
    variable discovery are used.
    """

    _block_type_name = "Qiskit Runtime Config"
    _block_type_slug = "qiskit_runtime_config"

    backend_name: str = Field(title="Backend Name")
    channel: str | None = Field(
        default=None,
        title="Channel",
        description=(
            "Qiskit Runtime channel. Leave unset to let Qiskit load saved "
            "account or environment credentials."
        ),
    )
    instance: str | None = Field(
        default=None,
        title="Instance",
        description="IBM Quantum service instance or CRN.",
    )
    token: SecretStr | None = Field(
        default=None,
        title="API Token",
        description=(
            "Optional IBM Quantum API token. Leave unset to use Qiskit's "
            "saved account file or environment variable discovery."
        ),
    )
    account_name: str | None = Field(
        default=None,
        title="Saved Account Name",
        description="Optional Qiskit saved account name passed as 'name'.",
    )
    filename: str | None = Field(
        default=None,
        title="Saved Account File",
        description="Optional Qiskit saved account filename.",
    )

    @field_validator("backend_name")
    @classmethod
    def _validate_backend_name(cls, value: str) -> str:
        backend_name = value.strip()
        if not backend_name:
            raise ValueError("backend_name must not be empty.")
        return backend_name

    def _safe_context(self) -> str:
        credential_source = "block token" if self.token is not None else "Qiskit discovery"
        return (
            f"backend_name={self.backend_name!r}, "
            f"channel={self.channel!r}, "
            f"instance={self.instance!r}, "
            f"account_name={self.account_name!r}, "
            f"filename={self.filename!r}, "
            f"credential_source={credential_source!r}"
        )

    def get_service(self) -> QiskitRuntimeService:
        """Create a native ``QiskitRuntimeService`` from configured fields.

        Returns:
            A native ``qiskit_ibm_runtime.QiskitRuntimeService``.

        Raises:
            QiskitRuntimeConfigError: If the service cannot be created.
        """

        kwargs: dict[str, Any] = {}
        if self.channel is not None:
            kwargs["channel"] = self.channel
        if self.instance is not None:
            kwargs["instance"] = self.instance
        if self.token is not None:
            kwargs["token"] = self.token.get_secret_value()
        if self.account_name is not None:
            kwargs["name"] = self.account_name
        if self.filename is not None:
            kwargs["filename"] = self.filename

        service_cls = _runtime_service_class()
        try:
            return service_cls(**kwargs)
        except Exception as exc:
            raise QiskitRuntimeConfigError(
                "Failed to create QiskitRuntimeService "
                f"({type(exc).__name__}). Context: {self._safe_context()}."
            ) from None

    def get_backend(self) -> Any:
        """Return the configured backend from the native Qiskit service.

        Returns:
            The native backend returned by ``service.backend(backend_name)``.

        Raises:
            QiskitRuntimeConfigError: If the service or backend cannot be
                created.
        """

        service = self.get_service()
        try:
            return service.backend(self.backend_name)
        except Exception as exc:
            raise QiskitRuntimeConfigError(
                "Failed to load Qiskit backend "
                f"{self.backend_name!r} ({type(exc).__name__}). "
                f"Context: {self._safe_context()}."
            ) from None
