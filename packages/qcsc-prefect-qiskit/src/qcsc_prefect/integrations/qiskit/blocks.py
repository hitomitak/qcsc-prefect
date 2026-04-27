"""Prefect blocks for native Qiskit Runtime configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prefect.blocks.core import Block
from pydantic import Field, SecretStr

if TYPE_CHECKING:
    from qiskit_ibm_runtime import QiskitRuntimeService


def _runtime_service_class():
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService


class QiskitRuntimeConfig(Block):
    """Configuration needed to create native Qiskit Runtime objects."""

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

    def get_service(self) -> QiskitRuntimeService:
        """Create a native ``QiskitRuntimeService`` from configured fields."""

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
        return service_cls(**kwargs)

    def get_backend(self) -> Any:
        """Return the configured backend from the native Qiskit service."""

        service = self.get_service()
        return service.backend(self.backend_name)
