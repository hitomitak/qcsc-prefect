from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol


class IdentityRecoveryNotSupportedError(NotImplementedError):
    """Raised when a scheduler adapter cannot search jobs by durable identity."""


@dataclass(frozen=True)
class SchedulerJobIdentity:
    """Backend-neutral inputs for one scheduler identity search.

    ``search_token`` is the scheduler-native lookup value, while
    ``stable_identity`` is the full immutable identity that a returned
    candidate must carry. Backend-specific filters belong in ``metadata``.
    """

    search_token: str
    stable_identity: str
    owner: str | None
    search_start: datetime
    search_end: datetime
    metadata: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class SchedulerJobCandidate:
    """Normalized scheduler candidate returned to shared recovery logic."""

    job_id: str
    state: str
    source: str
    identity_matches: bool = True
    metadata_error: str | None = None


class IdentityRecoveryRuntime(Protocol):
    """Optional per-backend capability for finding jobs by durable identity."""

    async def find_candidates_by_identity(
        self,
        identity: SchedulerJobIdentity,
    ) -> list[SchedulerJobCandidate]:
        """Return scheduler candidates without choosing one to attach."""
