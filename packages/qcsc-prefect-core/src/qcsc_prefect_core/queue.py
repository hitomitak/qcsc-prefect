from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class QueueCapacity:
    """Current scheduler queue capacity estimate.

    Attributes:
        max_active_jobs: Maximum active jobs allowed by the scheduler policy or
            local workflow policy.
        current_active_jobs: Number of currently active scheduler jobs.
        available_slots: Number of jobs that may be submitted now.
        raw_output: Optional raw scheduler output used to estimate capacity.
    """

    max_active_jobs: int
    current_active_jobs: int
    available_slots: int
    raw_output: str | None = None


@runtime_checkable
class QueueProbe(Protocol):
    """Probe queue state and return a generic capacity estimate."""

    def get_capacity(self) -> QueueCapacity:
        """Return the current queue capacity estimate."""


class QueueAwareSubmitGate:
    """Compute conservative submit allowance from a queue capacity probe."""

    def __init__(
        self,
        *,
        queue_probe: QueueProbe,
        max_active_jobs: int = 1000,
        safety_margin: int = 20,
        max_submit_per_refill: int = 100,
    ) -> None:
        self.queue_probe = queue_probe
        self.max_active_jobs = max(0, int(max_active_jobs))
        self.safety_margin = max(0, int(safety_margin))
        self.max_submit_per_refill = max(0, int(max_submit_per_refill))

    def allowed_submit_count(self) -> int:
        """Return how many new jobs may be submitted in this refill cycle.

        Queue probing failures are treated conservatively and return zero.
        """

        try:
            capacity = self.queue_probe.get_capacity()
        except Exception:
            return 0

        effective_max_active = min(
            self.max_active_jobs,
            max(0, int(capacity.max_active_jobs)),
        )
        allowed_by_available_slots = int(capacity.available_slots)
        allowed_by_max_active = (
            effective_max_active - int(capacity.current_active_jobs) - self.safety_margin
        )

        allowed = min(
            allowed_by_available_slots,
            allowed_by_max_active,
            self.max_submit_per_refill,
        )
        return max(0, allowed)
