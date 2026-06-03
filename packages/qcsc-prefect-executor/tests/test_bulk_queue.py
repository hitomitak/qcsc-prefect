from __future__ import annotations

from qcsc_prefect_executor.bulk.queue import QueueAwareSubmitGate, QueueCapacity


class _FakeQueueProbe:
    def __init__(self, capacity: QueueCapacity) -> None:
        self.capacity = capacity

    def get_capacity(self) -> QueueCapacity:
        return self.capacity


class _FailingQueueProbe:
    def get_capacity(self) -> QueueCapacity:
        raise RuntimeError("queue probe failed")


def test_allowed_submit_count_respects_available_slots():
    gate = QueueAwareSubmitGate(
        queue_probe=_FakeQueueProbe(
            QueueCapacity(
                max_active_jobs=1000,
                current_active_jobs=10,
                available_slots=7,
            )
        ),
        safety_margin=0,
        max_submit_per_refill=100,
    )

    assert gate.allowed_submit_count() == 7


def test_allowed_submit_count_respects_max_submit_per_refill():
    gate = QueueAwareSubmitGate(
        queue_probe=_FakeQueueProbe(
            QueueCapacity(
                max_active_jobs=1000,
                current_active_jobs=10,
                available_slots=500,
            )
        ),
        safety_margin=0,
        max_submit_per_refill=25,
    )

    assert gate.allowed_submit_count() == 25


def test_allowed_submit_count_respects_safety_margin():
    gate = QueueAwareSubmitGate(
        queue_probe=_FakeQueueProbe(
            QueueCapacity(
                max_active_jobs=100,
                current_active_jobs=85,
                available_slots=100,
            )
        ),
        max_active_jobs=100,
        safety_margin=10,
        max_submit_per_refill=100,
    )

    assert gate.allowed_submit_count() == 5


def test_allowed_submit_count_never_returns_negative_values():
    gate = QueueAwareSubmitGate(
        queue_probe=_FakeQueueProbe(
            QueueCapacity(
                max_active_jobs=100,
                current_active_jobs=120,
                available_slots=-5,
            )
        ),
        max_active_jobs=100,
        safety_margin=20,
        max_submit_per_refill=100,
    )

    assert gate.allowed_submit_count() == 0


def test_probe_failure_results_in_zero_submit_allowance():
    gate = QueueAwareSubmitGate(
        queue_probe=_FailingQueueProbe(),
        max_active_jobs=100,
        safety_margin=0,
        max_submit_per_refill=100,
    )

    assert gate.allowed_submit_count() == 0


def test_queue_capacity_preserves_raw_output():
    capacity = QueueCapacity(
        max_active_jobs=10,
        current_active_jobs=3,
        available_slots=7,
        raw_output="raw scheduler output",
    )

    assert capacity.raw_output == "raw scheduler output"
