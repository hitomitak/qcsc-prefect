from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from qcsc_prefect_adapters.slurm import runtime as slurm_runtime
from qcsc_prefect_executor import from_blocks as mod
from qcsc_prefect_executor.bulk.models import (
    BulkCancelOutcome,
    BulkJobSpec,
    BulkJobStatus,
)
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry


class _CancelRuntimeStub:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.cancel_calls: list[dict[str, Any]] = []

    async def cancel(self, job_id: str, **kwargs: Any) -> None:
        self.cancel_calls.append({"job_id": job_id, **kwargs})
        if self.error is not None:
            raise self.error


def _submitted_registry(tmp_path: Path) -> BulkJobRegistry:
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")
    registry.upsert_jobs([BulkJobSpec(job_key="job-1", work_dir=tmp_path / "job-1")])
    registry.mark_submitted("job-1", "12345")
    return registry


def _patch_slurm_target(monkeypatch) -> None:
    async def resolve_target(*, hpc_profile_block_name: str) -> str:
        assert hpc_profile_block_name == "hpc-slurm"
        return "slurm"

    monkeypatch.setattr(mod, "resolve_hpc_target", resolve_target)


def _patch_scheduler_rows(monkeypatch, rows: dict[str, dict[str, Any]]) -> None:
    async def query(*, hpc_target: str, scheduler_job_ids: list[str]):
        assert hpc_target == "slurm"
        assert scheduler_job_ids == ["12345"]
        return rows

    monkeypatch.setattr(mod, "_query_scheduler_statuses", query)


def test_cancel_executor_without_intent_has_no_scheduler_side_effect(tmp_path: Path, monkeypatch):
    registry = _submitted_registry(tmp_path)

    class _UnexpectedRuntime:
        def __init__(self) -> None:
            raise AssertionError("SlurmRuntime must not be constructed without cancel intent")

    monkeypatch.setattr(mod, "SlurmRuntime", _UnexpectedRuntime)

    outcome = asyncio.run(
        mod.execute_cancel_request(
            registry=registry,
            job_key="job-1",
            hpc_profile_block="hpc-slurm",
        )
    )

    assert outcome is None
    assert registry.get_job("job-1").cancel_attempts == 0


def test_cancel_executor_dispatches_once_for_durable_intent(tmp_path: Path, monkeypatch):
    registry = _submitted_registry(tmp_path)
    registry.request_cancel("job-1", requested_by="operator", reason="stop test")
    runtime = _CancelRuntimeStub()
    _patch_slurm_target(monkeypatch)
    _patch_scheduler_rows(monkeypatch, {"12345": {"JobID": "12345", "State": "RUNNING"}})
    monkeypatch.setattr(mod, "SlurmRuntime", lambda: runtime)

    first = asyncio.run(
        mod.execute_cancel_request(
            registry=registry,
            job_key="job-1",
            hpc_profile_block="hpc-slurm",
            scheduler_command_timeout_seconds=12.5,
        )
    )
    second = asyncio.run(
        mod.execute_cancel_request(
            registry=registry,
            job_key="job-1",
            hpc_profile_block="hpc-slurm",
        )
    )

    assert first == second == BulkCancelOutcome.REQUEST_ACCEPTED
    assert runtime.cancel_calls == [
        {"job_id": "12345", "intent_confirmed": True, "timeout_seconds": 12.5}
    ]
    record = registry.get_job("job-1")
    assert record.cancel_attempts == 1
    assert record.cancel_outcome == BulkCancelOutcome.REQUEST_ACCEPTED
    assert record.status == BulkJobStatus.SUBMITTED


def test_cancel_executor_validates_timeout_before_dispatch_claim(tmp_path: Path, monkeypatch):
    registry = _submitted_registry(tmp_path)
    registry.request_cancel("job-1", requested_by="operator", reason="stop test")

    with pytest.raises(ValueError, match="greater than 0"):
        asyncio.run(
            mod.execute_cancel_request(
                registry=registry,
                job_key="job-1",
                hpc_profile_block="hpc-slurm",
                scheduler_command_timeout_seconds=0,
            )
        )

    record = registry.get_job("job-1")
    assert record.cancel_attempts == 0
    assert record.cancel_outcome is None


def test_cancel_executor_does_not_scancel_already_terminal_job(tmp_path: Path, monkeypatch):
    registry = _submitted_registry(tmp_path)
    registry.request_cancel("job-1", requested_by="operator", reason="stop test")
    runtime = _CancelRuntimeStub()
    _patch_slurm_target(monkeypatch)
    _patch_scheduler_rows(monkeypatch, {"12345": {"JobID": "12345", "State": "CANCELLED"}})
    monkeypatch.setattr(mod, "SlurmRuntime", lambda: runtime)

    outcome = asyncio.run(
        mod.execute_cancel_request(
            registry=registry,
            job_key="job-1",
            hpc_profile_block="hpc-slurm",
        )
    )

    assert outcome == BulkCancelOutcome.ALREADY_TERMINAL
    assert runtime.cancel_calls == []
    record = registry.get_job("job-1")
    assert record.status == BulkJobStatus.CANCELLED
    assert record.cancel_outcome == BulkCancelOutcome.ALREADY_TERMINAL
    assert record.cancel_attempts == 0


@pytest.mark.parametrize(
    ("error", "expected_outcome"),
    [
        (
            slurm_runtime.CancelNotFoundError("not found"),
            BulkCancelOutcome.NOT_FOUND,
        ),
        (
            slurm_runtime.TemporaryCancelError("controller unavailable"),
            BulkCancelOutcome.TEMPORARY_FAILURE,
        ),
        (
            slurm_runtime.CancelRejectedError("permission denied"),
            BulkCancelOutcome.REJECTED,
        ),
    ],
)
def test_cancel_executor_records_failed_dispatch_outcomes(
    tmp_path: Path,
    monkeypatch,
    error: BaseException,
    expected_outcome: BulkCancelOutcome,
):
    registry = _submitted_registry(tmp_path)
    registry.request_cancel("job-1", requested_by="operator", reason="stop test")
    runtime = _CancelRuntimeStub(error=error)
    _patch_slurm_target(monkeypatch)
    _patch_scheduler_rows(monkeypatch, {})
    monkeypatch.setattr(mod, "SlurmRuntime", lambda: runtime)

    outcome = asyncio.run(
        mod.execute_cancel_request(
            registry=registry,
            job_key="job-1",
            hpc_profile_block="hpc-slurm",
        )
    )

    assert outcome == expected_outcome
    assert len(runtime.cancel_calls) == 1
    record = registry.get_job("job-1")
    assert record.status == BulkJobStatus.AWAITING_OPERATOR
    assert record.cancel_outcome == expected_outcome
    assert record.cancel_last_error == str(error)


def test_cancelled_cancel_executor_preserves_dispatch_claim(tmp_path: Path, monkeypatch):
    registry = _submitted_registry(tmp_path)
    registry.request_cancel("job-1", requested_by="operator", reason="stop test")
    runtime = _CancelRuntimeStub(error=asyncio.CancelledError())
    _patch_slurm_target(monkeypatch)
    _patch_scheduler_rows(monkeypatch, {})
    monkeypatch.setattr(mod, "SlurmRuntime", lambda: runtime)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            mod.execute_cancel_request(
                registry=registry,
                job_key="job-1",
                hpc_profile_block="hpc-slurm",
            )
        )

    second = asyncio.run(
        mod.execute_cancel_request(
            registry=registry,
            job_key="job-1",
            hpc_profile_block="hpc-slurm",
        )
    )

    assert second == BulkCancelOutcome.DISPATCHING
    assert len(runtime.cancel_calls) == 1
    record = registry.get_job("job-1")
    assert record.cancel_attempts == 1
    assert record.cancel_outcome == BulkCancelOutcome.DISPATCHING


def test_pending_cancel_intent_finishes_without_scheduler_id(tmp_path: Path, monkeypatch):
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")
    registry.upsert_jobs([BulkJobSpec(job_key="job-1", work_dir=tmp_path / "job-1")])
    registry.request_cancel("job-1", requested_by="operator", reason="do not submit")

    class _UnexpectedRuntime:
        def __init__(self) -> None:
            raise AssertionError("SlurmRuntime must not be constructed for a pending job")

    monkeypatch.setattr(mod, "SlurmRuntime", _UnexpectedRuntime)

    outcome = asyncio.run(
        mod.execute_cancel_request(
            registry=registry,
            job_key="job-1",
            hpc_profile_block="hpc-slurm",
        )
    )

    assert outcome == BulkCancelOutcome.NOT_SUBMITTED
    record = registry.get_job("job-1")
    assert record.status == BulkJobStatus.CANCELLED
    assert record.scheduler_job_id is None
    assert record.cancel_attempts == 0


def test_prepared_cancel_without_job_id_waits_for_recovery(tmp_path: Path, monkeypatch):
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")
    registry.upsert_jobs(
        [
            BulkJobSpec(
                job_key="job-1",
                work_dir=tmp_path / "job-1",
                spec_hash="spec-v1",
                job_name="qcsc-job-1",
                job_comment="qcsc-prefect-slurm-identity-v1:sha256:abc",
            )
        ]
    )
    assert registry.claim_prepared(
        job_key="job-1",
        spec_hash="spec-v1",
        job_name="qcsc-job-1",
        job_comment="qcsc-prefect-slurm-identity-v1:sha256:abc",
    )
    registry.request_cancel("job-1", requested_by="operator", reason="stop if found")

    outcome = asyncio.run(
        mod.execute_cancel_request(
            registry=registry,
            job_key="job-1",
            hpc_profile_block="hpc-slurm",
        )
    )

    assert outcome is None
    record = registry.get_job("job-1")
    assert record.status == BulkJobStatus.PREPARED
    assert record.cancel_outcome is None
    assert record.cancel_attempts == 0


def test_bulk_runner_finishes_pending_cancel_without_queue_or_submit(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    job = BulkJobSpec(job_key="job-1", work_dir=tmp_path / "job-1")
    registry = BulkJobRegistry(registry_path)
    registry.upsert_jobs([job])
    registry.request_cancel("job-1", requested_by="operator", reason="do not submit")

    async def keep_registered_specs(**kwargs: Any) -> list[BulkJobSpec]:
        return list(kwargs["jobs"])

    async def unexpected_probe(**_kwargs: Any):
        raise AssertionError("queue probing must not run for a pre-submit cancellation")

    monkeypatch.setattr(mod, "_resolve_registered_bulk_spec_hashes", keep_registered_specs)
    monkeypatch.setattr(mod, "_resolve_default_bulk_queue_probe", unexpected_probe)

    result = asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=[job],
            command_block="unused",
            execution_profile_block="unused",
            hpc_profile_block="unused",
            registry_path=registry_path,
        )
    )

    assert result.cancelled == 1
    record = BulkJobRegistry(registry_path).get_job("job-1")
    assert record.status == BulkJobStatus.CANCELLED
    assert record.cancel_outcome == BulkCancelOutcome.NOT_SUBMITTED


def test_bulk_runner_returns_ambiguous_cancel_dispatch_for_operator(tmp_path: Path):
    registry_path = tmp_path / "bulk.sqlite"
    registry = BulkJobRegistry(registry_path)
    registry.upsert_jobs([BulkJobSpec(job_key="job-1", work_dir=tmp_path / "job-1")])
    registry.mark_submitted("job-1", "12345")
    registry.request_cancel("job-1", requested_by="operator", reason="stop")
    assert registry.claim_cancel_dispatch("job-1")

    result = asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=[],
            command_block="unused",
            execution_profile_block="unused",
            hpc_profile_block="unused",
            registry_path=registry_path,
        )
    )

    assert result.operator_action_required_jobs == ["job-1"]
    record = BulkJobRegistry(registry_path).get_job("job-1")
    assert record.status == BulkJobStatus.SUBMITTED
    assert record.cancel_outcome == BulkCancelOutcome.DISPATCHING
