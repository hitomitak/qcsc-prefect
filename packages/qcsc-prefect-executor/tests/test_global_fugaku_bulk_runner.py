from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from qcsc_prefect_core.queue import QueueCapacity
from qcsc_prefect_executor.bulk import (
    BulkJobSpec,
    BulkJobStatus,
    BulkTickResult,
    GlobalFugakuBulkRunner,
)
from qcsc_prefect_executor.bulk import global_fugaku_runner as runner_mod
from qcsc_prefect_executor.bulk.exceptions import QueueFullError
from qcsc_prefect_executor.bulk.models import SubmittedJob
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry


class _FixedCapacityProbe:
    def __init__(self, available_slots: int) -> None:
        self.available_slots = available_slots
        self.calls = 0

    def get_capacity(self) -> QueueCapacity:
        self.calls += 1
        return QueueCapacity(
            max_active_jobs=self.available_slots,
            current_active_jobs=0,
            available_slots=self.available_slots,
            raw_output="fixed capacity",
        )


def _spec(
    tmp_path: Path,
    job_key: str,
    stage_id: str,
    *,
    expected_outputs: list[Path] | None = None,
) -> BulkJobSpec:
    return BulkJobSpec(
        job_key=job_key,
        stage_id=stage_id,
        work_dir=tmp_path / job_key,
        command_args={"job_key": job_key},
        expected_outputs=expected_outputs or [],
    )


def _install_single_submit_fakes(
    monkeypatch,
    *,
    submitted: list[str],
    submit_failures: dict[str, Exception] | None = None,
    mark_monitored_succeeded: bool = False,
) -> None:
    submit_failures = submit_failures or {}

    async def fake_submit_job_from_blocks(
        *,
        work_dir: Path,
        job_key: str,
        command_block: str,
        execution_profile_block: str,
        hpc_profile_block: str,
        command_args: dict[str, Any] | None = None,
        registry: BulkJobRegistry | None = None,
    ) -> SubmittedJob:
        if job_key in submit_failures:
            raise submit_failures[job_key]

        scheduler_job_id = f"sched-{job_key}"
        submitted.append(job_key)
        if registry is not None:
            registry.mark_submitted(job_key, scheduler_job_id)
        return SubmittedJob(
            job_key=job_key,
            scheduler_job_id=scheduler_job_id,
            status=BulkJobStatus.SUBMITTED,
            work_dir=work_dir,
        )

    async def fake_monitor_jobs_many(
        *,
        hpc_profile_block: str,
        scheduler_job_ids: list[str],
        registry: BulkJobRegistry | None = None,
    ) -> dict[str, BulkJobStatus]:
        if registry is not None and mark_monitored_succeeded:
            records_by_scheduler_id = {
                record.effective_scheduler_job_id: record
                for record in registry.get_all_jobs()
                if record.effective_scheduler_job_id
            }
            for scheduler_job_id in scheduler_job_ids:
                record = records_by_scheduler_id[scheduler_job_id]
                registry.mark_succeeded(record.job_key)
            return {
                scheduler_job_id: BulkJobStatus.SUCCEEDED
                for scheduler_job_id in scheduler_job_ids
            }
        return {
            scheduler_job_id: BulkJobStatus.SUBMITTED
            for scheduler_job_id in scheduler_job_ids
        }

    monkeypatch.setattr(runner_mod, "submit_job_from_blocks", fake_submit_job_from_blocks)
    monkeypatch.setattr(runner_mod, "monitor_jobs_many", fake_monitor_jobs_many)


def _runner(
    tmp_path: Path,
    *,
    queue_probe: _FixedCapacityProbe | None = None,
    initial_submit_count: int | None = 3,
    max_submit_per_refill: int = 2,
) -> GlobalFugakuBulkRunner:
    return GlobalFugakuBulkRunner(
        command_block="cmd",
        execution_profile_block="exec",
        hpc_profile_block="hpc",
        registry_path=tmp_path / "bulk.sqlite",
        queue_probe=queue_probe or _FixedCapacityProbe(10),
        initial_submit_count=initial_submit_count,
        max_submit_per_refill=max_submit_per_refill,
    )


def test_public_bulk_api_exports_global_fugaku_bulk_runner():
    assert callable(GlobalFugakuBulkRunner)
    assert BulkTickResult.__name__ == "BulkTickResult"


def test_tick_submits_initial_count_then_refill_count(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    _install_single_submit_fakes(monkeypatch, submitted=submitted)
    runner = _runner(tmp_path, initial_submit_count=3, max_submit_per_refill=2)
    runner.register_jobs([_spec(tmp_path, f"qpy-{index}", "qpy") for index in range(5)])

    first = asyncio.run(runner.tick())
    second = asyncio.run(runner.tick())

    assert [job.job_key for job in first.submitted] == ["qpy-0", "qpy-1", "qpy-2"]
    assert [job.job_key for job in second.submitted] == ["qpy-3", "qpy-4"]
    assert submitted == ["qpy-0", "qpy-1", "qpy-2", "qpy-3", "qpy-4"]
    assert runner.all_submitted("qpy") is True


def test_all_submitted_false_while_pending_or_deferred_remains(
    tmp_path: Path,
    monkeypatch,
):
    submitted: list[str] = []
    _install_single_submit_fakes(
        monkeypatch,
        submitted=submitted,
        submit_failures={"qpy-0": QueueFullError("queue full")},
    )
    runner = _runner(tmp_path, initial_submit_count=1, max_submit_per_refill=1)
    runner.register_jobs([_spec(tmp_path, f"qpy-{index}", "qpy") for index in range(2)])

    first = asyncio.run(runner.tick())
    second = asyncio.run(runner.tick())

    assert first.submitted == []
    assert [job.job_key for job in second.submitted] == ["qpy-1"]
    assert submitted == ["qpy-1"]
    assert runner.all_submitted("qpy") is False
    assert runner.status_counts("qpy") == {
        BulkJobStatus.SUBMIT_DEFERRED.value: 1,
        BulkJobStatus.SUBMITTED.value: 1,
    }


def test_register_trimsqd_jobs_later_and_tick_submits_them(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    _install_single_submit_fakes(monkeypatch, submitted=submitted)
    runner = _runner(tmp_path, initial_submit_count=2, max_submit_per_refill=2)
    runner.register_jobs([_spec(tmp_path, f"qpy-{index}", "qpy") for index in range(2)])

    asyncio.run(runner.tick())
    assert runner.all_submitted("qpy") is True

    runner.register_jobs([_spec(tmp_path, f"trim-{index}", "trim_sqd") for index in range(2)])
    trim_tick = asyncio.run(runner.tick())

    assert [job.job_key for job in trim_tick.submitted] == ["trim-0", "trim-1"]
    assert runner.all_submitted("trim_sqd") is True
    assert submitted == ["qpy-0", "qpy-1", "trim-0", "trim-1"]


def test_existing_succeeded_jobs_are_skipped(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    _install_single_submit_fakes(monkeypatch, submitted=submitted)
    runner = _runner(tmp_path, initial_submit_count=3, max_submit_per_refill=2)
    done_dir = tmp_path / "qpy-0"
    done_dir.mkdir()
    (done_dir / "done.marker").write_text("ok")
    runner.register_jobs(
        [
            _spec(
                tmp_path,
                "qpy-0",
                "qpy",
                expected_outputs=[Path("done.marker")],
            ),
            _spec(tmp_path, "qpy-1", "qpy"),
        ]
    )

    tick = asyncio.run(runner.tick())

    assert [job.job_key for job in tick.submitted] == ["qpy-1"]
    assert submitted == ["qpy-1"]
    assert runner.status_counts("qpy") == {
        BulkJobStatus.SUBMITTED.value: 1,
        BulkJobStatus.SUCCEEDED.value: 1,
    }


def test_queue_capacity_caps_tick_submit_count(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    _install_single_submit_fakes(monkeypatch, submitted=submitted)
    runner = _runner(
        tmp_path,
        queue_probe=_FixedCapacityProbe(2),
        initial_submit_count=5,
        max_submit_per_refill=5,
    )
    runner.register_jobs([_spec(tmp_path, f"qpy-{index}", "qpy") for index in range(5)])

    tick = asyncio.run(runner.tick())

    assert [job.job_key for job in tick.submitted] == ["qpy-0", "qpy-1"]
    assert submitted == ["qpy-0", "qpy-1"]
