from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from qcsc_prefect_executor import from_blocks as mod
from qcsc_prefect_executor.bulk.exceptions import (
    QueueFullError,
    SubmitError,
    TemporarySubmitError,
)
from qcsc_prefect_executor.bulk.models import BulkJobSpec, BulkJobStatus
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry

FUGAKU_HISTORY_VERBOSE_OUTPUT = (
    "JOB_ID     JOB_NAME   MD ST  USER     GROUP    START_DATE      "
    "ELAPSE_TIM ELAPSE_LIM            NODE_REQUIRE    VNODE  CORE "
    "V_MEM        V_POL E_POL RANK      LST EC  PC  SN PRI ACCEPT         "
    "RSC_GRP  REASON\n"
    "49047829   lucj-qpy-b NM EXT u13450   ra010014 06/01 15:03:44  "
    "0000:08:27 0000:15:00            1               -      -    "
    "-            -     -     bychip    RNO 0   0   0  127 "
    "06/01 15:03:14 small    -\n"
    "49047939   lucj-qpy-b NM EXT u13450   ra010014 06/01 15:22:46  "
    "0000:15:02 0000:15:00            1               -      -    "
    "-            -     -     bychip    RNO 0   11  24 127 "
    "06/01 15:22:16 small    ELAPSE LIMIT EXC\n"
)


class _CommandBlockStub:
    command_name = "bulk-command"
    executable_key = "bulk_executable"
    default_args: list[str] = []


class _ExecutionProfileBlockStub:
    profile_name = "bulk-profile"
    command_name = "bulk-command"
    resource_class = "cpu"
    num_nodes = 1
    mpiprocs = 1
    ompthreads = None
    walltime = "00:05:00"
    launcher = "single"
    mpi_options: list[str] = []
    modules: list[str] = []
    pre_commands: list[str] = []
    environments: dict[str, str] = {}


class _HPCProfileBlockStub:
    def __init__(self, hpc_target: str = "slurm") -> None:
        self.hpc_target = hpc_target
        self.queue_cpu = "compute"
        self.queue_gpu = "gpu"
        self.project_cpu = ""
        self.project_gpu = ""
        self.executable_map = {"bulk_executable": "/bin/echo"}
        self.slurm_qpu = None
        self.gfscache = None
        self.spack_modules: list[str] = []
        self.mpi_options_for_pjm: list[str] = []
        self.pjm_resources: list[str] = []


def _patch_block_loading(monkeypatch, *, hpc_target: str = "slurm") -> None:
    async def fake_command_load(_name: str):
        return _CommandBlockStub()

    async def fake_profile_load(_name: str):
        return _ExecutionProfileBlockStub()

    async def fake_hpc_load(_name: str):
        return _HPCProfileBlockStub(hpc_target=hpc_target)

    class _CmdAPI:
        load = staticmethod(fake_command_load)

    class _ProfileAPI:
        load = staticmethod(fake_profile_load)

    class _HpcAPI:
        load = staticmethod(fake_hpc_load)

    monkeypatch.setattr(mod, "CommandBlock", _CmdAPI)
    monkeypatch.setattr(mod, "ExecutionProfileBlock", _ProfileAPI)
    monkeypatch.setattr(mod, "HPCProfileBlock", _HpcAPI)


class _SubmitRuntimeStub:
    def __init__(self, *, job_id: str = "12345", error: Exception | None = None) -> None:
        self.job_id = job_id
        self.error = error
        self.submit_calls: list[dict[str, Any]] = []
        self.wait_calls = 0

    async def submit(self, script_path: Path, *, cwd: Path | None = None):
        self.submit_calls.append({"script_path": script_path, "cwd": cwd})
        if self.error is not None:
            raise self.error

        class _SubmitResult:
            def __init__(self, job_id: str) -> None:
                self.job_id = job_id

        return _SubmitResult(self.job_id)

    async def wait_final_status(self, *args, **kwargs):
        self.wait_calls += 1
        raise AssertionError("submit_job_from_blocks must not wait for completion")


def _patch_slurm_runtime(monkeypatch, runtime: _SubmitRuntimeStub) -> None:
    monkeypatch.setattr(mod, "SlurmRuntime", lambda: runtime)


def test_submit_job_from_blocks_records_submitted_with_scheduler_job_id(
    tmp_path: Path, monkeypatch
):
    _patch_block_loading(monkeypatch)
    runtime = _SubmitRuntimeStub(job_id="12345")
    _patch_slurm_runtime(monkeypatch, runtime)
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")

    result = asyncio.run(
        mod.submit_job_from_blocks(
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            work_dir=tmp_path / "job-1",
            job_key="job-1",
            command_args={"input": "a.dat"},
            registry=registry,
        )
    )

    record = registry.get_job("job-1")
    assert result.scheduler_job_id == "12345"
    assert result.status == BulkJobStatus.SUBMITTED
    assert record is not None
    assert record.status == BulkJobStatus.SUBMITTED
    assert record.scheduler_job_id == "12345"
    assert record.command_args == {"input": "a.dat"}


def test_submit_job_from_blocks_does_not_wait_for_completion(tmp_path: Path, monkeypatch):
    _patch_block_loading(monkeypatch)
    runtime = _SubmitRuntimeStub(job_id="12345")
    _patch_slurm_runtime(monkeypatch, runtime)

    asyncio.run(
        mod.submit_job_from_blocks(
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            work_dir=tmp_path / "job-1",
            job_key="job-1",
        )
    )

    assert len(runtime.submit_calls) == 1
    assert runtime.wait_calls == 0


def test_queue_full_is_recorded_as_submit_deferred_not_failed(tmp_path: Path, monkeypatch):
    _patch_block_loading(monkeypatch)
    runtime = _SubmitRuntimeStub(error=QueueFullError("queue full"))
    _patch_slurm_runtime(monkeypatch, runtime)
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")

    try:
        asyncio.run(
            mod.submit_job_from_blocks(
                command_block="cmd",
                execution_profile_block="exec",
                hpc_profile_block="hpc",
                work_dir=tmp_path / "job-1",
                job_key="job-1",
                registry=registry,
            )
        )
    except QueueFullError:
        pass
    else:
        raise AssertionError("Expected QueueFullError")

    record = registry.get_job("job-1")
    assert record is not None
    assert record.status == BulkJobStatus.SUBMIT_DEFERRED
    assert record.status != BulkJobStatus.FAILED


def test_queue_full_raises_queue_full_error(tmp_path: Path, monkeypatch):
    _patch_block_loading(monkeypatch)
    runtime = _SubmitRuntimeStub(error=RuntimeError("ru-accept job limit exceeded"))
    _patch_slurm_runtime(monkeypatch, runtime)

    try:
        asyncio.run(
            mod.submit_job_from_blocks(
                command_block="cmd",
                execution_profile_block="exec",
                hpc_profile_block="hpc",
                work_dir=tmp_path / "job-1",
                job_key="job-1",
            )
        )
    except QueueFullError as exc:
        assert "ru-accept" in str(exc)
    else:
        raise AssertionError("Expected QueueFullError")


def test_temporary_submit_error_is_recorded_as_submit_deferred(tmp_path: Path, monkeypatch):
    _patch_block_loading(monkeypatch)
    runtime = _SubmitRuntimeStub(error=TemporarySubmitError("scheduler temporarily unavailable"))
    _patch_slurm_runtime(monkeypatch, runtime)
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")

    try:
        asyncio.run(
            mod.submit_job_from_blocks(
                command_block="cmd",
                execution_profile_block="exec",
                hpc_profile_block="hpc",
                work_dir=tmp_path / "job-1",
                job_key="job-1",
                registry=registry,
            )
        )
    except TemporarySubmitError:
        pass
    else:
        raise AssertionError("Expected TemporarySubmitError")

    record = registry.get_job("job-1")
    assert record is not None
    assert record.status == BulkJobStatus.SUBMIT_DEFERRED


def test_unrecoverable_submit_error_is_recorded_as_failed(tmp_path: Path, monkeypatch):
    _patch_block_loading(monkeypatch)
    runtime = _SubmitRuntimeStub(error=RuntimeError("invalid account"))
    _patch_slurm_runtime(monkeypatch, runtime)
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")

    try:
        asyncio.run(
            mod.submit_job_from_blocks(
                command_block="cmd",
                execution_profile_block="exec",
                hpc_profile_block="hpc",
                work_dir=tmp_path / "job-1",
                job_key="job-1",
                registry=registry,
            )
        )
    except SubmitError:
        pass
    else:
        raise AssertionError("Expected SubmitError")

    record = registry.get_job("job-1")
    assert record is not None
    assert record.status == BulkJobStatus.FAILED


def test_monitor_jobs_many_maps_scheduler_states(monkeypatch):
    _patch_block_loading(monkeypatch)

    async def fake_query_scheduler_statuses(*, hpc_target: str, scheduler_job_ids: list[str]):
        assert hpc_target == "slurm"
        return {
            "1": {"State": "PENDING"},
            "2": {"State": "RUNNING"},
            "3": {"State": "COMPLETED"},
            "4": {"State": "FAILED"},
            "5": {"State": "CANCELLED"},
        }

    monkeypatch.setattr(mod, "_query_scheduler_statuses", fake_query_scheduler_statuses)

    statuses = asyncio.run(
        mod.monitor_jobs_many(
            hpc_profile_block="hpc",
            scheduler_job_ids=["1", "2", "3", "4", "5"],
        )
    )

    assert statuses == {
        "1": BulkJobStatus.QUEUED,
        "2": BulkJobStatus.RUNNING,
        "3": BulkJobStatus.SUCCEEDED,
        "4": BulkJobStatus.FAILED,
        "5": BulkJobStatus.CANCELLED,
    }


def test_fugaku_history_verbose_rows_map_completed_jobs():
    rows = mod._parse_fugaku_pjstat_rows(FUGAKU_HISTORY_VERBOSE_OUTPUT)

    assert rows["49047829"]["EC"] == "0"
    assert rows["49047829"]["REASON"] == "-"
    assert mod._bulk_status_from_scheduler_row(
        "fugaku", rows["49047829"]
    ) == BulkJobStatus.SUCCEEDED
    assert rows["49047939"]["PC"] == "11"
    assert rows["49047939"]["REASON"] == "ELAPSE LIMIT EXC"
    assert mod._bulk_status_from_scheduler_row(
        "fugaku", rows["49047939"]
    ) == BulkJobStatus.FAILED


def test_fugaku_ext_without_exit_code_is_unknown():
    assert mod._bulk_status_from_scheduler_row(
        "fugaku", {"JOB_ID": "49074516", "ST": "EXT"}
    ) == BulkJobStatus.UNKNOWN


def test_query_fugaku_scheduler_statuses_reads_history_for_missing_jobs(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append(args)
        if args == ("pjstat", "-v"):
            return "JOB_ID     JOB_NAME   MD ST  USER     GROUP\n"
        if args == ("pjstat", "-v", "-H"):
            return FUGAKU_HISTORY_VERBOSE_OUTPUT
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(mod.fugaku_runtime, "run_command", fake_run_command)

    rows = asyncio.run(
        mod._query_scheduler_statuses(
            hpc_target="fugaku",
            scheduler_job_ids=["49047829", "49047939"],
        )
    )

    assert calls == [("pjstat", "-v"), ("pjstat", "-v", "-H")]
    assert set(rows) == {"49047829", "49047939"}
    assert mod._bulk_status_from_scheduler_row(
        "fugaku", rows["49047829"]
    ) == BulkJobStatus.SUCCEEDED


def test_monitor_jobs_many_updates_multiple_jobs_in_one_call(monkeypatch):
    _patch_block_loading(monkeypatch)
    calls: list[list[str]] = []

    async def fake_query_scheduler_statuses(*, hpc_target: str, scheduler_job_ids: list[str]):
        calls.append(scheduler_job_ids)
        return {
            "1": {"State": "RUNNING"},
            "2": {"State": "COMPLETED"},
        }

    monkeypatch.setattr(mod, "_query_scheduler_statuses", fake_query_scheduler_statuses)

    statuses = asyncio.run(
        mod.monitor_jobs_many(
            hpc_profile_block="hpc",
            scheduler_job_ids=["1", "2"],
        )
    )

    assert calls == [["1", "2"]]
    assert statuses["1"] == BulkJobStatus.RUNNING
    assert statuses["2"] == BulkJobStatus.SUCCEEDED


def test_monitor_jobs_many_updates_registry_if_provided(tmp_path: Path, monkeypatch):
    _patch_block_loading(monkeypatch)

    async def fake_query_scheduler_statuses(*, hpc_target: str, scheduler_job_ids: list[str]):
        return {
            "1": {"State": "RUNNING"},
            "2": {"State": "COMPLETED"},
        }

    monkeypatch.setattr(mod, "_query_scheduler_statuses", fake_query_scheduler_statuses)
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")
    registry.upsert_jobs(
        [
            BulkJobSpec(job_key="job-1", work_dir=tmp_path / "job-1"),
            BulkJobSpec(job_key="job-2", work_dir=tmp_path / "job-2"),
        ]
    )
    registry.mark_submitted("job-1", "1")
    registry.mark_submitted("job-2", "2")

    asyncio.run(
        mod.monitor_jobs_many(
            hpc_profile_block="hpc",
            scheduler_job_ids=["1", "2"],
            registry=registry,
        )
    )

    job_1 = registry.get_job("job-1")
    job_2 = registry.get_job("job-2")
    assert job_1 is not None
    assert job_2 is not None
    assert job_1.status == BulkJobStatus.RUNNING
    assert job_2.status == BulkJobStatus.SUCCEEDED


def test_disappeared_job_with_valid_expected_outputs_becomes_succeeded(
    tmp_path: Path, monkeypatch
):
    _patch_block_loading(monkeypatch)

    async def fake_query_scheduler_statuses(*, hpc_target: str, scheduler_job_ids: list[str]):
        return {}

    monkeypatch.setattr(mod, "_query_scheduler_statuses", fake_query_scheduler_statuses)
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")
    work_dir = tmp_path / "job-1"
    registry.upsert_jobs(
        [
            BulkJobSpec(
                job_key="job-1",
                work_dir=work_dir,
                expected_outputs=[Path("done.txt")],
            )
        ]
    )
    registry.mark_submitted("job-1", "1")
    work_dir.mkdir()
    (work_dir / "done.txt").write_text("ok")

    statuses = asyncio.run(
        mod.monitor_jobs_many(
            hpc_profile_block="hpc",
            scheduler_job_ids=["1"],
            registry=registry,
        )
    )

    record = registry.get_job("job-1")
    assert statuses["1"] == BulkJobStatus.SUCCEEDED
    assert record is not None
    assert record.status == BulkJobStatus.SUCCEEDED


def test_disappeared_job_without_success_evidence_becomes_unknown(
    tmp_path: Path, monkeypatch
):
    _patch_block_loading(monkeypatch)

    async def fake_query_scheduler_statuses(*, hpc_target: str, scheduler_job_ids: list[str]):
        return {}

    monkeypatch.setattr(mod, "_query_scheduler_statuses", fake_query_scheduler_statuses)
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")
    registry.upsert_jobs([BulkJobSpec(job_key="job-1", work_dir=tmp_path / "job-1")])
    registry.mark_submitted("job-1", "1")

    statuses = asyncio.run(
        mod.monitor_jobs_many(
            hpc_profile_block="hpc",
            scheduler_job_ids=["1"],
            registry=registry,
        )
    )

    record = registry.get_job("job-1")
    assert statuses["1"] == BulkJobStatus.UNKNOWN
    assert record is not None
    assert record.status == BulkJobStatus.UNKNOWN
