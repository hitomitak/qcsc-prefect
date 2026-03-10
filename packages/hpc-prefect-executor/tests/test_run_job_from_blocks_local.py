from __future__ import annotations

import asyncio
from pathlib import Path

from hpc_prefect_executor import from_blocks as mod


class _CommandBlockStub:
    def __init__(self, command_name: str, executable_key: str, default_args: list[str]) -> None:
        self.command_name = command_name
        self.executable_key = executable_key
        self.default_args = default_args


class _ExecutionProfileBlockStub:
    def __init__(
        self,
        *,
        profile_name: str,
        command_name: str,
        resource_class: str = "cpu",
        num_nodes: int = 2,
        mpiprocs: int = 5,
        ompthreads: int = 1,
        walltime: str = "00:10:00",
        launcher: str = "mpiexec.hydra",
        mpi_options: list[str] | None = None,
        modules: list[str] | None = None,
        environments: dict[str, str] | None = None,
    ) -> None:
        self.profile_name = profile_name
        self.command_name = command_name
        self.resource_class = resource_class
        self.num_nodes = num_nodes
        self.mpiprocs = mpiprocs
        self.ompthreads = ompthreads
        self.walltime = walltime
        self.launcher = launcher
        self.mpi_options = mpi_options or []
        self.modules = modules or []
        self.environments = environments or {}


class _HPCProfileBlockStub:
    def __init__(
        self,
        *,
        hpc_target: str,
        executable_map: dict[str, str],
        queue_cpu: str = "regular-c",
        queue_gpu: str = "regular-g",
        project_cpu: str = "gz00",
        project_gpu: str = "gz00",
        gfscache: str | None = None,
        spack_modules: list[str] | None = None,
        mpi_options_for_pjm: list[str] | None = None,
        pjm_resources: list[str] | None = None,
    ) -> None:
        self.hpc_target = hpc_target
        self.executable_map = executable_map
        self.queue_cpu = queue_cpu
        self.queue_gpu = queue_gpu
        self.project_cpu = project_cpu
        self.project_gpu = project_gpu
        self.gfscache = gfscache
        self.spack_modules = spack_modules or []
        self.mpi_options_for_pjm = mpi_options_for_pjm or []
        self.pjm_resources = pjm_resources or []


def _patch_block_loading(monkeypatch, command, profile, hpc):
    async def fake_command_load(_name: str):
        return command

    async def fake_profile_load(_name: str):
        return profile

    async def fake_hpc_load(_name: str):
        return hpc

    class _CmdAPI:
        load = staticmethod(fake_command_load)

    class _ProfileAPI:
        load = staticmethod(fake_profile_load)

    class _HpcAPI:
        load = staticmethod(fake_hpc_load)

    monkeypatch.setattr(mod, "CommandBlock", _CmdAPI)
    monkeypatch.setattr(mod, "ExecutionProfileBlock", _ProfileAPI)
    monkeypatch.setattr(mod, "HPCProfileBlock", _HpcAPI)


def test_run_job_from_blocks_dispatches_to_miyabi(monkeypatch, tmp_path: Path):
    command = _CommandBlockStub("bitcount-hist", "bitcount_hist", [])
    profile = _ExecutionProfileBlockStub(profile_name="bitcount-mpi", command_name="bitcount-hist")
    hpc = _HPCProfileBlockStub(
        hpc_target="miyabi",
        executable_map={"bitcount_hist": "/work/gz00/z99999/get_counts_hist"},
    )
    _patch_block_loading(monkeypatch, command, profile, hpc)

    captured: dict[str, object] = {}

    class _MiyabiResult:
        def __init__(self) -> None:
            self.job_id = "12345.miyabi"
            self.exit_status = 0

    async def fake_run_miyabi_job(**kwargs):
        captured.update(kwargs)
        return _MiyabiResult()

    async def fake_run_fugaku_job(**kwargs):
        raise AssertionError("run_fugaku_job should not be called in this test")

    monkeypatch.setattr(mod, "run_miyabi_job", fake_run_miyabi_job)
    monkeypatch.setattr(mod, "run_fugaku_job", fake_run_fugaku_job)

    result = asyncio.run(
        mod.run_job_from_blocks(
            command_block_name="cmd",
            execution_profile_block_name="exec",
            hpc_profile_block_name="hpc",
            work_dir=tmp_path,
            script_filename="job.pbs",
            metrics_artifact_key="miyabi-metrics",
        )
    )

    assert result.job_id == "12345.miyabi"
    assert captured["req"].queue_name == "regular-c"
    assert captured["req"].project == "gz00"
    assert captured["req"].executable == "/work/gz00/z99999/get_counts_hist"
    assert captured["metrics_artifact_key"] == "miyabi-metrics"


def test_run_job_from_blocks_dispatches_to_fugaku(monkeypatch, tmp_path: Path):
    command = _CommandBlockStub("bitcount-hist", "bitcount_hist", [])
    profile = _ExecutionProfileBlockStub(profile_name="bitcount-mpi", command_name="bitcount-hist")
    hpc = _HPCProfileBlockStub(
        hpc_target="fugaku",
        executable_map={"bitcount_hist": "/vol0001/home/z99999/get_counts_hist"},
        queue_cpu="small",
        project_cpu="hp200999",
        gfscache="/vol0002",
        spack_modules=["fjmpi"],
        mpi_options_for_pjm=["max-proc-per-node=48"],
        pjm_resources=["freq=2000,eco_state=2"],
    )
    _patch_block_loading(monkeypatch, command, profile, hpc)

    captured: dict[str, object] = {}

    class _FugakuResult:
        def __init__(self) -> None:
            self.job_id = "900001"
            self.exit_status = 0
            self.state = "EXT"

    async def fake_run_miyabi_job(**kwargs):
        raise AssertionError("run_miyabi_job should not be called in this test")

    async def fake_run_fugaku_job(**kwargs):
        captured.update(kwargs)
        return _FugakuResult()

    monkeypatch.setattr(mod, "run_miyabi_job", fake_run_miyabi_job)
    monkeypatch.setattr(mod, "run_fugaku_job", fake_run_fugaku_job)

    result = asyncio.run(
        mod.run_job_from_blocks(
            command_block_name="cmd",
            execution_profile_block_name="exec",
            hpc_profile_block_name="hpc",
            work_dir=tmp_path,
            script_filename="job.pjm",
            metrics_artifact_key="fugaku-metrics",
        )
    )

    assert result.job_id == "900001"
    assert captured["req"].queue_name == "small"
    assert captured["req"].project == "hp200999"
    assert captured["req"].executable == "/vol0001/home/z99999/get_counts_hist"
    assert captured["req"].gfscache == "/vol0002"
    assert captured["req"].spack_modules == ["fjmpi"]
    assert captured["req"].mpi_options_for_pjm == ["max-proc-per-node=48"]
    assert captured["req"].pjm_resources == ["freq=2000,eco_state=2"]
    assert captured["metrics_artifact_key"] == "fugaku-metrics"
