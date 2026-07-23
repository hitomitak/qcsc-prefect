from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from qcsc_prefect_adapters.slurm.builder import SlurmJobRequest
from qcsc_prefect_core.models.execution_profile import ExecutionProfile
from qcsc_prefect_executor import from_blocks as mod
from qcsc_prefect_executor.bulk.exceptions import (
    SpecHashMismatchError,
    TemporarySubmitError,
)
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry
from qcsc_prefect_executor.bulk.spec_hash import (
    BULK_SPEC_HASH_SCHEMA_VERSION,
    build_bulk_spec_hash,
    canonical_bulk_spec_json,
)


def _prepared(
    tmp_path: Path,
    *,
    profile: ExecutionProfile | None = None,
    request: SlurmJobRequest | None = None,
    work_dir_name: str = "attempt-1",
) -> mod._PreparedBlockJob:
    return mod._PreparedBlockJob(
        submission_target=mod.SubmissionTarget(
            hpc_target="slurm",
            queue_name="compute",
            project="project-a",
        ),
        work_dir=tmp_path / work_dir_name,
        script_filename="job-1.slurm",
        exec_profile=profile
        or ExecutionProfile(
            command_key="solver",
            num_nodes=2,
            mpiprocs=4,
            ompthreads=8,
            walltime="00:30:00",
            launcher="srun",
            mpi_options=["--cpu-bind=cores"],
            modules=["python/3.12"],
            pre_commands=["ulimit -s unlimited"],
            environments={"OMP_NUM_THREADS": "8"},
            arguments=["--input", "fragment.json"],
        ),
        req=request
        or SlurmJobRequest(
            partition="compute",
            account="project-a",
            executable="/shared/bin/solver",
            qpu=None,
            memory="64G",
            ntasks=8,
        ),
    )


def test_canonical_hash_normalizes_mapping_order_paths_and_sequences():
    first = {
        "arguments": {"output": Path("results/out.json"), "indices": (1, 2, 3)},
        "resources": {"nodes": 2, "memory": "64G"},
    }
    second = {
        "resources": {"memory": "64G", "nodes": 2},
        "arguments": {"indices": [1, 2, 3], "output": "results/out.json"},
    }

    assert canonical_bulk_spec_json(first) == canonical_bulk_spec_json(second)
    assert build_bulk_spec_hash(first) == build_bulk_spec_hash(second)
    assert build_bulk_spec_hash(first).startswith(f"{BULK_SPEC_HASH_SCHEMA_VERSION}:sha256:")


def test_named_command_arguments_are_rendered_in_stable_key_order():
    assert mod._command_args_to_user_args(
        {"zeta": 3, "alpha": Path("input.json"), "middle": True}
    ) == [
        "--alpha",
        "input.json",
        "--middle",
        "--zeta",
        "3",
    ]


def test_canonical_payload_never_contains_plaintext_dynamic_values():
    secret = "top-secret-token-value"
    payload_json = canonical_bulk_spec_json(
        {
            "command": ["/shared/bin/solver", "--token", secret],
            "environment": {"API_TOKEN": secret},
            "input_digest": "input-digest-abc",
        }
    )

    assert secret not in payload_json
    assert "/shared/bin/solver" not in payload_json
    assert "input-digest-abc" not in payload_json


def test_resolved_hash_covers_resources_environment_and_caller_digests(tmp_path: Path):
    prepared = _prepared(tmp_path)
    baseline = mod._resolved_bulk_spec_hash(
        prepared,
        input_digest="input-a",
        code_digest="code-a",
        environment_digest="env-a",
    )

    assert (
        mod._resolved_bulk_spec_hash(
            _prepared(tmp_path, work_dir_name="attempt-999"),
            input_digest="input-a",
            code_digest="code-a",
            environment_digest="env-a",
        )
        == baseline
    )
    assert (
        mod._resolved_bulk_spec_hash(
            _prepared(
                tmp_path,
                profile=replace(prepared.exec_profile, walltime="01:00:00"),
            ),
            input_digest="input-a",
            code_digest="code-a",
            environment_digest="env-a",
        )
        != baseline
    )
    assert (
        mod._resolved_bulk_spec_hash(
            _prepared(
                tmp_path,
                profile=replace(
                    prepared.exec_profile,
                    environments={"OMP_NUM_THREADS": "16"},
                ),
            ),
            input_digest="input-a",
            code_digest="code-a",
            environment_digest="env-a",
        )
        != baseline
    )
    assert (
        mod._resolved_bulk_spec_hash(
            _prepared(
                tmp_path,
                request=replace(prepared.req, memory="128G"),
            ),
            input_digest="input-a",
            code_digest="code-a",
            environment_digest="env-a",
        )
        != baseline
    )
    assert (
        mod._resolved_bulk_spec_hash(
            prepared,
            input_digest="input-b",
            code_digest="code-a",
            environment_digest="env-a",
        )
        != baseline
    )


class _CommandBlockStub:
    command_name = "solver"
    executable_key = "solver"
    default_args = ["--mode", "production"]


class _ExecutionProfileBlockStub:
    command_name = "solver"
    resource_class = "cpu"
    num_nodes = 1
    mpiprocs = 1
    ompthreads = 2
    walltime = "00:05:00"
    launcher = "srun"
    mpi_options: list[str] = []
    modules = ["python/3.12"]
    pre_commands: list[str] = []
    environments = {"OMP_NUM_THREADS": "2"}


class _HPCProfileBlockStub:
    hpc_target = "slurm"
    queue_cpu = "compute"
    queue_gpu = "gpu"
    project_cpu = "project-a"
    project_gpu = "project-a"
    executable_map = {"solver": "/shared/bin/solver"}
    slurm_qpu = None
    slurm_memory = "64G"
    slurm_ntasks = 2


class _SubmitRuntimeStub:
    def __init__(self) -> None:
        self.error: Exception | None = TemporarySubmitError("scheduler busy")
        self.calls = 0

    async def submit(self, *_args, **_kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error

        class _Result:
            job_id = "12345"

        return _Result()


def _patch_submit_dependencies(monkeypatch, runtime: _SubmitRuntimeStub) -> None:
    async def load_command(_name: str):
        return _CommandBlockStub()

    async def load_execution(_name: str):
        return _ExecutionProfileBlockStub()

    async def load_hpc(_name: str):
        return _HPCProfileBlockStub()

    monkeypatch.setattr(
        mod,
        "CommandBlock",
        type("_CommandAPI", (), {"load": staticmethod(load_command)}),
    )
    monkeypatch.setattr(
        mod,
        "ExecutionProfileBlock",
        type("_ExecutionAPI", (), {"load": staticmethod(load_execution)}),
    )
    monkeypatch.setattr(
        mod,
        "HPCProfileBlock",
        type("_HPCAPI", (), {"load": staticmethod(load_hpc)}),
    )
    monkeypatch.setattr(mod, "SlurmRuntime", lambda: runtime)


def test_submit_retry_accepts_same_hash_and_rejects_changed_resolved_spec(
    tmp_path: Path,
    monkeypatch,
):
    runtime = _SubmitRuntimeStub()
    _patch_submit_dependencies(monkeypatch, runtime)
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")
    kwargs = {
        "command_block": "command",
        "execution_profile_block": "execution",
        "hpc_profile_block": "hpc",
        "work_dir": tmp_path / "job-1",
        "job_key": "job-1",
        "command_args": {"input": Path("fragment.json"), "count": 2},
        "input_digest": "input-a",
        "code_digest": "code-a",
        "environment_digest": "environment-a",
        "registry": registry,
    }

    with pytest.raises(TemporarySubmitError):
        asyncio.run(mod.submit_job_from_blocks(**kwargs))

    deferred = registry.get_job("job-1")
    assert deferred is not None
    assert deferred.spec_hash is not None
    original_hash = deferred.spec_hash

    runtime.error = None
    result = asyncio.run(mod.submit_job_from_blocks(**kwargs))
    assert result.scheduler_job_id == "12345"
    assert runtime.calls == 2

    _ExecutionProfileBlockStub.walltime = "00:10:00"
    try:
        with pytest.raises(SpecHashMismatchError):
            asyncio.run(mod.submit_job_from_blocks(**kwargs))
    finally:
        _ExecutionProfileBlockStub.walltime = "00:05:00"

    unchanged = registry.get_job("job-1")
    assert unchanged is not None
    assert unchanged.spec_hash == original_hash
    assert unchanged.command_args == {"count": 2, "input": "fragment.json"}
    assert runtime.calls == 2
