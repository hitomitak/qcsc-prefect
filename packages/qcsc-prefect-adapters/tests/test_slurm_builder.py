from __future__ import annotations

from pathlib import Path

import pytest
from qcsc_prefect_adapters.slurm.builder import (
    DEFAULT_SLURM_JOB_NAME_MAX_LENGTH,
    SlurmJobRequest,
    build_slurm_job_identity,
    render_script,
)
from qcsc_prefect_core.models.execution_profile import ExecutionProfile


def test_render_slurm_script(tmp_path: Path):
    profile = ExecutionProfile(
        command_key="hello",
        num_nodes=2,
        mpiprocs=4,
        ompthreads=8,
        walltime="00:10:00",
        launcher="srun",
        mpi_options=["--cpu-bind=cores"],
        modules=["gcc"],
        pre_commands=["echo before-run"],
        environments={"OMP_NUM_THREADS": "8"},
        arguments=["--foo", "bar"],
    )
    req = SlurmJobRequest(
        partition="compute",
        account="proj01",
        qpu="a100",
        memory="5001G",
        ntasks=8,
        executable="/path/to/hello",
    )

    text = render_script(work_dir=tmp_path, exec_profile=profile, req=req)

    assert "#SBATCH --partition=compute" in text
    assert "#SBATCH --account=proj01" in text
    assert "#SBATCH --mem=5001G" in text
    assert "#SBATCH --nodes=2" in text
    assert "#SBATCH --ntasks=8" in text
    assert "#SBATCH --ntasks-per-node=4" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --time=00:10:00" in text
    assert "#SBATCH --qpu=a100" in text
    assert "module load gcc" in text
    assert "echo before-run" in text
    assert 'export OMP_NUM_THREADS="8"' in text
    assert 'QCSC_PREFECT_EXECUTABLE="/path/to/hello"' in text
    assert "QCSC Prefect preflight failed: executable '${QCSC_PREFECT_EXECUTABLE}'" in text
    assert 'srun --cpu-bind=cores "${QCSC_PREFECT_EXECUTABLE}" --foo bar' in text


def test_render_slurm_script_omits_optional_directives(tmp_path: Path):
    profile = ExecutionProfile(
        command_key="hello",
        num_nodes=1,
        launcher="single",
    )
    req = SlurmJobRequest(
        partition="compute",
        memory=None,
        executable="hello",
    )

    text = render_script(work_dir=tmp_path, exec_profile=profile, req=req)

    assert "#SBATCH --partition=compute" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --mem" not in text
    assert "#SBATCH --ntasks=" not in text
    assert "#SBATCH --ntasks-per-node" not in text
    assert "#SBATCH --account" not in text
    assert "#SBATCH --qpu" not in text
    assert "#SBATCH --job-name" not in text
    assert "#SBATCH --comment" not in text


def test_build_slurm_job_identity_is_deterministic_and_spec_specific() -> None:
    first = build_slurm_job_identity(
        job_key="campaign-42/target-001",
        spec_hash="qcsc-prefect-bulk-spec-v1:sha256:abc123",
    )
    same = build_slurm_job_identity(
        job_key="campaign-42/target-001",
        spec_hash="qcsc-prefect-bulk-spec-v1:sha256:abc123",
    )
    changed = build_slurm_job_identity(
        job_key="campaign-42/target-001",
        spec_hash="qcsc-prefect-bulk-spec-v1:sha256:def456",
    )

    assert first == same
    assert first != changed
    assert first.job_name.startswith("qcsc-campaign-42-target-001-")
    assert len(first.comment.rsplit(":", maxsplit=1)[-1]) == 64


def test_build_slurm_job_identity_keeps_digest_suffix_for_long_unicode_key() -> None:
    identity = build_slurm_job_identity(
        job_key="解析結果/very-long-logical-job-key_" * 20,
        spec_hash="qcsc-prefect-bulk-spec-v1:sha256:abcdef",
    )

    suffix = identity.comment.rsplit(":", maxsplit=1)[-1][:24]
    assert len(identity.job_name) == DEFAULT_SLURM_JOB_NAME_MAX_LENGTH
    assert identity.job_name.endswith(f"-{suffix}")
    assert all(character.isascii() for character in identity.job_name)


def test_render_slurm_script_renders_identity_directives(tmp_path: Path) -> None:
    profile = ExecutionProfile(command_key="hello", num_nodes=1, launcher="single")
    identity = build_slurm_job_identity(
        job_key="campaign-42/target-001",
        spec_hash="qcsc-prefect-bulk-spec-v1:sha256:abc123",
    )
    req = SlurmJobRequest(
        partition="compute",
        executable="hello",
        job_name=identity.job_name,
        comment=identity.comment,
    )

    text = render_script(work_dir=tmp_path, exec_profile=profile, req=req)

    assert f"#SBATCH --job-name={identity.job_name}" in text
    assert f"#SBATCH --comment={identity.comment}" in text


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("job_name", "valid\n#SBATCH --output=unsafe"),
        ("comment", "valid comment with spaces"),
    ],
)
def test_render_slurm_script_rejects_unsafe_identity_directives(
    tmp_path: Path,
    field_name: str,
    value: str,
) -> None:
    profile = ExecutionProfile(command_key="hello", num_nodes=1, launcher="single")
    req = SlurmJobRequest(
        partition="compute",
        executable="hello",
        **{field_name: value},
    )

    with pytest.raises(ValueError, match="safe scheduler directive characters"):
        render_script(work_dir=tmp_path, exec_profile=profile, req=req)
