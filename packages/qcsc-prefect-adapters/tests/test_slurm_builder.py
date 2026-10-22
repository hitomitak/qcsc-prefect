from __future__ import annotations

import os
import subprocess
from pathlib import Path

from qcsc_prefect_adapters.slurm.builder import SlurmJobRequest, render_script
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


def test_mpirun_without_application_arguments_executes_the_binary(tmp_path: Path):
    # The BitCount PoC has MPI flags but no application arguments. Exercise
    # the generated shell command so a missing executable cannot pass silently.
    launcher = tmp_path / "mpirun"
    launcher.write_text('#!/bin/bash\nset -eu\ntest "$1" = --display-map\nshift\nexec "$@"\n')
    launcher.chmod(0o700)
    executable = tmp_path / "bitcount executable"
    executable.write_text('#!/bin/bash\nprintf executed > "application-ran"\n')
    executable.chmod(0o700)
    profile = ExecutionProfile(
        command_key="bitcount",
        num_nodes=2,
        mpiprocs=1,
        launcher="mpirun",
        mpi_options=["--display-map"],
    )
    text = render_script(
        work_dir=tmp_path,
        exec_profile=profile,
        req=SlurmJobRequest(partition="debug", executable=str(executable)),
    )
    subprocess.run(
        ["bash", "-c", text],
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        check=True,
        capture_output=True,
    )
    assert (tmp_path / "application-ran").read_text() == "executed"
