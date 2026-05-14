from __future__ import annotations

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
        executable="/path/to/hello",
    )

    text = render_script(work_dir=tmp_path, exec_profile=profile, req=req)

    assert "#SBATCH --partition=compute" in text
    assert "#SBATCH --account=proj01" in text
    assert "#SBATCH --nodes=2" in text
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
