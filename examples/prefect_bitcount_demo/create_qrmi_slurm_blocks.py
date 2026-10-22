"""Register the six fixed PoC blocks for the test_heron Slurm cluster."""

from __future__ import annotations

import shlex
from pathlib import Path

from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock


def create_blocks() -> None:
    example_dir = Path(__file__).resolve().parent
    binary = example_dir / "bin/get_counts_hist"
    if not binary.is_file():
        raise FileNotFoundError(f"Build the existing MPI BitCount executable first: {binary}")

    # Names are dedicated to this PoC; rerunning updates only these six blocks.
    CommandBlock(
        command_name="poc-qrmi",
        executable_key="python",
        default_args=[shlex.quote(str(example_dir / "qrmi_task_runner.py"))],
    ).save("poc-qrmi-command", overwrite=True)
    ExecutionProfileBlock(
        profile_name="poc-qrmi",
        command_name="poc-qrmi",
        num_nodes=1,
        mpiprocs=1,
        ompthreads=1,
        walltime="00:10:00",
        # Match run_bell.sbatch: run Python directly inside the batch allocation.
        # The node entrypoint then launches the real task_runner executable.
        launcher="single",
    ).save("poc-qrmi-execution", overwrite=True)
    HPCProfileBlock(
        hpc_target="slurm",
        queue_cpu="debug",
        slurm_qpu="test_heron",
        slurm_ntasks=1,
        executable_map={"python": "/shared/venv/qrmi/bin/python"},
    ).save("poc-qrmi-hpc", overwrite=True)

    CommandBlock(command_name="poc-bitcount", executable_key="bitcount").save(
        "poc-bitcount-command", overwrite=True
    )
    ExecutionProfileBlock(
        profile_name="poc-bitcount",
        command_name="poc-bitcount",
        num_nodes=2,
        mpiprocs=1,
        ompthreads=1,
        walltime="00:05:00",
        launcher="mpirun",
        mpi_options=["--display-map"],
        pre_commands=[
            "set -euo pipefail",
            "source /etc/profile.d/modules.sh",
            # Reapply the module environment in the batch shell even when
            # the submitter exported an already-loaded module's state.
            "module unload mpi/openmpi-x86_64",
            "module load mpi/openmpi-x86_64",
            "command -v mpirun",
        ],
        environments={
            "OMPI_MCA_btl_tcp_if_include": "eth0",
            "OMPI_MCA_oob_tcp_if_include": "eth0",
            "OMPI_MCA_pml": "ob1",
            "OMPI_MCA_btl": "tcp,self,vader",
        },
    ).save("poc-bitcount-execution", overwrite=True)
    HPCProfileBlock(
        hpc_target="slurm",
        queue_cpu="debug",
        executable_map={"bitcount": str(binary)},
        # No slurm_qpu: the classical job does not acquire a quantum resource.
    ).save("poc-bitcount-hpc", overwrite=True)
    print("Saved poc-qrmi-* and poc-bitcount-* blocks")


if __name__ == "__main__":
    create_blocks()
