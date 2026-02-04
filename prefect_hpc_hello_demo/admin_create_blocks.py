"""
Admin helper to create demo Blocks in Prefect (MPI hello).

This demo assumes you compile ./hello_mpi in the repo root and submit from there.
We set executable_map to "./hello_mpi" and the PBS script runs under PBS_O_WORKDIR.
"""
from prefect_hpc_demo.blocks import CommandBlock, ExecutionProfileBlock, MiyabiHPCProfileBlock

CommandBlock(
    command_name="mpi-hello",
    executable_key="hello_mpi",
    description="MPI Hello World demo (prints rank/size/hostname)",
).save("cmd-mpi-hello", overwrite=True)

ExecutionProfileBlock(
    profile_name="hello-n2",
    command_name="mpi-hello",
    resource_class="cpu",
    nodes=2,
    walltime="00:05:00",
    ranks_per_node=4,
    threads_per_rank=1,
    launcher="mpirun",
    modules=None,  # e.g., ["openmpi/4.1.5"]
).save("exec-hello-n2", overwrite=True)

ExecutionProfileBlock(
    profile_name="hello-n8",
    command_name="mpi-hello",
    resource_class="cpu",
    nodes=8,
    walltime="00:05:00",
    ranks_per_node=4,
    threads_per_rank=1,
    launcher="mpirun",
).save("exec-hello-n8", overwrite=True)

MiyabiHPCProfileBlock(
    queue_cpu="cpu",
    queue_gpu="gpu",
    project_cpu=None,      # set if your site requires it
    module_init=None,      # e.g., ["module purge"]
    spack_setup=None,
    executable_map={
        "hello_mpi": "./hello_mpi",
    },
).save("hpc-miyabi", overwrite=True)

print("Blocks created: cmd-mpi-hello, exec-hello-n2, exec-hello-n8, hpc-miyabi")
