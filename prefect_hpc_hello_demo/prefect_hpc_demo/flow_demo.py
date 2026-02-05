"""
Prefect flow demo for Miyabi HPC-agnostic execution (MPI hello).
"""

from __future__ import annotations
from prefect import flow

from .blocks import CommandBlock, ExecutionProfileBlock, MiyabiHPCProfileBlock
from .models import Tuning
from .tasks import generate_script, submit_script


@flow(name="miyabi-hpc-agnostic-mpi-hello")
def miyabi_mpi_hello_flow(
    *,
    command_block_name: str = "cmd-mpi-hello",
    exec_profile_block_name: str = "exec-hello-n2",
    hpc_profile_block_name: str = "hpc-miyabi",
    work_root: str = "./hello_mpi",
    job_name: str = "mpi-hello",
    tuning: Tuning | None = None,
    user_args: list[str] | None = None,
    submit: bool = True,
):
    cmd = CommandBlock.load(command_block_name)
    profile = ExecutionProfileBlock.load(exec_profile_block_name)
    hpc = MiyabiHPCProfileBlock.load(hpc_profile_block_name)

    script_path = generate_script(
        work_root=work_root,
        job_name=job_name,
        cmd=cmd,
        profile=profile,
        hpc=hpc,
        tuning=tuning,
        user_args=user_args,
    )

    if submit:
        job_id = submit_script(script_path)
        return {"script": script_path, "job_id": job_id}

    return {"script": script_path, "job_id": None}
