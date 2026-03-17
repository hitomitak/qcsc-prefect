from __future__ import annotations

import os
from pathlib import Path

from qcsc_prefect_blocks.miyabi.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock


def _resolve_demo_executable() -> str:
    env_path = os.getenv("MIYABI_DEMO_EXECUTABLE", "").strip()
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    return str((Path(__file__).resolve().parent / "hello_demo.sh").resolve())


def main() -> None:
    project = os.getenv("MIYABI_PBS_PROJECT", "").strip()
    if not project:
        raise RuntimeError("Set MIYABI_PBS_PROJECT before running create_blocks.py.")

    queue = os.getenv("MIYABI_PBS_QUEUE", "regular-c").strip()
    executable = _resolve_demo_executable()

    CommandBlock(
        command_name="hello-demo",
        executable_key="hello_demo",
        description="Simple hello script for Miyabi + Prefect block demo",
        default_args=[],
    ).save("cmd-hello-demo", overwrite=True)

    ExecutionProfileBlock(
        profile_name="hello-single-node",
        command_name="hello-demo",
        resource_class="cpu",
        num_nodes=1,
        mpiprocs=1,
        ompthreads=1,
        walltime="00:15:00",
        launcher="single",
        modules=[],
        environments={},
    ).save("exec-hello-single", overwrite=True)

    HPCProfileBlock(
        hpc_target="miyabi",
        queue_cpu=queue,
        queue_gpu="regular-g",
        project_cpu=project,
        project_gpu=project,
        executable_map={"hello_demo": executable},
    ).save("hpc-miyabi", overwrite=True)

    print("Saved blocks: cmd-hello-demo, exec-hello-single, hpc-miyabi")


if __name__ == "__main__":
    main()
