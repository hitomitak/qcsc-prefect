from __future__ import annotations

import os
from pathlib import Path

from hpc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock


def _resolve_demo_executable() -> str:
    env_path = os.getenv("FUGAKU_DEMO_EXECUTABLE", "").strip()
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    return str((Path(__file__).resolve().parent / "hello_demo.sh").resolve())


def main() -> None:
    project = os.getenv("FUGAKU_PROJECT", "").strip()
    if not project:
        raise RuntimeError("Set FUGAKU_PROJECT before running create_blocks.py.")

    rscgrp = os.getenv("FUGAKU_RSCGRP", "small").strip()
    gfscache = os.getenv("FUGAKU_GFSCACHE", "/vol0002").strip()
    executable = _resolve_demo_executable()

    CommandBlock(
        command_name="hello-demo",
        executable_key="hello_demo",
        description="Simple hello script for Fugaku + Prefect block demo",
        default_args=[],
    ).save("cmd-fugaku-hello-demo", overwrite=True)

    ExecutionProfileBlock(
        profile_name="hello-single-node",
        command_name="hello-demo",
        resource_class="cpu",
        num_nodes=1,
        mpiprocs=1,
        ompthreads=1,
        walltime="00:05:00",
        launcher="single",
        modules=[],
        environments={},
    ).save("exec-fugaku-hello-single", overwrite=True)

    HPCProfileBlock(
        hpc_target="fugaku",
        queue_cpu=rscgrp,
        queue_gpu=rscgrp,
        project_cpu=project,
        project_gpu=project,
        executable_map={"hello_demo": executable},
        gfscache=gfscache,
    ).save("hpc-fugaku", overwrite=True)

    print(f"Saved blocks: cmd-fugaku-hello-demo, exec-fugaku-hello-single, hpc-fugaku")
    print(f"  FUGAKU_PROJECT={project}")
    print(f"  FUGAKU_RSCGRP={rscgrp}")
    print(f"  FUGAKU_GFSCACHE={gfscache}")
    print(f"  executable={executable}")


if __name__ == "__main__":
    main()
