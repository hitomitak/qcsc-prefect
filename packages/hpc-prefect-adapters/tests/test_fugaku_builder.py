from __future__ import annotations

from pathlib import Path

from hpc_prefect_adapters.fugaku.builder import FugakuJobRequest, render_script
from hpc_prefect_core.models.execution_profile import ExecutionProfile


def test_render_fugaku_script_with_modules_env_and_extra_resources(tmp_path: Path):
    profile = ExecutionProfile(
        command_key="hello",
        num_nodes=4,
        mpiprocs=1,
        ompthreads=48,
        walltime="00:10:00",
        launcher="mpirun",
        mpi_options=["-n", "4"],
        modules=["LLVM/llvmorg-21.1.0"],
        environments={
            "OMP_NUM_THREADS": "48",
            "UTOFU_SWAP_PROTECT": "1",
            "LD_LIBRARY_PATH": "/lib64:$LD_LIBRARY_PATH",
        },
        arguments=["--foo", "bar"],
    )
    req = FugakuJobRequest(
        queue_name="small",
        project="ra010014",
        executable="/path/to/gb-demo",
        job_name="gbsqd-test",
        gfscache="/vol0004:/vol0002",
        mpi_options_for_pjm=["max-proc-per-node=1"],
        pjm_resources=["freq=2000,eco_state=2"],
    )

    text = render_script(work_dir=tmp_path, exec_profile=profile, req=req)

    assert '#PJM -L "rscgrp=small"' in text
    assert '#PJM -L "node=4"' in text
    assert '#PJM -L "elapse=00:10:00"' in text
    assert '#PJM -L "freq=2000,eco_state=2"' in text
    assert "#PJM -g ra010014" in text
    assert '#PJM --mpi "max-proc-per-node=1"' in text
    assert "#PJM -x PJM_LLIO_GFSCACHE=/vol0004:/vol0002" in text
    assert "module load LLVM/llvmorg-21.1.0" in text
    assert 'export OMP_NUM_THREADS="48"' in text
    assert 'export UTOFU_SWAP_PROTECT="1"' in text
    assert 'export LD_LIBRARY_PATH="/lib64:$LD_LIBRARY_PATH"' in text
    assert "mpirun -n 4 /path/to/gb-demo --foo bar" in text
