"""The ROQUO multi-GPU example config: its blocks, and the job script of one solve.

As in ``test_uhf_sqd_roquo_gpu_blocks.py``: the blocks come from
``uhf_sqd_blocks.roquo-multigpu.example.toml`` through ``create_blocks`` and the real
``create_sbd_blocks``, with saving replaced, and the script is built along the path of a real
solve up to the Slurm template.
"""

from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

import pytest
from prefect.blocks.core import Block
from qcsc_prefect_adapters.slurm import builder as slurm_builder
from qcsc_prefect_executor import from_blocks
from qcsc_prefect_sbd import block_utils, solver_job
from uhf_sqd import create_blocks

EXAMPLE = Path(__file__).resolve().parents[1] / "uhf_sqd_blocks.roquo-multigpu.example.toml"


@pytest.fixture
def blocks(monkeypatch):
    saved = {}
    monkeypatch.setattr(block_utils, "register_sbd_block_types", lambda: None)
    monkeypatch.setattr(
        Block, "save", lambda self, name, overwrite=False: saved.update({name: self})
    )
    config = tomllib.loads(EXAMPLE.read_text())
    config["solver"].update(
        project="rq-test", root_dir="/shared/uhf", sbd_executable="/sw/diag-gpu_uhf-mpi"
    )
    names = create_blocks.create_blocks(config)
    return names, saved


def test_the_example_gives_multi_gpu_blocks_of_their_own(blocks):
    names, saved = blocks
    assert names == {
        "command_block_name": "cmd-sbd-uhf-solver-roquo-multigpu",
        "execution_profile_block_name": "exec-sbd-uhf-roquo-multigpu",
        "hpc_profile_block_name": "hpc-roquo-multigpu-sbd-uhf",
        "solver_block_name": "sbd-uhf-solver-roquo-multigpu",
        "quantum_sources": {},
    }
    solver = saved["sbd-uhf-solver-roquo-multigpu"]
    assert (solver.solver_mode, solver.integrals) == ("gpu", "unrestricted")
    assert (solver.task_comm_size, solver.adet_comm_size, solver.bdet_comm_size) == (1, 1, 2)
    for key, value in create_blocks.SOLVER_DEFAULTS.items():
        assert getattr(solver, key) == value
    execution = saved["exec-sbd-uhf-roquo-multigpu"]
    assert (execution.num_nodes, execution.mpiprocs, execution.gpus_per_node) == (1, 2, 2)
    hpc = saved["hpc-roquo-multigpu-sbd-uhf"]
    assert hpc.executable_map == {"sbd_uhf_solver": "/sw/diag-gpu_uhf-mpi"}


def test_the_example_names_none_of_the_one_gpu_blocks(blocks):
    _, saved = blocks
    one_gpu = tomllib.loads(EXAMPLE.with_name("uhf_sqd_blocks.roquo-gpu.example.toml").read_text())
    names = {value for key, value in one_gpu["solver"].items() if key.endswith("_block_name")}
    assert not names & set(saved)


def test_one_solve_is_one_job_of_two_ranks_on_two_gpus(blocks, monkeypatch, tmp_path):
    _, saved = blocks
    solver = saved["sbd-uhf-solver-roquo-multigpu"]

    def api(kind):
        async def load(name):
            assert name in saved, name
            return saved[name]

        return type(kind, (), {"load": staticmethod(load)})

    for kind in ("CommandBlock", "ExecutionProfileBlock", "HPCProfileBlock"):
        monkeypatch.setattr(from_blocks, kind, api(kind))
    captured = {}

    async def fake_run_slurm_job(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(from_blocks, "run_slurm_job", fake_run_slurm_job)
    asyncio.run(
        from_blocks.run_job_from_blocks(
            command_block_name=solver.command_block_name,
            execution_profile_block_name=solver.execution_profile_block_name,
            hpc_profile_block_name=solver.hpc_profile_block_name,
            work_dir=tmp_path,
            script_filename=solver.script_filename,
            user_args=solver_job._build_solver_args(solver, separate_beta=True),
        )
    )
    script = slurm_builder.render_script(
        work_dir=captured["work_dir"], exec_profile=captured["exec_profile"], req=captured["req"]
    )
    lines = script.splitlines()
    work_dir = tmp_path.resolve()
    assert [line for line in lines if line.startswith("#SBATCH")] == [
        "#SBATCH --partition=roquo",
        "#SBATCH --account=rq-test",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=2",
        "#SBATCH --cpus-per-task=36",
        "#SBATCH --gpus-per-node=2",
        "#SBATCH --time=00:20:00",
        f"#SBATCH --output={work_dir}/output.out",
        f"#SBATCH --error={work_dir}/output.err",
    ]
    assert 'export OMP_NUM_THREADS="36"' in lines
    launch = lines[-1]
    # Every rank sees both GPUs (no --gpu-bind); the driver picks local rank % GPUs.
    assert launch.startswith('srun --mpi=pmix --cpus-per-task=36 "${QCSC_PREFECT_EXECUTABLE}" ')
    assert "--gpu-bind" not in script
    grid = "--task_comm_size 1 --adet_comm_size 1 --bdet_comm_size 2"
    assert grid in launch
    for arg in ("--adetfile AlphaDets.bin", "--bdetfile BetaDets.bin", "--carryoverfile"):
        assert arg in launch
