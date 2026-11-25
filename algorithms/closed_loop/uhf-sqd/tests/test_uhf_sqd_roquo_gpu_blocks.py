"""The ROQUO GPU example config: its blocks, and the job script of one solve.

The blocks come from ``uhf_sqd_blocks.roquo-gpu.example.toml`` through ``create_blocks`` and the
real ``create_sbd_blocks``, with saving replaced. The script is then built along the path of a
real solve: ``run_job_from_blocks`` with the solver's arguments, up to ``run_slurm_job``, and the
Slurm template.
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

EXAMPLE = Path(__file__).resolve().parents[1] / "uhf_sqd_blocks.roquo-gpu.example.toml"


@pytest.fixture
def blocks(monkeypatch):
    saved = {}
    monkeypatch.setattr(block_utils, "register_sbd_block_types", lambda: None)
    monkeypatch.setattr(
        Block, "save", lambda self, name, overwrite=False: saved.update({name: self})
    )
    config = tomllib.loads(EXAMPLE.read_text())
    config["solver"].update(
        project="rq-test", root_dir="/shared/uhf", sbd_executable="/sw/diag-gpu_uhf"
    )
    names = create_blocks.create_blocks(config)
    return names, saved


def test_the_example_gives_gpu_blocks_of_their_own(blocks):
    names, saved = blocks
    assert names == {
        "command_block_name": "cmd-sbd-uhf-solver-roquo-gpu",
        "execution_profile_block_name": "exec-sbd-uhf-roquo-gpu",
        "hpc_profile_block_name": "hpc-roquo-gpu-sbd-uhf",
        "solver_block_name": "sbd-uhf-solver-roquo-gpu",
        "quantum_sources": {},
    }
    solver = saved["sbd-uhf-solver-roquo-gpu"]
    assert (solver.solver_mode, solver.integrals) == ("gpu", "unrestricted")
    # The Davidson settings are the source's loose defaults, as in the CPU example.
    for key, value in create_blocks.SOLVER_DEFAULTS.items():
        assert getattr(solver, key) == value
    hpc = saved["hpc-roquo-gpu-sbd-uhf"]
    assert (hpc.hpc_target, hpc.queue_gpu, hpc.project_gpu) == ("slurm", "roquo", "rq-test")
    assert hpc.executable_map == {"sbd_uhf_solver": "/sw/diag-gpu_uhf"}


def test_one_solve_is_a_one_gpu_job_with_the_roquo_environment(blocks, monkeypatch, tmp_path):
    _, saved = blocks
    solver = saved["sbd-uhf-solver-roquo-gpu"]

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
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=36",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --time=00:20:00",
        f"#SBATCH --output={work_dir}/output.out",
        f"#SBATCH --error={work_dir}/output.err",
    ]
    assert [line for line in lines if line.startswith("module load")] == [
        "module load cuda/13.2",
        "module load nvhpc/26.5",
        "module load hpcx/2.50",
    ]
    assert 'export OMP_NUM_THREADS="36"' in lines
    launch = lines[-1]
    assert launch.startswith('srun --mpi=pmix "${QCSC_PREFECT_EXECUTABLE}" ')
    # GPU mode: separate alpha and beta files, and the GPU carryover file.
    for arg in ("--adetfile AlphaDets.bin", "--bdetfile BetaDets.bin", "--carryoverfile"):
        assert arg in launch
