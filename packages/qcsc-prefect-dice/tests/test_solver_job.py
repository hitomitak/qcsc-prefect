from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qiskit_addon_sqd.fermion import SCIResult

from qcsc_prefect_dice import solver_job as mod
from qcsc_prefect_dice.solver_job import DiceSHCISolverJob


def test_run_dice_inner_runs_through_block_executor(monkeypatch, tmp_path: Path):
    work_dir = tmp_path / "dice_jobs"
    prepared: dict[str, object] = {}
    executed: dict[str, object] = {}

    def fake_prep_dice_input_files(**kwargs):
        prepared.update(kwargs)

    def fake_read_dice_output_files(**kwargs):
        return SCIResult(
            energy=-1.0,
            sci_state=None,
            orbital_occupancies=(np.array([0.4]), np.array([0.6])),
        )

    async def fake_run_job_from_blocks(**kwargs):
        executed.update(kwargs)
        return SimpleNamespace(exit_status=0)

    monkeypatch.setattr(mod, "prep_dice_input_files", fake_prep_dice_input_files)
    monkeypatch.setattr(mod, "read_dice_output_files", fake_read_dice_output_files)
    monkeypatch.setattr(mod, "run_job_from_blocks", fake_run_job_from_blocks)

    solver = DiceSHCISolverJob(root_dir=str(work_dir))

    result = asyncio.run(
        mod._run_dice_inner.fn(
            solver=solver,
            ci_strings=(np.array([1]), np.array([1])),
            one_body_tensor=np.eye(1),
            two_body_tensor=np.zeros((1, 1, 1, 1)),
            norb=1,
            nelec=(1, 1),
            spin_sq=0.0,
        )
    )

    assert isinstance(result, SCIResult)
    assert prepared["norb"] == 1
    assert prepared["nelec"] == (1, 1)
    assert executed["command_block_name"] == "cmd-dice-solver"
    assert executed["execution_profile_block_name"] == "exec-dice-mpi"
    assert executed["hpc_profile_block_name"] == "hpc-miyabi-dice"
    assert Path(executed["work_dir"]).parent == work_dir.resolve()
    assert executed["user_args"] == []


def test_run_dice_inner_allows_nonzero_exit_when_outputs_exist(monkeypatch, tmp_path: Path):
    def fake_prep_dice_input_files(**kwargs):
        return None

    def fake_read_dice_output_files(**kwargs):
        return SCIResult(
            energy=-0.5,
            sci_state=None,
            orbital_occupancies=(np.array([0.1]), np.array([0.9])),
        )

    async def fake_run_job_from_blocks(**kwargs):
        return SimpleNamespace(exit_status=17)

    monkeypatch.setattr(mod, "prep_dice_input_files", fake_prep_dice_input_files)
    monkeypatch.setattr(mod, "read_dice_output_files", fake_read_dice_output_files)
    monkeypatch.setattr(mod, "run_job_from_blocks", fake_run_job_from_blocks)

    solver = DiceSHCISolverJob(root_dir=str(tmp_path / "dice_jobs"))

    result = asyncio.run(
        mod._run_dice_inner.fn(
            solver=solver,
            ci_strings=(np.array([1]), np.array([1])),
            one_body_tensor=np.eye(1),
            two_body_tensor=np.zeros((1, 1, 1, 1)),
            norb=1,
            nelec=(1, 1),
        )
    )

    assert result.energy == -0.5
