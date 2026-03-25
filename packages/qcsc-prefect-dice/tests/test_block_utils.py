from __future__ import annotations

from qcsc_prefect_dice import block_utils as mod


def test_create_dice_blocks_miyabi(monkeypatch, tmp_path):
    saved: list[tuple[str, str, bool, object]] = []

    def fake_register():
        return None

    def fake_command_save(self, name, overwrite=False):
        saved.append(("command", name, overwrite, self))
        return self

    def fake_exec_save(self, name, overwrite=False):
        saved.append(("execution", name, overwrite, self))
        return self

    def fake_hpc_save(self, name, overwrite=False):
        saved.append(("hpc", name, overwrite, self))
        return self

    def fake_solver_save(self, name, overwrite=False):
        saved.append(("solver", name, overwrite, self))
        return self

    monkeypatch.setattr(mod, "register_dice_block_types", fake_register)
    monkeypatch.setattr(mod.CommandBlock, "save", fake_command_save, raising=False)
    monkeypatch.setattr(mod.ExecutionProfileBlock, "save", fake_exec_save, raising=False)
    monkeypatch.setattr(mod.HPCProfileBlock, "save", fake_hpc_save, raising=False)
    monkeypatch.setattr(mod.DiceSHCISolverJob, "save", fake_solver_save, raising=False)

    names = mod.create_dice_blocks(
        hpc_target="miyabi",
        project="gz00",
        queue="regular-c",
        root_dir=str(tmp_path / "jobs"),
        dice_executable=str(tmp_path / "bin" / "Dice"),
    )

    assert names == {
        "command_block_name": "cmd-dice-solver",
        "execution_profile_block_name": "exec-dice-mpi",
        "hpc_profile_block_name": "hpc-miyabi-dice",
        "solver_block_name": "dice-solver",
    }
    kinds = [kind for kind, *_ in saved]
    assert kinds == ["command", "execution", "hpc", "solver"]

    hpc_block = next(model for kind, _, _, model in saved if kind == "hpc")
    assert hpc_block.hpc_target == "miyabi"
    assert hpc_block.queue_cpu == "regular-c"
    assert hpc_block.project_cpu == "gz00"


def test_create_dice_blocks_fugaku(monkeypatch, tmp_path):
    saved: list[tuple[str, str, bool, object]] = []

    def fake_register():
        return None

    def fake_save(kind):
        def _save(self, name, overwrite=False):
            saved.append((kind, name, overwrite, self))
            return self

        return _save

    monkeypatch.setattr(mod, "register_dice_block_types", fake_register)
    monkeypatch.setattr(mod.CommandBlock, "save", fake_save("command"), raising=False)
    monkeypatch.setattr(mod.ExecutionProfileBlock, "save", fake_save("execution"), raising=False)
    monkeypatch.setattr(mod.HPCProfileBlock, "save", fake_save("hpc"), raising=False)
    monkeypatch.setattr(mod.DiceSHCISolverJob, "save", fake_save("solver"), raising=False)

    names = mod.create_dice_blocks(
        hpc_target="fugaku",
        project="ra000000",
        queue="small",
        root_dir=str(tmp_path / "jobs"),
        dice_executable=str(tmp_path / "bin" / "Dice"),
        spack_modules=["fjmpi"],
        mpi_options_for_pjm=["max-proc-per-node=2"],
        pjm_resources=["freq=2200"],
    )

    assert names["execution_profile_block_name"] == "exec-dice-fugaku"
    assert names["hpc_profile_block_name"] == "hpc-fugaku-dice"

    execution_block = next(model for kind, _, _, model in saved if kind == "execution")
    hpc_block = next(model for kind, _, _, model in saved if kind == "hpc")
    solver_block = next(model for kind, _, _, model in saved if kind == "solver")

    assert execution_block.launcher == "mpiexec"
    assert hpc_block.hpc_target == "fugaku"
    assert hpc_block.spack_modules == ["fjmpi"]
    assert hpc_block.mpi_options_for_pjm == ["max-proc-per-node=2"]
    assert hpc_block.pjm_resources == ["freq=2200"]
    assert solver_block.script_filename == "dice_solver.pjm"
    assert solver_block.metrics_artifact_key == "fugaku-dice-metrics"
