from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


def _import_sbd_solver_block():
    # Supports direct execution:
    # python algorithms/sbd/create_blocks.py ...
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from sbd.solver_job import SBDSolverJob

    return SBDSolverJob


def _register_block_types(*custom_block_classes) -> None:
    from hpc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock

    block_types = [
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
        *custom_block_classes,
    ]
    for block_cls in block_types:
        register = getattr(block_cls, "register_type_and_schema", None)
        if callable(register):
            register()


def _set_variable(variable_name: str, value: Any) -> None:
    subprocess.run(
        ["prefect", "variable", "set", variable_name, json.dumps(value), "--overwrite"],
        check=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Prefect blocks for SBD workflow on Miyabi.")
    parser.add_argument("--config", type=Path, help="Path to TOML/JSON config file.")

    parser.add_argument("--project")
    parser.add_argument("--queue")
    parser.add_argument("--work-dir")
    parser.add_argument("--sbd-executable")

    parser.add_argument("--launcher")
    parser.add_argument("--walltime")
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--mpiprocs", type=int)
    parser.add_argument("--ompthreads", type=int)
    parser.add_argument("--modules", nargs="+")
    parser.add_argument("--mpi-options", nargs="*")

    parser.add_argument("--command-block-name")
    parser.add_argument("--execution-profile-block-name")
    parser.add_argument("--hpc-profile-block-name")
    parser.add_argument("--solver-block-name")
    parser.add_argument("--options-variable-name")

    parser.add_argument("--task-comm-size", type=int)
    parser.add_argument("--adet-comm-size", type=int)
    parser.add_argument("--bdet-comm-size", type=int)
    parser.add_argument("--block", type=int)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--carryover-ratio", type=float)
    parser.add_argument("--solver-mode", choices=["cpu", "gpu"])
    parser.add_argument("--shots", type=int)
    parser.add_argument(
        "--sqd-options-json",
        help="Raw JSON for Prefect variable value (e.g. '{\"params\": {\"shots\": 500000}}').",
    )
    return parser.parse_args()


def _load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file was not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as f:
        return tomllib.load(f)


def _pick_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"Expected list[str] or CSV string, got: {type(value)}")


def _env_values() -> dict[str, Any]:
    def env_int(name: str) -> int | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return int(raw)

    def env_float(name: str) -> float | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return float(raw)

    def env_csv(name: str) -> list[str] | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return [item.strip() for item in raw.split(",") if item.strip()]

    return {
        "project": os.getenv("MIYABI_PBS_PROJECT", "").strip() or None,
        "queue": os.getenv("MIYABI_PBS_QUEUE", "").strip() or None,
        "work_dir": os.getenv("SBD_WORK_DIR", "").strip() or None,
        "sbd_executable": os.getenv("SBD_EXECUTABLE", "").strip() or None,
        "launcher": os.getenv("MIYABI_LAUNCHER", "").strip() or None,
        "walltime": os.getenv("MIYABI_WALLTIME", "").strip() or None,
        "num_nodes": env_int("MIYABI_NUM_NODES"),
        "mpiprocs": env_int("MIYABI_MPIPROCS"),
        "ompthreads": env_int("MIYABI_OMPTHREADS"),
        "modules": env_csv("MIYABI_MODULES"),
        "mpi_options": env_csv("MIYABI_MPI_OPTIONS"),
        "command_block_name": os.getenv("SBD_CMD_BLOCK_NAME", "").strip() or None,
        "execution_profile_block_name": os.getenv("SBD_EXEC_PROFILE_BLOCK_NAME", "").strip() or None,
        "hpc_profile_block_name": os.getenv("SBD_HPC_PROFILE_BLOCK_NAME", "").strip() or None,
        "solver_block_name": os.getenv("SBD_SOLVER_BLOCK_NAME", "").strip() or None,
        "options_variable_name": os.getenv("SBD_OPTIONS_VARIABLE", "").strip() or None,
        "task_comm_size": env_int("SBD_TASK_COMM_SIZE"),
        "adet_comm_size": env_int("SBD_ADET_COMM_SIZE"),
        "bdet_comm_size": env_int("SBD_BDET_COMM_SIZE"),
        "block": env_int("SBD_BLOCK"),
        "iteration": env_int("SBD_ITERATION"),
        "tolerance": env_float("SBD_TOLERANCE"),
        "carryover_ratio": env_float("SBD_CARRYOVER_RATIO"),
        "solver_mode": os.getenv("SBD_SOLVER_MODE", "").strip() or None,
        "shots": env_int("SBD_SHOTS"),
        "sqd_options_json": os.getenv("SBD_OPTIONS_JSON", "").strip() or None,
    }


def main() -> None:
    args = _parse_args()
    config = _load_config_file(args.config)
    env = _env_values()

    from hpc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock

    sbd_solver_cls = _import_sbd_solver_block()
    _register_block_types(sbd_solver_cls)

    project = _pick_value(args.project, config.get("project"), env.get("project"))
    if not project:
        raise RuntimeError("Set 'project' in --config, --project, or MIYABI_PBS_PROJECT.")
    queue = _pick_value(args.queue, config.get("queue"), env.get("queue"))
    if not queue:
        raise RuntimeError("Set 'queue' in --config, --queue, or MIYABI_PBS_QUEUE.")
    work_dir = _pick_value(args.work_dir, config.get("work_dir"), env.get("work_dir"))
    if not work_dir:
        raise RuntimeError("Set 'work_dir' in --config, --work-dir, or SBD_WORK_DIR.")
    sbd_executable = _pick_value(args.sbd_executable, config.get("sbd_executable"), env.get("sbd_executable"))
    if not sbd_executable:
        raise RuntimeError("Set 'sbd_executable' in --config, --sbd-executable, or SBD_EXECUTABLE.")

    launcher = str(_pick_value(args.launcher, config.get("launcher"), env.get("launcher"), "mpiexec.hydra")).strip()
    walltime = str(_pick_value(args.walltime, config.get("walltime"), env.get("walltime"), "02:00:00")).strip()

    num_nodes = int(_pick_value(args.num_nodes, config.get("num_nodes"), env.get("num_nodes"), 1))
    mpiprocs = int(_pick_value(args.mpiprocs, config.get("mpiprocs"), env.get("mpiprocs"), 4))
    ompthreads = int(_pick_value(args.ompthreads, config.get("ompthreads"), env.get("ompthreads"), 1))

    modules = _normalize_str_list(
        _pick_value(args.modules, config.get("modules"), env.get("modules"), ["intel/2023.2.0", "impi/2021.10.0"])
    )
    mpi_options = _normalize_str_list(
        _pick_value(args.mpi_options, config.get("mpi_options"), env.get("mpi_options"), [])
    )

    command_block_name = str(
        _pick_value(args.command_block_name, config.get("command_block_name"), env.get("command_block_name"), "cmd-sbd-diag")
    ).strip()
    execution_profile_block_name = str(
        _pick_value(
            args.execution_profile_block_name,
            config.get("execution_profile_block_name"),
            env.get("execution_profile_block_name"),
            "exec-sbd-mpi",
        )
    ).strip()
    hpc_profile_block_name = str(
        _pick_value(args.hpc_profile_block_name, config.get("hpc_profile_block_name"), env.get("hpc_profile_block_name"), "hpc-miyabi-sbd")
    ).strip()
    solver_block_name = str(
        _pick_value(args.solver_block_name, config.get("solver_block_name"), env.get("solver_block_name"), "davidson-solver")
    ).strip()
    options_variable_name = str(
        _pick_value(args.options_variable_name, config.get("options_variable_name"), env.get("options_variable_name"), "sqd_options")
    ).strip()

    task_comm_size = int(_pick_value(args.task_comm_size, config.get("task_comm_size"), env.get("task_comm_size"), 1))
    adet_comm_size = int(_pick_value(args.adet_comm_size, config.get("adet_comm_size"), env.get("adet_comm_size"), 1))
    bdet_comm_size = int(_pick_value(args.bdet_comm_size, config.get("bdet_comm_size"), env.get("bdet_comm_size"), 1))
    block = int(_pick_value(args.block, config.get("block"), env.get("block"), 4))
    iteration = int(_pick_value(args.iteration, config.get("iteration"), env.get("iteration"), 1))
    tolerance = float(_pick_value(args.tolerance, config.get("tolerance"), env.get("tolerance"), 1e-2))
    carryover_ratio = float(_pick_value(args.carryover_ratio, config.get("carryover_ratio"), env.get("carryover_ratio"), 0.1))
    solver_mode = str(_pick_value(args.solver_mode, config.get("solver_mode"), env.get("solver_mode"), "cpu")).strip()

    shots = int(_pick_value(args.shots, config.get("shots"), env.get("shots"), 500000))
    sqd_options_json = _pick_value(args.sqd_options_json, config.get("sqd_options_json"), env.get("sqd_options_json"))

    CommandBlock(
        command_name="sbd-diag",
        executable_key="sbd_diag",
        description="SBD diagonalization executable",
        default_args=[],
    ).save(command_block_name, overwrite=True)

    ExecutionProfileBlock(
        profile_name="sbd-mpi",
        command_name="sbd-diag",
        resource_class="cpu",
        num_nodes=num_nodes,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        walltime=walltime,
        launcher=launcher,
        mpi_options=mpi_options or [],
        modules=modules or [],
        environments={},
    ).save(execution_profile_block_name, overwrite=True)

    HPCProfileBlock(
        hpc_target="miyabi",
        queue_cpu=str(queue),
        queue_gpu="regular-g",
        project_cpu=str(project),
        project_gpu=str(project),
        executable_map={"sbd_diag": str(Path(str(sbd_executable)).expanduser().resolve())},
    ).save(hpc_profile_block_name, overwrite=True)

    sbd_solver_cls(
        root_dir=str(Path(str(work_dir)).expanduser().resolve()),
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        script_filename="sbd_solver.pbs",
        metrics_artifact_key="miyabi-sbd-metrics",
        task_comm_size=task_comm_size,
        adet_comm_size=adet_comm_size,
        bdet_comm_size=bdet_comm_size,
        block=block,
        iteration=iteration,
        tolerance=tolerance,
        carryover_ratio=carryover_ratio,
        solver_mode=solver_mode,
        user_args=[],
    ).save(solver_block_name, overwrite=True)

    if sqd_options_json:
        options_value = json.loads(str(sqd_options_json))
    else:
        options_value = {"params": {"shots": shots}}
    _set_variable(options_variable_name, options_value)

    print("Saved blocks and variable for SBD workflow (Miyabi)")
    print(f"  SBD solver block: {solver_block_name}")
    print(f"  Command block: {command_block_name}")
    print(f"  Execution profile block: {execution_profile_block_name}")
    print(f"  HPC profile block: {hpc_profile_block_name}")
    print(f"  Options variable: {options_variable_name}")
    print(f"  Work directory: {Path(str(work_dir)).expanduser().resolve()}")
    print(f"  SBD executable: {Path(str(sbd_executable)).expanduser().resolve()}")


if __name__ == "__main__":
    main()
