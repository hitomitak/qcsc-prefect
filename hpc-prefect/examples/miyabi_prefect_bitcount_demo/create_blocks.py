from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any


def _import_wrapper_block_class():
    try:
        from examples.miyabi_prefect_bitcount_demo.wrapper_block import BitCounterWrapperBlock

        return BitCounterWrapperBlock
    except ModuleNotFoundError:
        # Supports direct script execution:
        # python examples/miyabi_prefect_bitcount_demo/create_blocks.py ...
        from wrapper_block import BitCounterWrapperBlock

        return BitCounterWrapperBlock


def _env_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return int(raw)


def _split_csv(name: str) -> list[str] | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]

def _resolve_example_path() -> Path:
    return Path(__file__).resolve().parent


def _register_block_types(bit_counter_cls) -> None:
    from hpc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock

    block_types = [
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
        bit_counter_cls,
    ]
    for block_cls in block_types:
        register = getattr(block_cls, "register_type_and_schema", None)
        if callable(register):
            register()


def _set_variable(variable_name: str, shots: int) -> None:
    value = json.dumps({"params": {"shots": shots}})
    subprocess.run(
        ["prefect", "variable", "set", variable_name, value, "--overwrite"],
        check=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Prefect blocks for Miyabi BitCount tutorial.")
    parser.add_argument("--config", type=Path, help="Path to TOML/JSON config file.")
    parser.add_argument("--project")
    parser.add_argument("--queue")
    parser.add_argument("--work-dir")
    parser.add_argument("--launcher")
    parser.add_argument("--walltime")
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--mpiprocs", type=int)
    parser.add_argument("--ompthreads", type=int)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--modules", nargs="+")
    parser.add_argument("--mpi-options", nargs="*")
    parser.add_argument("--wrapper-executable")
    parser.add_argument("--optimized-executable")
    parser.add_argument("--wrapper-block-name")
    parser.add_argument("--command-block-name")
    parser.add_argument("--execution-profile-block-name")
    parser.add_argument("--hpc-profile-block-name")
    parser.add_argument("--options-variable-name")
    return parser.parse_args()


def _normalize_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"Expected list[str] or CSV string, got: {type(value)}")


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


def _env_values() -> dict[str, Any]:
    return {
        "project": os.getenv("MIYABI_PBS_PROJECT", "").strip() or None,
        "queue": os.getenv("MIYABI_PBS_QUEUE", "").strip() or None,
        "work_dir": os.getenv("MIYABI_BITCOUNT_WORK_DIR", "").strip() or None,
        "launcher": os.getenv("MIYABI_LAUNCHER", "").strip() or None,
        "walltime": os.getenv("MIYABI_WALLTIME", "").strip() or None,
        "num_nodes": _env_int("MIYABI_NUM_NODES"),
        "mpiprocs": _env_int("MIYABI_MPIPROCS"),
        "ompthreads": _env_int("MIYABI_OMPTHREADS"),
        "shots": _env_int("BITCOUNT_SHOTS"),
        "modules": _split_csv("MIYABI_MODULES"),
        "mpi_options": _split_csv("MIYABI_MPI_OPTIONS"),
        "wrapper_executable": os.getenv("BITCOUNT_WRAPPER_EXECUTABLE", "").strip() or None,
        "optimized_executable": os.getenv("BITCOUNT_OPT_EXECUTABLE", "").strip() or None,
        "wrapper_block_name": os.getenv("BITCOUNT_WRAPPER_BLOCK_NAME", "").strip() or None,
        "command_block_name": os.getenv("BITCOUNT_CMD_BLOCK_NAME", "").strip() or None,
        "execution_profile_block_name": os.getenv("BITCOUNT_EXEC_PROFILE_BLOCK_NAME", "").strip() or None,
        "hpc_profile_block_name": os.getenv("BITCOUNT_HPC_PROFILE_BLOCK_NAME", "").strip() or None,
        "options_variable_name": os.getenv("BITCOUNT_OPTIONS_VARIABLE", "").strip() or None,
    }


def main() -> None:
    args = _parse_args()
    from hpc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
    bit_counter_cls = _import_wrapper_block_class()

    config = _load_config_file(args.config)
    env = _env_values()

    example_dir = _resolve_example_path()
    default_wrapper_exec = str((example_dir / "bin/get_counts_json").resolve())
    default_optimized_exec = str((example_dir / "bin/get_counts_hist").resolve())

    project = _pick_value(args.project, config.get("project"), env.get("project"))
    if not project:
        raise RuntimeError("Set 'project' in --config, --project, or MIYABI_PBS_PROJECT.")

    queue = _pick_value(args.queue, config.get("queue"), env.get("queue"))
    if not queue:
        raise RuntimeError("Set 'queue' in --config, --queue, or MIYABI_PBS_QUEUE.")

    launcher = _pick_value(args.launcher, config.get("launcher"), env.get("launcher"), "mpiexec.hydra")
    walltime = _pick_value(args.walltime, config.get("walltime"), env.get("walltime"), "00:10:00")

    num_nodes = int(_pick_value(args.num_nodes, config.get("num_nodes"), env.get("num_nodes"), 2))
    mpiprocs = int(_pick_value(args.mpiprocs, config.get("mpiprocs"), env.get("mpiprocs"), 5))
    ompthreads = int(_pick_value(args.ompthreads, config.get("ompthreads"), env.get("ompthreads"), 1))
    shots = int(_pick_value(args.shots, config.get("shots"), env.get("shots"), 100000))

    modules = _normalize_str_list(
        _pick_value(args.modules, config.get("modules"), env.get("modules"), ["intel/2023.2.0", "impi/2021.10.0"])
    )
    mpi_options = _normalize_str_list(
        _pick_value(args.mpi_options, config.get("mpi_options"), env.get("mpi_options"), [])
    )

    wrapper_exec = str(
        _pick_value(args.wrapper_executable, config.get("wrapper_executable"), env.get("wrapper_executable"), default_wrapper_exec)
    ).strip()
    optimized_exec = str(
        _pick_value(args.optimized_executable, config.get("optimized_executable"), env.get("optimized_executable"), default_optimized_exec)
    ).strip()

    work_dir_raw = _pick_value(args.work_dir, config.get("work_dir"), env.get("work_dir"))
    if not work_dir_raw:
        raise RuntimeError("Set 'work_dir' in --config, --work-dir, or MIYABI_BITCOUNT_WORK_DIR.")
    work_dir_raw = str(work_dir_raw).strip()
    work_dir = os.path.expandvars(work_dir_raw)

    wrapper_block_name = str(
        _pick_value(args.wrapper_block_name, config.get("wrapper_block_name"), env.get("wrapper_block_name"), "bit-counter-wrapper-demo")
    ).strip()
    cmd_block_name = str(
        _pick_value(args.command_block_name, config.get("command_block_name"), env.get("command_block_name"), "cmd-bitcount-hist")
    ).strip()
    exec_block_name = str(
        _pick_value(
            args.execution_profile_block_name,
            config.get("execution_profile_block_name"),
            env.get("execution_profile_block_name"),
            "exec-bitcount-mpi",
        )
    ).strip()
    hpc_block_name = str(
        _pick_value(args.hpc_profile_block_name, config.get("hpc_profile_block_name"), env.get("hpc_profile_block_name"), "hpc-miyabi-bitcount")
    ).strip()
    options_variable_name = str(
        _pick_value(args.options_variable_name, config.get("options_variable_name"), env.get("options_variable_name"), "miyabi-bitcount-options")
    ).strip()

    _register_block_types(bit_counter_cls)

    bit_counter_cls(
        root_dir=work_dir,
        executable=wrapper_exec,
        queue_name=queue,
        project=project,
        num_nodes=num_nodes,
        num_mpi_processes=mpiprocs,
        ompthreads=ompthreads,
        walltime=walltime,
        launcher=launcher,
        load_modules=modules,
        mpi_options=mpi_options,
        environments={},
        script_filename="bitcount_wrapper.pbs",
    ).save(wrapper_block_name, overwrite=True)

    CommandBlock(
        command_name="bitcount-hist",
        executable_key="bitcount_hist",
        description="Optimized bitcount executable (binary histogram output)",
        default_args=[],
    ).save(cmd_block_name, overwrite=True)

    ExecutionProfileBlock(
        profile_name="bitcount-mpi",
        command_name="bitcount-hist",
        resource_class="cpu",
        num_nodes=num_nodes,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        walltime=walltime,
        launcher=launcher,
        mpi_options=mpi_options,
        modules=modules,
        environments={},
    ).save(exec_block_name, overwrite=True)

    HPCProfileBlock(
        hpc_target="miyabi",
        queue_cpu=queue,
        queue_gpu="regular-g",
        project_cpu=project,
        project_gpu=project,
        executable_map={"bitcount_hist": optimized_exec},
    ).save(hpc_block_name, overwrite=True)

    _set_variable(options_variable_name, shots)

    print("Saved blocks and variable for Miyabi BitCount demo")
    print(f"  Wrapper block: {wrapper_block_name}")
    print(f"  Command block: {cmd_block_name}")
    print(f"  Execution profile block: {exec_block_name}")
    print(f"  HPC profile block: {hpc_block_name}")
    print(f"  Options variable: {options_variable_name}")
    print(f"  Work directory: {work_dir}")
    print(f"  Wrapper executable: {wrapper_exec}")
    print(f"  Optimized executable: {optimized_exec}")


if __name__ == "__main__":
    main()
