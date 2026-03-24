"""Create Prefect blocks for GB SQD workflows."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path


def _import_block_classes():
    """Import block classes from qcsc-prefect packages."""
    # Add qcsc-prefect packages to path if running in development mode
    project_root = Path(__file__).resolve().parents[2]
    if (project_root / "packages").exists():
        sys.path.insert(0, str(project_root / "packages" / "qcsc-prefect-blocks" / "src"))
    
    from qcsc_prefect_blocks.common.blocks import (
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
    )
    
    return CommandBlock, ExecutionProfileBlock, HPCProfileBlock


def _register_block_types(*block_classes):
    """Register block types with Prefect."""
    for block_cls in block_classes:
        register = getattr(block_cls, "register_type_and_schema", None)
        if callable(register):
            register()


def _load_config(config_path: Path) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _default_target_name(*, hpc_target: str, resource_class: str) -> str:
    if resource_class == "gpu":
        return f"{hpc_target}-gpu"
    return hpc_target


def _default_executable_path(
    *,
    script_dir: Path,
    hpc_target: str,
    resource_class: str,
) -> str:
    candidates: list[Path] = []
    source_dir = script_dir / "gb_demo_2026"

    if hpc_target == "miyabi":
        if resource_class == "gpu":
            candidates.extend(
                [
                    source_dir / "build-miyabi-gpu" / "gb-demo",
                    source_dir / "build_gpu" / "gb-demo",
                    source_dir / "build" / "gb-demo",
                ]
            )
        else:
            candidates.extend(
                [
                    source_dir / "build-miyabi-cpu" / "gb-demo",
                    source_dir / "build" / "gb-demo",
                ]
            )
    else:
        candidates.extend(
            [
                source_dir / "build-fugaku" / "gb-demo",
                source_dir / "build" / "gb-demo",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create Prefect blocks for GB SQD workflows (Miyabi or Fugaku)."
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to TOML configuration file (e.g., gb_sqd_blocks.toml)",
    )
    parser.add_argument(
        "--hpc-target",
        choices=["miyabi", "fugaku"],
        help="Target HPC system (overrides config)",
    )
    parser.add_argument(
        "--resource-class",
        choices=["cpu", "gpu"],
        help="Execution resource class (overrides config, default: cpu)",
    )
    parser.add_argument("--project", help="Project/group name (overrides config)")
    parser.add_argument("--queue", help="Queue/resource group name (overrides config)")
    parser.add_argument(
        "--work-dir",
        help="Working directory for job outputs (overrides config)",
    )
    
    # Execution parameters (override config)
    parser.add_argument("--num-nodes", type=int, help="Number of nodes (overrides config)")
    parser.add_argument("--mpiprocs", type=int, help="MPI processes per node (overrides config)")
    parser.add_argument("--ompthreads", type=int, help="OMP threads (overrides config)")
    parser.add_argument("--walltime", help="Walltime HH:MM:SS (overrides config)")
    parser.add_argument(
        "--launcher",
        help="MPI launcher (overrides config)",
    )
    
    # Executable path (override config)
    parser.add_argument(
        "--executable",
        help="Path to gb-demo executable (overrides config)",
    )
    
    # Modules and environment (override config)
    parser.add_argument("--modules", nargs="*", help="Modules to load (overrides config)")
    parser.add_argument("--mpi-options", nargs="*", help="MPI options (overrides config)")
    
    # Fugaku-specific (override config)
    parser.add_argument("--fugaku-gfscache", help="Fugaku GFS cache (overrides config)")
    parser.add_argument("--fugaku-spack-modules", nargs="*", help="Fugaku spack modules (overrides config)")
    parser.add_argument("--fugaku-mpi-options-for-pjm", nargs="*", help="Fugaku MPI options for PJM (overrides config)")
    parser.add_argument(
        "--fugaku-pjm-resources",
        nargs="*",
        help='Additional Fugaku PJM -L directives such as "freq=2000,eco_state=2" (overrides config)',
    )
    
    # Block names (override config)
    parser.add_argument(
        "--command-block-name-ext",
        help="Name for ExtSQD command block (overrides config)",
    )
    parser.add_argument(
        "--command-block-name-trim",
        help="Name for TrimSQD command block (overrides config)",
    )
    parser.add_argument(
        "--command-block-name-init",
        help="Name for init subcommand block (overrides config)",
    )
    parser.add_argument(
        "--command-block-name-recovery",
        help="Name for recovery subcommand block (overrides config)",
    )
    parser.add_argument(
        "--command-block-name-finalize",
        help="Name for finalize subcommand block (overrides config)",
    )
    parser.add_argument(
        "--execution-profile-block-name",
        help="(Legacy) Name for Ext/Trim execution profile block (overrides config)",
    )
    parser.add_argument(
        "--execution-profile-block-name-ext",
        help="Name for ExtSQD execution profile block (overrides config)",
    )
    parser.add_argument(
        "--execution-profile-block-name-trim",
        help="Name for TrimSQD execution profile block (overrides config)",
    )
    parser.add_argument(
        "--execution-profile-block-name-init",
        help="Name for init execution profile block (overrides config)",
    )
    parser.add_argument(
        "--execution-profile-block-name-recovery",
        help="Name for recovery execution profile block (overrides config)",
    )
    parser.add_argument(
        "--execution-profile-block-name-finalize",
        help="Name for finalize execution profile block (overrides config)",
    )
    parser.add_argument(
        "--hpc-profile-block-name",
        help="Name for HPC profile block (overrides config)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to create blocks."""
    args = _parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        config = _load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    
    # Merge config and CLI args (CLI args take precedence)
    def get_value(arg_name: str, config_key: str | None = None, default=None):
        """Get value from CLI args, config, or default."""
        key = config_key if config_key is not None else arg_name
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            return arg_value
        return config.get(key, default)
    
    # Required parameters
    hpc_target = get_value("hpc_target")
    resource_class = str(get_value("resource_class", default="cpu")).strip().lower()
    project = get_value("project")
    queue = get_value("queue")
    work_dir = get_value("work_dir")

    if not all([hpc_target, project, queue, work_dir]):
        print("Error: Missing required parameters.")
        print("Either provide --config with a TOML file, or specify:")
        print("  --hpc-target, --project, --queue, --work-dir")
        sys.exit(1)
    if resource_class not in {"cpu", "gpu"}:
        print(f"Error: Unsupported resource_class={resource_class!r}. Use 'cpu' or 'gpu'.")
        sys.exit(1)
    
    CommandBlock, ExecutionProfileBlock, HPCProfileBlock = _import_block_classes()
    _register_block_types(CommandBlock, ExecutionProfileBlock, HPCProfileBlock)
    
    is_miyabi = hpc_target == "miyabi"
    
    # Get execution parameters
    num_nodes = get_value("num_nodes", default=1)
    mpiprocs = get_value("mpiprocs", default=1)
    ompthreads = get_value("ompthreads", default=None if is_miyabi else 48)
    walltime = get_value("walltime", default="01:00:00")
    
    # Determine launcher
    launcher = get_value("launcher")
    if launcher is None:
        if is_miyabi:
            launcher = "mpiexec.hydra" if resource_class == "cpu" else "mpirun"
        else:
            launcher = "mpirun"
    
    # Get modules
    modules = get_value("modules")
    if modules is None:
        if is_miyabi:
            modules = ["intel/2023.2.0", "impi/2021.10.0"] if resource_class == "cpu" else []
        else:
            modules = ["LLVM/llvmorg-21.1.0"]
    
    # Get MPI options
    mpi_options = get_value("mpi_options")
    if mpi_options is None:
        if is_miyabi:
            mpi_options = [] if resource_class == "cpu" else ["-n", str(num_nodes * mpiprocs)]
        else:
            mpi_options = ["-n", str(num_nodes)]
    
    # Determine executable path
    executable = get_value("executable")
    if executable:
        executable_path = executable
    else:
        script_dir = Path(__file__).parent
        executable_path = _default_executable_path(
            script_dir=script_dir,
            hpc_target=hpc_target,
            resource_class=resource_class,
        )

    target_name = _default_target_name(hpc_target=hpc_target, resource_class=resource_class)

    # Block names
    cmd_block_ext = get_value("command_block_name_ext", default="cmd-gb-sqd-ext")
    cmd_block_trim = get_value("command_block_name_trim", default="cmd-gb-sqd-trim")
    cmd_block_init = get_value("command_block_name_init", default="cmd-gb-sqd-init")
    cmd_block_recovery = get_value("command_block_name_recovery", default="cmd-gb-sqd-recovery")
    cmd_block_finalize = get_value("command_block_name_finalize", default="cmd-gb-sqd-finalize")

    legacy_exec_block_name = get_value("execution_profile_block_name")
    exec_block_ext = get_value(
        "execution_profile_block_name_ext",
        default=legacy_exec_block_name or f"exec-gb-sqd-ext-{target_name}",
    )
    exec_block_trim = get_value(
        "execution_profile_block_name_trim",
        default=legacy_exec_block_name or f"exec-gb-sqd-trim-{target_name}",
    )
    exec_block_init = get_value(
        "execution_profile_block_name_init",
        default=f"exec-gb-sqd-init-{target_name}",
    )
    exec_block_recovery = get_value(
        "execution_profile_block_name_recovery",
        default=f"exec-gb-sqd-recovery-{target_name}",
    )
    exec_block_finalize = get_value(
        "execution_profile_block_name_finalize",
        default=f"exec-gb-sqd-finalize-{target_name}",
    )
    hpc_block_name = get_value("hpc_profile_block_name") or (
        f"hpc-{target_name}-gb-sqd"
    )
    
    # Create CommandBlocks
    print("Creating CommandBlocks...")
    
    CommandBlock(
        command_name="gb-sqd-ext",
        executable_key="gb_sqd",
        description="GB SQD ExtSQD workflow",
        default_args=["--mode", "ext_sqd"],
    ).save(cmd_block_ext, overwrite=True)
    print(f"  ✓ {cmd_block_ext}")
    
    CommandBlock(
        command_name="gb-sqd-trim",
        executable_key="gb_sqd",
        description="GB SQD TrimSQD workflow",
        default_args=["--mode", "trim_sqd"],
    ).save(cmd_block_trim, overwrite=True)
    print(f"  ✓ {cmd_block_trim}")

    CommandBlock(
        command_name="gb-sqd-init",
        executable_key="gb_sqd",
        description="GB SQD init subcommand",
        default_args=["init"],
    ).save(cmd_block_init, overwrite=True)
    print(f"  ✓ {cmd_block_init}")

    CommandBlock(
        command_name="gb-sqd-recovery",
        executable_key="gb_sqd",
        description="GB SQD recovery subcommand",
        default_args=["recovery"],
    ).save(cmd_block_recovery, overwrite=True)
    print(f"  ✓ {cmd_block_recovery}")

    CommandBlock(
        command_name="gb-sqd-finalize",
        executable_key="gb_sqd",
        description="GB SQD finalize subcommand",
        default_args=["finalize"],
    ).save(cmd_block_finalize, overwrite=True)
    print(f"  ✓ {cmd_block_finalize}")
    
    # Create ExecutionProfileBlock
    print("\nCreating ExecutionProfileBlock...")
    
    def _save_execution_profile(*, block_name: str, profile_name: str, command_name: str) -> None:
        environments = (
            {
                "KMP_AFFINITY": "granularity=fine,compact,1,0",
            }
            if is_miyabi
            else {
                "UTOFU_SWAP_PROTECT": "1",
                "LD_LIBRARY_PATH": "/lib64:$LD_LIBRARY_PATH",
            }
        )
        if not is_miyabi and ompthreads is not None:
            environments["OMP_NUM_THREADS"] = str(ompthreads)
        ExecutionProfileBlock(
            profile_name=profile_name,
            command_name=command_name,
            resource_class=resource_class,
            num_nodes=num_nodes,
            mpiprocs=mpiprocs,
            ompthreads=ompthreads,
            walltime=walltime,
            launcher=launcher,
            mpi_options=mpi_options,
            modules=modules,
            environments=environments,
        ).save(block_name, overwrite=True)
        print(f"  ✓ {block_name}")

    _save_execution_profile(
        block_name=exec_block_ext,
        profile_name=f"gb-sqd-ext-{target_name}",
        command_name="gb-sqd-ext",
    )
    _save_execution_profile(
        block_name=exec_block_trim,
        profile_name=f"gb-sqd-trim-{target_name}",
        command_name="gb-sqd-trim",
    )
    _save_execution_profile(
        block_name=exec_block_init,
        profile_name=f"gb-sqd-init-{target_name}",
        command_name="gb-sqd-init",
    )
    _save_execution_profile(
        block_name=exec_block_recovery,
        profile_name=f"gb-sqd-recovery-{target_name}",
        command_name="gb-sqd-recovery",
    )
    _save_execution_profile(
        block_name=exec_block_finalize,
        profile_name=f"gb-sqd-finalize-{target_name}",
        command_name="gb-sqd-finalize",
    )
    
    # Create HPCProfileBlock
    print("\nCreating HPCProfileBlock...")
    
    if is_miyabi:
        queue_cpu = queue if resource_class == "cpu" else "regular-c"
        queue_gpu = queue if resource_class == "gpu" else "regular-g"
        HPCProfileBlock(
            hpc_target="miyabi",
            queue_cpu=queue_cpu,
            queue_gpu=queue_gpu,
            project_cpu=project,
            project_gpu=project,
            executable_map={"gb_sqd": executable_path},
        ).save(hpc_block_name, overwrite=True)
    else:
        fugaku_gfscache = get_value("fugaku_gfscache", default="/vol0004:/vol0002")
        fugaku_spack_modules = get_value("fugaku_spack_modules", default=[])
        fugaku_mpi_options = get_value("fugaku_mpi_options_for_pjm", default=["max-proc-per-node=1"])
        fugaku_pjm_resources = get_value("fugaku_pjm_resources", default=["freq=2000,eco_state=2"])
        
        HPCProfileBlock(
            hpc_target="fugaku",
            queue_cpu=queue,
            queue_gpu=queue,
            project_cpu=project,
            project_gpu=project,
            executable_map={"gb_sqd": executable_path},
            gfscache=fugaku_gfscache,
            spack_modules=fugaku_spack_modules,
            mpi_options_for_pjm=fugaku_mpi_options,
            pjm_resources=fugaku_pjm_resources,
        ).save(hpc_block_name, overwrite=True)
    
    print(f"  ✓ {hpc_block_name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ Blocks created successfully!")
    print("=" * 60)
    if args.config:
        print(f"Configuration: {args.config}")
    print(f"HPC Target: {hpc_target}")
    print(f"Resource Class: {resource_class}")
    print(f"Project: {project}")
    print(f"Queue: {queue}")
    print(f"Work Directory: {work_dir}")
    print(f"Executable: {executable_path}")
    print(f"\nCommand Blocks:")
    print(f"  - {cmd_block_ext}")
    print(f"  - {cmd_block_trim}")
    print(f"  - {cmd_block_init}")
    print(f"  - {cmd_block_recovery}")
    print(f"  - {cmd_block_finalize}")
    print("Execution Profile Blocks:")
    print(f"  - {exec_block_ext}")
    print(f"  - {exec_block_trim}")
    print(f"  - {exec_block_init}")
    print(f"  - {exec_block_recovery}")
    print(f"  - {exec_block_finalize}")
    print(f"HPC Profile Block: {hpc_block_name}")
    print("\nNext steps:")
    print("1. Build the gb-demo executable if not already built:")
    print("   cd native && ./build_gb_sqd.sh")
    print("2. Run the workflow:")
    print("   Use these block names in ext_sqd_flow / trim_sqd_flow parameters:")
    print(f"     init_command_block_name={cmd_block_init}")
    print(f"     recovery_command_block_name={cmd_block_recovery}")
    print(f"     finalize_command_block_name={cmd_block_finalize}")
    print(f"     init_execution_profile_block_name={exec_block_init}")
    print(f"     recovery_execution_profile_block_name={exec_block_recovery}")
    print(f"     finalize_execution_profile_block_name={exec_block_finalize}")
    print(f"     hpc_profile_block_name={hpc_block_name}")


if __name__ == "__main__":
    main()
