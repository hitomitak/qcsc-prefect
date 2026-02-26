"""Create Prefect blocks for GB SQD workflows."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path


def _import_block_classes():
    """Import block classes from hpc-prefect packages."""
    # Add hpc-prefect packages to path if running in development mode
    project_root = Path(__file__).resolve().parents[2]
    if (project_root / "packages").exists():
        sys.path.insert(0, str(project_root / "packages" / "hpc-prefect-blocks" / "src"))
    
    from hpc_prefect_blocks.common.blocks import (
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
        "--execution-profile-block-name",
        help="Name for execution profile block (overrides config)",
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
        key = config_key if config_key is not None else arg_name.replace("_", "")
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            return arg_value
        return config.get(key, default)
    
    # Required parameters
    hpc_target = get_value("hpc_target")
    project = get_value("project")
    queue = get_value("queue")
    work_dir = get_value("work_dir")
    
    if not all([hpc_target, project, queue, work_dir]):
        print("Error: Missing required parameters.")
        print("Either provide --config with a TOML file, or specify:")
        print("  --hpc-target, --project, --queue, --work-dir")
        sys.exit(1)
    
    CommandBlock, ExecutionProfileBlock, HPCProfileBlock = _import_block_classes()
    _register_block_types(CommandBlock, ExecutionProfileBlock, HPCProfileBlock)
    
    is_miyabi = hpc_target == "miyabi"
    
    # Get execution parameters
    num_nodes = get_value("num_nodes", default=1)
    mpiprocs = get_value("mpiprocs", default=1)
    ompthreads = get_value("ompthreads", default=48)
    walltime = get_value("walltime", default="01:00:00")
    
    # Determine launcher
    launcher = get_value("launcher")
    if launcher is None:
        launcher = "mpiexec.hydra" if is_miyabi else "mpiexec"
    
    # Get modules
    modules = get_value("modules")
    if modules is None:
        modules = ["intel/2023.2.0", "impi/2021.10.0"] if is_miyabi else []
    
    # Get MPI options
    mpi_options = get_value("mpi_options", default=[])
    
    # Determine executable path
    executable = get_value("executable")
    if executable:
        executable_path = executable
    else:
        # Default: gb_demo_2026/build/gb-demo relative to this script
        script_dir = Path(__file__).parent
        executable_path = str((script_dir / "gb_demo_2026/build/gb-demo").resolve())
    
    # Block names
    cmd_block_ext = get_value("command_block_name_ext", default="cmd-gb-sqd-ext")
    cmd_block_trim = get_value("command_block_name_trim", default="cmd-gb-sqd-trim")
    exec_block_name = get_value("execution_profile_block_name") or (
        f"exec-gb-sqd-{'miyabi' if is_miyabi else 'fugaku'}"
    )
    hpc_block_name = get_value("hpc_profile_block_name") or (
        f"hpc-{'miyabi' if is_miyabi else 'fugaku'}-gb-sqd"
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
    
    # Create ExecutionProfileBlock
    print("\nCreating ExecutionProfileBlock...")
    
    ExecutionProfileBlock(
        profile_name=f"gb-sqd-{'miyabi' if is_miyabi else 'fugaku'}",
        command_name="gb-sqd-ext",  # Can be used for both ext and trim
        resource_class="cpu",
        num_nodes=num_nodes,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        walltime=walltime,
        launcher=launcher,
        mpi_options=mpi_options,
        modules=modules,
        environments={
            "OMP_NUM_THREADS": str(ompthreads),
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
        },
    ).save(exec_block_name, overwrite=True)
    print(f"  ✓ {exec_block_name}")
    
    # Create HPCProfileBlock
    print("\nCreating HPCProfileBlock...")
    
    if is_miyabi:
        HPCProfileBlock(
            hpc_target="miyabi",
            queue_cpu=queue,
            queue_gpu="regular-g",
            project_cpu=project,
            project_gpu=project,
            executable_map={"gb_sqd": executable_path},
        ).save(hpc_block_name, overwrite=True)
    else:
        fugaku_gfscache = get_value("fugaku_gfscache", default="/vol0004:/vol0002")
        fugaku_spack_modules = get_value("fugaku_spack_modules", default=[])
        fugaku_mpi_options = get_value("fugaku_mpi_options_for_pjm", default=["max-proc-per-node=1"])
        
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
        ).save(hpc_block_name, overwrite=True)
    
    print(f"  ✓ {hpc_block_name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ Blocks created successfully!")
    print("=" * 60)
    if args.config:
        print(f"Configuration: {args.config}")
    print(f"HPC Target: {hpc_target}")
    print(f"Project: {project}")
    print(f"Queue: {queue}")
    print(f"Work Directory: {work_dir}")
    print(f"Executable: {executable_path}")
    print(f"\nCommand Blocks:")
    print(f"  - {cmd_block_ext}")
    print(f"  - {cmd_block_trim}")
    print(f"Execution Profile Block: {exec_block_name}")
    print(f"HPC Profile Block: {hpc_block_name}")
    print("\nNext steps:")
    print("1. Build the gb-demo executable if not already built:")
    print("   cd native && ./build_gb_sqd.sh")
    print("2. Run the workflow:")
    print(f"   python -m gb_sqd.main \\")
    print(f"     --command-block {cmd_block_ext} \\")
    print(f"     --execution-profile {exec_block_name} \\")
    print(f"     --hpc-profile {hpc_block_name}")


if __name__ == "__main__":
    main()

