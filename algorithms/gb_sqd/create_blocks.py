"""Create Prefect blocks for GB SQD workflows."""

from __future__ import annotations

import argparse
import os
import sys
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


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create Prefect blocks for GB SQD workflows (Miyabi or Fugaku)."
    )
    
    parser.add_argument(
        "--hpc-target",
        choices=["miyabi", "fugaku"],
        required=True,
        help="Target HPC system",
    )
    parser.add_argument("--project", required=True, help="Project/group name")
    parser.add_argument("--queue", required=True, help="Queue/resource group name")
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Working directory for job outputs",
    )
    
    # Execution parameters
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--mpiprocs", type=int, default=1, help="MPI processes per node")
    parser.add_argument("--ompthreads", type=int, default=48, help="OMP threads")
    parser.add_argument("--walltime", default="01:00:00", help="Walltime (HH:MM:SS)")
    parser.add_argument(
        "--launcher",
        default=None,
        help="MPI launcher (default: mpiexec.hydra for Miyabi, mpiexec for Fugaku)",
    )
    
    # Executable path
    parser.add_argument(
        "--executable",
        help="Path to gb-demo executable (default: ../../build/gb-demo)",
    )
    
    # Modules and environment
    parser.add_argument("--modules", nargs="*", default=None, help="Modules to load")
    parser.add_argument("--mpi-options", nargs="*", default=None, help="MPI options")
    
    # Fugaku-specific
    parser.add_argument("--fugaku-gfscache", default="/vol0004:/vol0002")
    parser.add_argument("--fugaku-spack-modules", nargs="*", default=None)
    parser.add_argument("--fugaku-mpi-options-for-pjm", nargs="*", default=None)
    
    # Block names
    parser.add_argument(
        "--command-block-name-ext",
        default="cmd-gb-sqd-ext",
        help="Name for ExtSQD command block",
    )
    parser.add_argument(
        "--command-block-name-trim",
        default="cmd-gb-sqd-trim",
        help="Name for TrimSQD command block",
    )
    parser.add_argument(
        "--execution-profile-block-name",
        default=None,
        help="Name for execution profile block",
    )
    parser.add_argument(
        "--hpc-profile-block-name",
        default=None,
        help="Name for HPC profile block",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to create blocks."""
    args = _parse_args()
    
    CommandBlock, ExecutionProfileBlock, HPCProfileBlock = _import_block_classes()
    _register_block_types(CommandBlock, ExecutionProfileBlock, HPCProfileBlock)
    
    is_miyabi = args.hpc_target == "miyabi"
    
    # Determine defaults
    if args.launcher is None:
        launcher = "mpiexec.hydra" if is_miyabi else "mpiexec"
    else:
        launcher = args.launcher
    
    if args.modules is None:
        modules = ["intel/2023.2.0", "impi/2021.10.0"] if is_miyabi else []
    else:
        modules = args.modules
    
    if args.mpi_options is None:
        mpi_options = []
    else:
        mpi_options = args.mpi_options
    
    # Determine executable path
    if args.executable:
        executable_path = args.executable
    else:
        # Default: relative to this script
        script_dir = Path(__file__).parent
        executable_path = str((script_dir / "../../build/gb-demo").resolve())
    
    # Block names
    exec_block_name = args.execution_profile_block_name or (
        f"exec-gb-sqd-{'miyabi' if is_miyabi else 'fugaku'}"
    )
    hpc_block_name = args.hpc_profile_block_name or (
        f"hpc-{'miyabi' if is_miyabi else 'fugaku'}-gb-sqd"
    )
    
    # Create CommandBlocks
    print("Creating CommandBlocks...")
    
    CommandBlock(
        command_name="gb-sqd-ext",
        executable_key="gb_sqd",
        description="GB SQD ExtSQD workflow",
        default_args=["--mode", "ext_sqd"],
    ).save(args.command_block_name_ext, overwrite=True)
    print(f"  ✓ {args.command_block_name_ext}")
    
    CommandBlock(
        command_name="gb-sqd-trim",
        executable_key="gb_sqd",
        description="GB SQD TrimSQD workflow",
        default_args=["--mode", "trim_sqd"],
    ).save(args.command_block_name_trim, overwrite=True)
    print(f"  ✓ {args.command_block_name_trim}")
    
    # Create ExecutionProfileBlock
    print("\nCreating ExecutionProfileBlock...")
    
    ExecutionProfileBlock(
        profile_name=f"gb-sqd-{'miyabi' if is_miyabi else 'fugaku'}",
        command_name="gb-sqd-ext",  # Can be used for both ext and trim
        resource_class="cpu",
        num_nodes=args.num_nodes,
        mpiprocs=args.mpiprocs,
        ompthreads=args.ompthreads,
        walltime=args.walltime,
        launcher=launcher,
        mpi_options=mpi_options,
        modules=modules,
        environments={
            "OMP_NUM_THREADS": str(args.ompthreads),
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
        },
    ).save(exec_block_name, overwrite=True)
    print(f"  ✓ {exec_block_name}")
    
    # Create HPCProfileBlock
    print("\nCreating HPCProfileBlock...")
    
    if is_miyabi:
        HPCProfileBlock(
            hpc_target="miyabi",
            queue_cpu=args.queue,
            queue_gpu="regular-g",
            project_cpu=args.project,
            project_gpu=args.project,
            executable_map={"gb_sqd": executable_path},
        ).save(hpc_block_name, overwrite=True)
    else:
        fugaku_spack_modules = args.fugaku_spack_modules or []
        fugaku_mpi_options = args.fugaku_mpi_options_for_pjm or ["max-proc-per-node=1"]
        
        HPCProfileBlock(
            hpc_target="fugaku",
            queue_cpu=args.queue,
            queue_gpu=args.queue,
            project_cpu=args.project,
            project_gpu=args.project,
            executable_map={"gb_sqd": executable_path},
            gfscache=args.fugaku_gfscache,
            spack_modules=fugaku_spack_modules,
            mpi_options_for_pjm=fugaku_mpi_options,
        ).save(hpc_block_name, overwrite=True)
    
    print(f"  ✓ {hpc_block_name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ Blocks created successfully!")
    print("=" * 60)
    print(f"HPC Target: {args.hpc_target}")
    print(f"Project: {args.project}")
    print(f"Queue: {args.queue}")
    print(f"Work Directory: {args.work_dir}")
    print(f"Executable: {executable_path}")
    print(f"\nCommand Blocks:")
    print(f"  - {args.command_block_name_ext}")
    print(f"  - {args.command_block_name_trim}")
    print(f"Execution Profile Block: {exec_block_name}")
    print(f"HPC Profile Block: {hpc_block_name}")
    print("\nNext steps:")
    print("1. Build the gb-demo executable if not already built")
    print("2. Run the workflow:")
    print(f"   python -m gb_sqd.main \\")
    print(f"     --command-block {args.command_block_name_ext} \\")
    print(f"     --execution-profile {exec_block_name} \\")
    print(f"     --hpc-profile {hpc_block_name}")


if __name__ == "__main__":
    main()

