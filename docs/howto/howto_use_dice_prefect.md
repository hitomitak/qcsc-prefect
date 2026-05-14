# DICE Integration

`qcsc-prefect-dice` provides the Python-side integration for DICE/SBD workflows:

- Prefect block and task helpers
- Scheduler script templates
- DICE input file rendering
- DICE output parsing
- Documentation and block creation helpers

It does not build, download, vendor, or install the external DICE/SBD executable. The executable
must already be compiled separately for the target HPC environment.

## Installation

Install through the meta-package extra:

```bash
pip install "qcsc-prefect[dice]"
```

Install all optional integrations with:

```bash
pip install "qcsc-prefect[all]"
```

Direct installation also works:

```bash
pip install qcsc-prefect-dice
```

These commands install only Python dependencies and Python integration code. They do not compile
or install the DICE/SBD solver binary.

## External Solver Binary

Build the DICE/SBD executable on each target HPC system according to local site requirements.
Build instructions are site-specific because compilers, MPI implementations, BLAS/LAPACK
libraries, GPU support, filesystem layout, and scheduler environments differ across systems.

Use an executable path that is valid from the submitted HPC job. The local machine that creates
Prefect blocks does not need to be able to access that path.

## Command Block Configuration

The shared execution model resolves executable paths through the command key configured in
Prefect blocks. The `CommandBlock` declares the logical executable key, and the `HPCProfileBlock`
maps that key to the target-system executable path.

```python
from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock

CommandBlock(
    command_name="dice",
    executable_key="dice_solver",
    description="DICE SHCI solver executable",
    default_args=[],
).save("cmd-dice-solver", overwrite=True)

ExecutionProfileBlock(
    profile_name="dice-mpi",
    command_name="dice",
    resource_class="cpu",
    num_nodes=1,
    mpiprocs=4,
    walltime="01:00:00",
    launcher="mpiexec.hydra",
).save("exec-dice-mpi", overwrite=True)

HPCProfileBlock(
    hpc_target="miyabi",
    queue_cpu="regular-c",
    queue_gpu="regular-g",
    project_cpu="gz00",
    project_gpu="gz00",
    executable_map={
        "dice_solver": "/work/gz00/<user>/dice/bin/Dice",
    },
).save("hpc-miyabi-dice", overwrite=True)
```

The helper `qcsc_prefect_dice.create_dice_blocks()` creates the same block set and accepts the
target-system path through its `dice_executable` argument.

## Runtime Validation

Generated Miyabi, Fugaku, and Slurm scripts validate the configured executable at job runtime.
The preflight runs on the HPC node before the solver launch and fails early if the executable is
empty, missing, not executable, or not found on `PATH`.

The generated check is intentionally script-side rather than local Python-side, because the
machine creating Prefect blocks may not have access to the target HPC filesystem.
