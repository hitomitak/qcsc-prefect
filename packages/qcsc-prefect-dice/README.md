# QCSC Prefect DICE

Shared Python-side DICE SHCI solver integration for qcsc-prefect workflows.

This package provides:

- `DiceSHCISolverJob` as a Prefect block
- DICE input/output utilities
- Block registration and block creation helpers for Miyabi and Fugaku
- Scheduler script integration and runtime executable preflight checks

It is intended to be reused by multiple algorithms such as SQD and SKQD. Installing this
package does not build, download, vendor, or install the external DICE/SBD executable.
The executable must be compiled separately for each target HPC environment.

Build instructions are site-specific because compilers, MPI stacks, BLAS/LAPACK libraries,
GPU support, filesystem layout, and scheduler environments differ across HPC systems.

Source-tree native build helper assets live under:

```text
packages/qcsc-prefect-dice/native
```

They are not part of the Python package install. See [native/README.md](./native/README.md)
for development notes and how to point `dice_executable` at the separately built binary.

## Usage Example

```python
from qcsc_prefect_dice import DiceSHCISolverJob, create_dice_blocks

block_names = create_dice_blocks(
    hpc_target="miyabi",
    project="gz00",
    queue="regular-c",
    root_dir="/work/gz00/<user>/dice_jobs",
    dice_executable="/work/gz00/<user>/dice/bin/Dice",
    solver_block_name="sqd-dice-solver",
)

solver = await DiceSHCISolverJob.load(block_names["solver_block_name"])
result = await solver.run(
    ci_strings=ci_strings,
    one_body_tensor=h1,
    two_body_tensor=h2,
    norb=norb,
    nelec=nelec,
)
```

`SQD` and `SKQD` can use the same shared package and only vary block names or
their algorithm-specific setup wrappers.

The configured executable path is stored in the HPC profile executable map under the command
block key `dice_solver`. Generated job scripts validate that path on the HPC node before
launching DICE and fail early with a clear message if it is missing or not executable.
