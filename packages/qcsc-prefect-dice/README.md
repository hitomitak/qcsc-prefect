# QCSC Prefect DICE

Shared DICE SHCI solver integration for qcsc-prefect workflows.

This package provides:

- `DiceSHCISolverJob` as a Prefect block
- DICE input/output utilities
- Block registration and block creation helpers for Miyabi and Fugaku

It is intended to be reused by multiple algorithms such as SQD and SKQD.

Native build assets live under:

```text
packages/qcsc-prefect-dice/native
```

See [native/README.md](/Users/hitomi/Project/qcsc-prefect/packages/qcsc-prefect-dice/native/README.md)
for build instructions and how to point `dice_executable` at the resulting
binary.

## Usage Example

```python
from qcsc_prefect_dice import DiceSHCISolverJob, create_dice_blocks

block_names = create_dice_blocks(
    hpc_target="miyabi",
    project="gz00",
    queue="regular-c",
    root_dir="/work/gz00/<user>/dice_jobs",
    dice_executable="/work/gz00/<user>/qcsc-prefect-dice/native/bin/Dice",
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
