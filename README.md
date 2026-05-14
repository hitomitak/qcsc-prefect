# QCSC Prefect

This repository provides a modular workspace for portable HPC workflow orchestration with Prefect.
It is designed so the same workflow code can run across multiple HPC systems by switching profile blocks.

The workspace is organized into core packages, optional integrations, and a PyPI meta-package:

```
qcsc-prefect/
├── docs
│   └── concept.md
├── examples
│   ├── fugaku_prefect_hello_demo
│   ├── prefect_bitcount_demo
│   └── miyabi_prefect_hello_demo
├── packages
│   ├── qcsc-prefect
│   ├── qcsc-prefect-core
│   ├── qcsc-prefect-blocks
│   ├── qcsc-prefect-adapters
│   ├── qcsc-prefect-executor
│   ├── qcsc-prefect-qiskit
│   └── qcsc-prefect-dice
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Repository Structure

- `packages/qcsc-prefect/`
  Meta-package for installing the core QCSC Prefect packages from PyPI.
- `packages/qcsc-prefect-core/`
  Common execution model definitions (for example `ExecutionProfile`) shared by all targets.
- `packages/qcsc-prefect-blocks/`
  Prefect Block schemas for command, execution profile, and HPC profile layers.
- `packages/qcsc-prefect-adapters/`
  Target-specific script builders and runtime adapters (currently Miyabi/PBS and Fugaku/PJM).
- `packages/qcsc-prefect-executor/`
  High-level execution entrypoints that resolve blocks, derive scheduler routing,
  and dispatch to target runtimes.
- `packages/qcsc-prefect-qiskit/`
  Optional native Qiskit Runtime integration utilities.
- `packages/qcsc-prefect-dice/`
  Optional DICE SHCI solver integration.
- `examples/`
  End-to-end runnable examples for Miyabi and Fugaku.
- `docs/`
  Concept and architecture documents for the block-based execution model.

## Installation

Install the core QCSC Prefect packages from PyPI with:

```bash
pip install qcsc-prefect
```

Install the core packages plus the native Qiskit integration with:

```bash
pip install "qcsc-prefect[qiskit]"
```

Install the core packages plus the Python-side DICE integration with:

```bash
pip install "qcsc-prefect[dice]"
```

Install all optional integrations with:

```bash
pip install "qcsc-prefect[all]"
```

The DICE integration can also be installed directly:

```bash
pip install qcsc-prefect-dice
```

For source development, install the workspace with uv from the repository root:

```bash
git clone https://github.com/qiskit-community/qcsc-prefect.git
cd qcsc-prefect
uv sync --all-packages
uv run pytest
```

The root `pyproject.toml` is a workspace coordinator and is not the package published to PyPI.
Each package under `packages/` remains independently buildable.

## DICE Integration

`qcsc-prefect-dice` is a Python-side integration package. Installing it provides Prefect
tasks/blocks, scheduler templates, DICE input/output handling, and documentation. It does
not build, download, vendor, or install the external DICE/SBD executable.

The DICE/SBD executable must already be compiled on the target HPC system. Build instructions
are site-specific because compilers, MPI stacks, BLAS/LAPACK libraries, GPU support, filesystem
layout, and scheduler environments differ across HPC systems.

Configure the DICE executable through the command configuration used by the target HPC profile.
In the current block model, the `CommandBlock` names the executable key and the `HPCProfileBlock`
maps that key to the executable path visible on the target HPC system:

```python
from qcsc_prefect_blocks.common.blocks import CommandBlock, HPCProfileBlock

CommandBlock(
    command_name="dice",
    executable_key="dice_solver",
    description="DICE SHCI solver executable",
    default_args=[],
).save("cmd-dice-solver", overwrite=True)

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

Generated Miyabi, Fugaku, and Slurm job scripts include a preflight check that runs on the
HPC node before launching the job. The script fails early with a clear message if the configured
executable path is empty, missing, not executable, or not found on `PATH`.

See [DICE Integration](./docs/howto/howto_use_dice_prefect.md) for a longer setup example.

## Documentation

- Concept and architecture:
  - [HPC-Prefect Concept](./docs/concept.md)
- Example guides:
  - [BitCount Tutorial for Miyabi](./docs/tutorials/create_qcsc_workflow_for_miyabi.md)
  - [BitCount Tutorial for Fugaku](./docs/tutorials/create_qcsc_workflow_for_fugaku.md)
  - [Native Qiskit on Prefect](./docs/howto/howto_use_native_qiskit_prefect.md)
  - [Miyabi Hello Demo](./examples/miyabi_prefect_hello_demo/README.md)
  - [Fugaku Hello Demo](./examples/fugaku_prefect_hello_demo/README.md)

## Code Management

Code quality checks are configured with pre-commit (`.pre-commit-config.yaml`):

- `ruff check --fix`
- `ruff format`
- `bandit` (Python security checks for medium/high-confidence findings)
- `detect-secrets` (secret scanning backed by `.secrets.baseline`)
- basic repository hygiene hooks (`check-yaml`, trailing whitespace, EOF fix, merge conflict checks)

## Versioning Policy

Each sub-package under `packages/` maintains its own version in its own `pyproject.toml`.
The root project is a workspace coordinator (`qcsc-prefect-workspace`) and is not intended for distribution.
For a coordinated PyPI release, keep the `qcsc-prefect-*` package versions and exact internal
dependency pins aligned, for example `0.1.0`.

## Release Checklist

Maintainers can publish through GitHub Actions with PyPI Trusted Publishing.
Do not add PyPI API tokens or password secrets to this repository.

1. Update versions in every publishable package under `packages/`.
2. Update exact internal dependency pins between `qcsc-prefect-*` packages.
3. Run `python -m pip install --upgrade build twine`.
4. Run `bash scripts/build-all-packages.sh`.
5. Run `python -m twine check dist/*`.
6. Test local wheel installs from `dist/`, including `qcsc-prefect`,
   `qcsc-prefect[qiskit]`, `qcsc-prefect[dice]`, and `qcsc-prefect[all]`.
7. Confirm PyPI or TestPyPI Trusted Publisher settings match the workflow and environment.
8. Push a tag like `v0.1.0` to trigger the PyPI publish workflow.

## Contribution Guidelines

1. Install pre-commit hooks:
   - `pre-commit install`
2. Run checks before commit:
   - `pre-commit run --all-files`
   - or let them run automatically on `git commit`
3. Run tests as needed:
   - `uv run pytest`

When adding a new HPC target, include:

- adapter implementation under `packages/qcsc-prefect-adapters/`
- executor integration under `packages/qcsc-prefect-executor/`
- at least one runnable example under `examples/`
