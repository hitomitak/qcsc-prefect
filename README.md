# QCSC Prefect

This repository provides a modular workspace for portable HPC workflow orchestration with Prefect.
It is designed so the same workflow code can run across multiple HPC systems by switching profile blocks.

The workspace is organized into four core packages:

```
qcsc-prefect/
├── docs
│   └── concept.md
├── examples
│   ├── fugaku_prefect_hello_demo
│   ├── prefect_bitcount_demo
│   └── miyabi_prefect_hello_demo
├── packages
│   ├── qcsc-prefect-core
│   ├── qcsc-prefect-blocks
│   ├── qcsc-prefect-adapters
│   └── qcsc-prefect-executor
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Repository Structure

- `packages/qcsc-prefect-core/`
  Common execution model definitions (for example `ExecutionProfile`) shared by all targets.
- `packages/qcsc-prefect-blocks/`
  Prefect Block schemas for command, execution profile, and HPC profile layers.
- `packages/qcsc-prefect-adapters/`
  Target-specific script builders and runtime adapters (currently Miyabi/PBS and Fugaku/PJM).
- `packages/qcsc-prefect-executor/`
  High-level execution entrypoints that resolve blocks, derive scheduler routing,
  and dispatch to target runtimes.
- `examples/`
  End-to-end runnable examples for Miyabi and Fugaku.
- `docs/`
  Concept and architecture documents for the block-based execution model.

## Documentation

- Concept and architecture:
  - [HPC-Prefect Concept](./docs/concept.md)
- Example guides:
  - [BitCount Tutorial for Miyabi](./docs/tutorials/create_qcsc_workflow_for_miyabi.md)
  - [BitCount Tutorial for Fugaku](./docs/tutorials/create_qcsc_workflow_for_fugaku.md)
  - [Miyabi Hello Demo](./examples/miyabi_prefect_hello_demo/README.md)
  - [Fugaku Hello Demo](./examples/fugaku_prefect_hello_demo/README.md)

## Code Management

Code quality checks are configured with pre-commit (`.pre-commit-config.yaml`):

- `ruff check --fix`
- `ruff format`
- basic repository hygiene hooks (`check-yaml`, trailing whitespace, EOF fix, merge conflict checks)

## Versioning Policy

Each sub-package under `packages/` maintains its own version in its own `pyproject.toml`.
The root project is a workspace coordinator (`qcsc-prefect-workspace`) and is not intended for distribution.

## Contribution Guidelines

1. Install pre-commit hooks:
   - `pre-commit install`
2. Run checks before commit:
   - `pre-commit run --all-files`
3. Run tests as needed:
   - `uv run pytest`

When adding a new HPC target, include:

- adapter implementation under `packages/qcsc-prefect-adapters/`
- executor integration under `packages/qcsc-prefect-executor/`
- at least one runnable example under `examples/`
