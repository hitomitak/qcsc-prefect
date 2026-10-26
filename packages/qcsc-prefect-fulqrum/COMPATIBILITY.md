# Local compatibility adapter for QCSC Prefect 0.3.0

This package exists only on the integration-validation branch. It is not a
published QCSC Prefect release and is not part of the root workspace.

The Python modules, original README and native build script are copied unchanged
from PR #42 commit `2d435b19af4648cbbc5fbc6a27ff258c93817d1a`. A manifest under
`packages/qcsc-prefect-qiskit/tests/fixtures/pr42/postprocessing.json` verifies
their hashes. The original README describes that fixed source; use the
[integration guide](../../examples/qrmi_sqd_validation/README_recovery.md) for
the current branch's installation and execution instructions.

Only `pyproject.toml` is adapted: the local package version is `0.3.0.dev0`, and
blocks/core/executor dependencies require `==0.3.0` instead of `==0.2.5`.
Install this directory explicitly in editable mode with the recovery application;
installing the native Fulqrum backend is a separate host-specific step.

The application provides a strict Slurm adapter that requires a successful job
before decoding solver output. It reuses this package's block configuration,
solver options and file protocol without changing the pinned wrapper source.
Its local reference path uses PySCF and does not establish native Fulqrum
execution. Native numerical execution remains unverified on the current Mac
because its compiler lacks the required OpenMP build support.
