# One H2 flow: QRMI sampling to configuration recovery and Fulqrum

`end_to_end.py` connects a **fresh** `QrmiRuntime` 0.3.0 sampling call, the
configuration recovery at PR #42 commit
`2d435b19af4648cbbc5fbc6a27ff258c93817d1a`, classical Slurm Fulqrum execution
and independent PySCF full-CI verification in one Prefect flow.
PR #42, its source snapshots and the released 0.3.0 library code are unchanged.

```text
Parent flow: qrmi-pr42-h2-fulqrum
  -> validate pinned source and classical / quantum HPC routes
  -> H2/STO-3G integrals and pinned LUCJ construction
  -> application ffSIM preparation -> QrmiRuntime sampling task -> Slurm/QRMI
  -> ordered meas BitArray, optionally selected by test == 0
  -> saved evidence validation -> pinned configuration recovery
  -> classical Slurm qcsc-fulqrum-solve -> SCIResult
  -> saved input hashes, job states, native output and full-CI comparison
  -> integration_summary.json
```

The sampling-only flow still returns `BitArray`. Its implementation is shared
with the parent flow through `sample_h2_run`, which also exposes the exact saved
run directory. The parent does not discover inputs by globbing old sampling
runs and cannot silently reuse the previous eight-shot result.

## Existing cluster setup

Use a checkout containing `end_to_end.py` and the dedicated control environment
installed with `examples/qrmi_sqd_validation[recovery]`. A checkout at the earlier
`e072c81` does not contain this entrypoint. Update the clean integration checkout
from `origin/qrmi-pr42-recovery-v0.3.0` with `git pull --ff-only` before running it.
Keep any existing local changes; do not reset or overwrite them to update.
Editable installation makes the new module available after updating the checkout.
There are no additional Python dependencies for this step.

Keep the successful QRMI environment and native Fulqrum build. The existing
`poc-qrmi-command`, `poc-qrmi-execution` and `poc-qrmi-hpc` blocks are used for
sampling on `test_heron`. Pass an existing **classical** solver block explicitly;
the user-reported successful block is
`qrmi-h2-recovery-49c599aef168-solver`. No block registration is needed.

For the previously reported cluster paths, the control environment is
`/shared/qrmi-pr42-recovery.YUxWez/control-venv`, and its editable checkout is
`/shared/qrmi-pr42-recovery.YUxWez/qcsc-prefect`. Update those paths if a different
deployment is used. The workflow host and compute nodes must retain access to
the required shared code, environments and solver directories. Keep the same
Prefect workspace authentication as the successful saved-result validation.

For a new setup, see the [saved-result cluster runbook](README_recovery_cluster.md).
That site's CPU partition was reported as `debug`. Building native Fulqrum also
required the matching Python 3.12 development headers. Those setup issues do
not require repeating block creation or rebuilding an already working backend.

## Run a new end-to-end validation

The following command **submits new quantum sampling and classical Slurm solver
jobs**. It is a command for a subsequent authorized hardware run; it has not been
executed as part of the local implementation work.

```bash
cd /shared/qrmi-pr42-recovery.YUxWez/qcsc-prefect
git status --short
git pull --ff-only origin qrmi-pr42-recovery-v0.3.0
export UV_PROJECT_ENVIRONMENT=/shared/qrmi-pr42-recovery.YUxWez/control-venv
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

uv run --no-sync python -m examples.qrmi_sqd_validation.end_to_end run \
  --solver-block qrmi-h2-recovery-49c599aef168-solver \
  --work-dir /shared/qrmi-pr42-end-to-end \
  --shots 100 --reset-mitigation --seed 24
```

Each invocation creates a unique `run_<id>` under `--work-dir`. The quantum
request is 100 shots; the number retained by reset mitigation is measured from
the **new** raw result and is not required to equal eight. Omitting
`--reset-mitigation` runs meas-only sampling. Empty retained data fails explicitly.

The acceptance check fixes the recovery to dimension four, one batch and one
iteration. For H2/STO-3G this can span the full one-alpha/one-beta space. If a
new sample does not provide the full four-dimensional subspace, verification
fails and retains the output; it does not manufacture additional samples or
automatically submit another quantum job. Parent flow and sampling task retries
and result persistence are disabled. There is no reference-solver fallback.

## Verify saved output without new jobs

The flow logs its integration directory and prints a JSON summary. Substitute
that exact directory for `run_id` below:

```bash
uv run --no-sync python -m examples.qrmi_sqd_validation.end_to_end verify \
  /shared/qrmi-pr42-end-to-end/run_id
```

This command reads saved evidence and performs the local FCI comparison. It
does not start a Prefect flow, call the Prefect API, query `sacct` or submit jobs.
There is no manual `sampling-input.json`, exported `SAVED_RUN`, or assumed
eight-shot count. The recovery's own pre-solve hashes identify the quantum input.

Success requires `validation: passed`, a four-bit H2 input with a positive
retained shot count, successful quantum and classical job records, unchanged
quantum input hashes, valid native Fulqrum output and total energy within
`1e-8` Hartree of independent PySCF full CI. Nuclear repulsion is added once.
The returned `quantum_job` and `solver_jobs` include the saved Slurm identifiers,
states and exit statuses; solver records include native metadata. Accounting
storage can remain disabled because `sacct` is not part of this verifier.

```text
run_<id>/
  integration_request.json     # pinned source, options and block name
  stages.json                  # exact sampling and recovery directories
  sampling/run_<id>/           # ordered shots and full quantum evidence
  recovery/recovery_<id>/      # pre-solve hashes, integrals and solver records
  integration_summary.json    # written only after successful verification
  failure.json                # present instead on a failed stage
```

The classical solver directories remain under the configured solver block's
root, and the recovery summary records their paths. Retain those directories
with the parent run. On failure, inspect the recorded `phase` and existing
quantum/solver logs before rerunning. A new `run` command resubmits quantum
sampling; a `verify` command never does. A solver timeout does not cancel Slurm
automatically, so check the existing job before retrying.

## Validation scope

On 2026-10-26 the user reported a successful **saved eight-shot** recovery run:
native Fulqrum total energy `-1.1372838344885021` Hartree, PySCF FCI
`-1.1372838344885023` Hartree and ten unchanged quantum input hashes.
That is evidence for the saved-result stage, not for this new fresh-sampling flow.

Local tests of the new flow use real H2 chemistry, LUCJ preparation, QPY,
target-specific compilation, raw-result decoding, pinned configuration recovery,
Fulqrum's CLI/NPZ protocol and FCI checks. Quantum readouts and the native
numerical core are explicit test doubles; the latter uses a reference solver
and marks its metadata `test-double`. Tests cover both reset modes, failed jobs,
empty selection, invalid routes, modified evidence, a fresh-interpreter offline
verifier and the actual Prefect parent flow/sampling task on a temporary local
server. No cluster or native Fulqrum execution is established by these tests.

The local full suite on 2026-10-26 passed with **382 passed, 3 skipped,
2 xfailed**, using `uv run --no-sync pytest`. The skips are native Fulqrum and
the existing Fugaku/Miyabi hardware tests; expected failures retain the known
raw-LUCJ QPY attribute-loss cases. Ruff checks/formatting and dependency checks
also passed.

Passing the subsequent hardware run establishes H2 integration of these pinned
PR #42 stages with QrmiRuntime 0.3.0. It does not execute the original PR's main
flow directly or establish larger-molecule/production settings, sampling
accuracy, statistical quality, or sparse/PRIMME solver paths.
