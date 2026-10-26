# Saved H2 results: configuration recovery and Fulqrum integration

This branch adds a separate postprocessing entrypoint to the sampling adapter.
It reads an already completed H2/STO-3G run produced with `QrmiRuntime` 0.3.0,
validates the saved evidence, and calls the configuration-recovery functions
from PR #42 at commit `2d435b19af4648cbbc5fbc6a27ff258c93817d1a`.
The original PR and the released QCSC Prefect libraries are unchanged.

For Git deployment, an isolated cluster environment, the saved eight-shot input,
dedicated classical Slurm blocks and result/FCI checks, follow the
[cluster validation runbook](README_recovery_cluster.md). Its execution commands
are instructions for a subsequent user-operated run and have not been run here.

```text
Saved sampling run (ordered meas, optional test, raw result and job metadata)
  -> verify provenance, registers, raw shot selection and counts
  -> BitArray -> configurations and probabilities
  -> pinned configuration recovery and subspace selection
  -> explicitly selected reference / native Fulqrum / classical Slurm solver
  -> electronic energy + nuclear repulsion once -> recovery_summary.json
```

## Dependency boundary

| Component | Version / source | Installation |
| --- | --- | --- |
| QCSC Prefect workspace libraries | `0.3.0`, unchanged from `v0.3.0` | Workspace editable packages |
| Sampling application | `qcsc-qrmi-pr42-validation==0.1.0` | Editable with `[recovery]` extra |
| Workflow chemistry utility | `0.1.0` from this checkout | Editable |
| Local Fulqrum adapter | `qcsc-prefect-fulqrum==0.3.0.dev0` | Explicit local editable installation |
| SQD helper library | `qiskit-addon-sqd==0.11.0` | Application recovery extra |
| Native Fulqrum backend | Commit `52d7c66ab0986580b00ad9382fe4d67bbe4d883e` required by the pinned build script | Separate native build for the target host |

The local Fulqrum adapter is copied from the fixed PR source. Its six Python
modules, original README and native build script remain byte-identical; a
SHA-256 manifest verifies them before replay. Only package metadata changes:
the local adapter version becomes `0.3.0.dev0`, and its blocks/core/executor pins
become `==0.3.0` instead of `==0.2.5`. The original metadata is also retained as
a verified snapshot. This avoids downgrading the runtime libraries to 0.2.5.

`0.3.0.dev0` identifies an unreleased, branch-local compatibility adapter. It is
not a new QCSC Prefect release. The package is deliberately absent from the root
workspace and release configuration; install it explicitly. No native backend
is installed by the application's recovery extra.

The recovery/subsampling snapshots are loaded in a private namespace. The only
AST adjustment redirects their internal module import to that namespace;
the recovery function bodies are unchanged. Keep the entire source checkout
available because the application verifies fixtures from it.

## Create a separate local environment

From this branch's repository root:

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/qrmi-pr42-recovery-venv
uv sync --all-packages --python 3.12
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  -r examples/qrmi_sqd_validation/requirements.txt \
  -e algorithms/qcsc_workflow_utility \
  -e packages/qcsc-prefect-fulqrum \
  -e 'examples/qrmi_sqd_validation[recovery]'
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
```

Use `uv run --no-sync` for the commands below to retain the separately installed
scientific pins and local adapters. These setup commands do not submit jobs.

## Required saved input

Copy the **whole completed `run_<id>` directory**, retaining this structure:

```text
run_<id>/
  sampling_summary.json
  bitstrings.txt
  quantum/<job_id>/
    slurm_job.json
    sampler_request.json
    transpile_config.json
    transpile_report.json
    result.json
    circuit.qpy
    isa.qpy
    sampler_input.json
```

A printed counts dictionary alone is insufficient. The reader resolves files
relative to the supplied run directory, so moving a full run from `/shared`
to a local directory is supported without editing its recorded cluster paths.
It requires exactly one successful Batch quantum job with exit status zero,
the pinned sampling source/hash, library 0.3.0 and four-bit H2/STO-3G samples.
It decodes the raw result, repeats the original `test == 0` shot selection when
enabled, and compares retained shot order and counts with the saved files.
It hashes all ten input files and checks that they remain unchanged after replay.
QPY files are retained and hashed; they are not executed during recovery.

Both 100-shot `meas`-only runs and reset-mitigated runs that retain fewer shots
are accepted. A 100-shot request that retains eight shots uses those eight
shots; the code never synthesizes replacement samples or submits a new QPU job.

## Local reference replay

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync python \
  -m examples.qrmi_sqd_validation.recovery \
  --sampling-run /absolute/path/to/run_id \
  --output-root /absolute/path/to/recovery-results \
  --backend reference \
  --subspace-dim 4 --num-batches 1 --max-iterations 1 --seed 24
```

Replace the placeholder paths before running. Keep the output outside the input
run directory. The command prints a JSON summary and writes a new
`recovery_<id>` directory. This backend runs the pinned configuration recovery
with the PySCF selected-CI solver exposed by `qiskit-addon-sqd`. It needs no
Prefect server, network connection or native Fulqrum installation. It is a
reference calculation, and its summary explicitly records `backend: reference`.

For this two-orbital, one-alpha/one-beta H2 problem, a subspace dimension of four
can span the complete fixed-particle-number space. Matching the full-CI energy
therefore checks the plumbing and energy convention; it does not establish QPU
accuracy or the statistical quality of eight retained shots. The defaults above
are a small integration check, not the original PR's production SQD settings.

The Python API also requires an explicit backend:

```python
from examples.qrmi_sqd_validation.recovery import RecoverySettings, recover_saved_h2

summary = await recover_saved_h2(
    "/absolute/path/to/run_id",
    "/absolute/path/to/recovery-results",
    backend="reference",
    settings=RecoverySettings(max_iterations=2),
)
```

## Fulqrum execution paths

`--backend fulqrum-local` calls the unchanged adapter's `solve_fulqrum` directly.
It requires the native backend and its scientific dependencies on the local host.
There is no fallback to the reference solver when the backend is unavailable.
The pinned adapter expects `fulqrum>=0.5.0`; during this validation, the available
PyPI releases did not satisfy that requirement. The fixed PR's native build
script pins source commit `52d7c66ab0986580b00ad9382fe4d67bbe4d883e`.
The local macOS build failed because Apple Clang rejected `-fopenmp`.
No global compiler changes were made, so native Fulqrum numerical execution
remains unverified on this host.

`--backend fulqrum-block --solver-block <name>` loads a saved
`FulqrumSolverJob` block, writes its input/option files and submits classical
solver jobs through the 0.3.0 `run_job_from_blocks` executor. This path **does
submit Slurm workloads** and is reserved for a subsequent authorized cluster
validation. Its HPC profile must target Slurm and have no `slurm_qpu` value.
The compute environment must provide `qcsc-fulqrum-solve` and the native backend;
the command/execution/HPC blocks must already describe that environment.

The integration adapter requires `COMPLETED` and exit status zero before reading
solver outputs, and records each solver job directory and Slurm result. This
explicit check is stricter than the fixed PR's original solver-job wrapper,
which could continue after a nonzero exit. The original wrapper source remains
unchanged. No blocks are created or changed by this entrypoint.

## Results and failures

The output directory contains:

- `input_validation.json`: source provenance, settings and saved-input hashes.
- `integrals.npz`: H2 integrals, electron counts and nuclear repulsion energy.
- `recovery_input.npz`: configurations and probabilities passed to recovery.
- `solve_0000.json`, etc.: requested alpha/beta CI strings and completed solver
  energies/occupancies; a pending record remains if that call fails.
- `recovery_summary.json`: backend, requested/retained shots, solver calls,
  round records, electronic energy, nuclear repulsion and total energy.

The solver energy is electronic only. The pinned recovery adds the nuclear
repulsion once. Invalid input fails before solver invocation and output creation.
A failure after output creation writes `failure.json`, retains diagnostic files
and produces no success summary. Classical solver directories are retained too.

## Local verification and remaining validation

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_saved_recovery.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
```

Tests prohibit network access and quantum sampling. They cover ordered replay,
reset selection, two recovery rounds, independent PySCF full-CI comparison,
energy offsets, corrupted evidence/source snapshots, failure diagnostics,
0.3.0 dependency pins and CLI use from outside the checkout. The classical
Fulqrum boundary test exercises the real CLI and NPZ file protocol with a fake
scheduler and an explicitly substituted reference numerical core. It does not
execute native Fulqrum or Slurm. A separate native test runs only when Fulqrum
is installed; it is skipped on the current host.

Local verification completed on 2026-10-23 with Python 3.12.15: the full suite
reported **370 passed, 3 skipped, 2 xfailed**. The skips cover Fugaku/Miyabi
cluster integration and native Fulqrum. The expected failures cover the known
raw-LUCJ QPY attribute loss. Ruff checks/formatting passed, and `uv pip check`
reported compatible dependencies for all 152 installed packages.

The input fixture is synthetic and matches the user's reported eight-shot
counts; the user's actual cluster files are not present locally. User-reported
sampling success establishes a different validation stage from these tests.
Replay of those actual files, native Fulqrum execution and classical Slurm
execution remain to be checked. No quantum or cluster workload was run while
implementing this entrypoint.
