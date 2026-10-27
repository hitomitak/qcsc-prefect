# PR #42 unchanged calculation, a small QRMI patch, and a fast workflow check

This is the reviewer-oriented integration scenario:

1. Start from PR #42 commit `2d435b19af4648cbbc5fbc6a27ff258c93817d1a`.
2. Apply the displayed [QRMI boundary patch](pr42_qrmi.patch) to its `main.py`.
3. Run the preserved PR flow with the separate [fast parameter file](pr42_smoke_parameters.json).
4. Confirm workflow completion, successful scheduler jobs and native outputs.

PR #42 itself is never edited. The executable [pr42_workflow.py](pr42_workflow.py)
is a copy on this integration branch. The immutable original is retained in the
PR fixture. The patch exactly reconstructs the executable copy from that fixture;
tests apply the patch to the original file and compare the resulting bytes.

## What changed in main.py

The Runtime construction and primitive call are written directly in
`sample_bitstrings`. There is no bridge wrapper hiding the replacement:

```text
QuantumRuntime.load(block name)
  -> read runtime JSON -> QrmiRuntime(**runtime_config)

workflow-host get_target -> transpile -> sampler(ISA circuit)
  -> runtime.sampler(sampler_pubs=[(lucj_circ,)], options=options)
     with QPY preparation before Slurm and Target / transpilation inside Slurm
```

The executable copy contains the released `QrmiRuntime(scheduler="slurm")`
configuration, hook settings, direct sampler call and scheduler-evidence write.
The hook settings are visible in the same function:

```python
runtime_config.update(
    prepare=partial(prepare_sampler_circuit, circuit_parameters=circuit_params.model_dump()),
    transpiler="examples.qrmi_sqd_validation.pr42:transpile_for_target",
)
runtime = QrmiRuntime(**runtime_config)
```

The two transport callbacks remain in [pr42.py](pr42.py):
`prepare_sampler_circuit` applies ffSIM PRE_INIT/decomposition before QPY;
`transpile_for_target` calls the fixed PR transpiler after the compute node
acquires its Target. Raw ffSIM gates lose Python attributes through QPY, so
preparation must happen before transport. The compute-node callback must be
importable by name. These callbacks are explicit application-owned transport
hooks, not a replacement sampler API. Install the validation application on
both sides; the patch alone does not contain those hook implementations.

The original `runner_name` string argument now names a runtime JSON file written
by the smoke launcher. It is not interpreted as the old QuantumRuntime Block.
The original sampler-options Variable interface remains in use. The launcher
creates a unique Variable for the explicit 16-shot options and records its name;
it does not overwrite existing Variables or register new blocks.

There are two non-computational adjustments in the patch: the original recovery
objects are bound explicitly with `load_recovery().SolveFn` and
`load_recovery().run_configuration_recovery`, and an import-order lint suppression
keeps the original source layout intact. The source proof checks those exact
bindings and the tests verify function identity. The unused sampler-local RNG line is
removed because transpilation moves into the allocation; the transpiler hook
preserves seed 24. The recovery RNG in the original flow is unchanged.

## What stays the same

`proof` verifies exact source segments and AST/decorators for eleven definitions:
the five parameter models, `_resolve_shots`, `make_block_solver`, the original
`sqd_2405_05068_fulqrum` flow, LUCJ circuit construction, the dedicated transpiler
and `deploy`. The original meas/test postprocessing inside `sample_bitstrings`
is also compared verbatim.

The full original flow body still computes chemistry, samples a `BitArray`,
converts it to configurations/probabilities, binds the original Fulqrum solver,
calls the recovery loop, creates its recovery artifact and returns its minimum
energy as a float. It is not replaced by the earlier H2 orchestration flow.

Recovery/subsampling are the verified fixed-source function objects, not new
implementations. `FulqrumSolverJob.run` and the native solver adapter modules
remain unchanged. As before, only that local package's metadata is adjusted
from 0.2.5 dependency pins to 0.3.0 and labeled `0.3.0.dev0`. No released library
code is changed. The outer launcher checks scheduler artifacts after the original
flow returns, including rejecting nonzero exit status even if the original
solver wrapper returned complete outputs.

## Fast parameters and acceptance

The original parameter classes and defaults remain intact. All speed-oriented
choices live in `pr42_smoke_parameters.json`:

| Input | Value |
| --- | --- |
| Molecule | H2, 0.74 Angstrom, STO-3G, four qubits |
| Quantum source | QRMI, 16 shots, meas only |
| LUCJ layers | 1 |
| Optimization level | 1 |
| SABRE layout / swap / iterations | 1 / 1 / 1 |
| Subspace / batches / recovery iterations | 1 / 1 / 1 |
| Seed | 24 |

This minimizes computation; scheduler and service overhead still apply.
The success criterion is **workflow completion**, not an accurate ground-state
energy. No FCI calculation, FCI tolerance or four-dimensional-subspace condition
is used. A one-dimensional result can differ substantially from full CI and
still pass this integration check. Results must remain finite and agree with
the original flow's own recovery records; scheduler jobs must complete with
exit status zero, and the native output/occupancies must be well formed.

The existing Slurm executor labels all stderr text as ERROR, even when the
native message itself says INFO. This display alone does not indicate failure;
the launcher checks the scheduler's state/exit status and actual output.

## Run on the existing cluster setup

Use the same control environment and successful QRMI/native-Fulqrum setup as
the previous end-to-end validation. No rebuild or block registration is needed.
The solver block must be dedicated to this validation and not used by another
concurrent flow: new native job directories are identified relative to its
pre-run directory snapshot, and their count is checked against this flow's
scheduler artifacts.

After this change is pushed, update the integration checkout, keeping any local
changes intact:

```bash
cd /shared/qrmi-pr42-recovery.YUxWez/qcsc-prefect
git status --short
git pull --ff-only origin qrmi-pr42-recovery-v0.3.0
export UV_PROJECT_ENVIRONMENT=/shared/qrmi-pr42-recovery.YUxWez/control-venv
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

First prove the source relationship without submitting a job or using Prefect API:

```bash
uv run --no-sync python -m examples.qrmi_sqd_validation.pr42_smoke proof
```

Then, when ready to submit **new quantum and classical jobs**, run the preserved
PR flow with the fast inputs:

```bash
uv run --no-sync python -m examples.qrmi_sqd_validation.pr42_smoke run \
  --solver-block qrmi-h2-recovery-49c599aef168-solver \
  --work-dir /shared/qrmi-pr42-original-smoke
```

`--parameter-file /absolute/path/to/inputs.json` can select another file with
the same `parameters` / `sampler_options` envelope. The default smoke inputs
are sufficient for demonstrating the Runtime connection. The launcher calls the
preserved flow with caching disabled and flow retries zero. Original task
decorators and result persistence remain visible in the copied source.

The run directory retains both source versions, their patch and hashes, actual
inputs, Runtime config, options-Variable name, raw quantum evidence, ordered
bitstrings and `workflow_summary.json`. The summary contains:

```json
{
  "validation": "workflow-passed",
  "accuracy_checked": false,
  "flow_name": "sqd_2405_05068_fulqrum"
}
```

It also records this flow's scheduler artifacts, new native solver output paths,
native version and the original recovery records. It reads the original
executor's Prefect artifacts; Slurm accounting storage is not required.
The uniquely named sampler-options Variable is retained for reproducibility.
On failure it writes `failure.json` and retains evidence. Rerunning `run`
creates new quantum sampling; `proof` only checks source preservation.

## Evidence scope

The earlier H2-specific entrypoint already has user-reported hardware PASS
(quantum job 24, classical job 25, 100 requested / 5 retained shots). Keep that
result as separate evidence. It must not be relabeled as an execution of this
new preserved-flow copy.

The new path is locally tested with the actual original Prefect flow/tasks,
real H2 chemistry, LUCJ/QPY/pinned compilation, configuration recovery and the
original Fulqrum block/CLI. Quantum readouts, scheduler execution and the native
numerical core are test doubles. A reference numerical solve deliberately
produces a non-FCI one-dimensional energy and still passes. Another test writes
complete native outputs with a failed scheduler result and verifies rejection.
A subsequent user-operated cluster run is needed before claiming that this
minimal-patch version also ran through QRMI and native Fulqrum on the cluster.

Local verification on 2026-10-26: **389 passed, 3 skipped, 2 xfailed** for the
full suite via `uv run --no-sync pytest`; the seven smoke checks passed. Ruff
checks/formatting and the dependency consistency check also passed.
