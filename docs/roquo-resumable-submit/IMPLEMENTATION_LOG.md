# ROQUO resumable submit: implementation log

This log is updated in the same pull request as each implementation increment. “Real
machine verified” is used only after a human records an actual scheduler/runtime result in
`REAL_MACHINE_RUNBOOK.md`.

## Progress

| PR | G items | State | Changed files | What and why | Compatibility | Automated checks | Real machine | Open items |
|---|---|---|---|---|---|---|---|---|
| PR0 | Design gate for G1/G4/G5/G6 | approved | `docs/roquo-resumable-submit/{ATOMICITY_DECISION,IMPLEMENTATION_LOG,G_STATUS,REAL_MACHINE_RUNBOOK}.md` | Recorded storage/locking choices and the approved shared-SQLite-only direction before changing scheduler behavior. | Documentation only; no public API, schema, submit, monitor, or cancel behavior changed. | `uv run --with mkdocs-material --with 'mkdocstrings[python]' mkdocs build --strict` passed. No code tests required. | Atomicity preflight waived as a gate. Tests 1–5 remain future human gates. | No sidecar/external mutex. Revisit only if the trusted filesystem assumption fails. |
| PR1 | Foundation for G1/G4/G5/G6 | implemented (CI) | `bulk/{__init__,models,registry}.py`, `tests/test_bulk_registry.py`, PR tracking docs | Added durable schema/model vocabulary required by later immutable-spec, submit-or-attach, hold, and cancel-intent PRs. | Existing constructors keep defaults; old databases migrate in place; submit, monitor, and cancel candidate selection is unchanged. | Registry tests, executor tests, Ruff, and strict docs build; see PR1 checks below. | Not required. | No production transition enters `PREPARED` or `AWAITING_OPERATOR` until PR4; cancel intent behavior waits for PR5. |
| PR2 | G4 | implemented (CI) | `bulk/{spec_hash,exceptions,models,registry}.py`, `from_blocks.py`, focused tests, PR tracking docs | Added versioned canonical resolved-spec hashing and rejected reuse of a `job_key` when its stored hash differs. | Optional caller digests default to `NULL`; legacy rows remain readable; stored hashed rows become immutable before scheduler side effects. | Canonicalization/guard tests, executor tests, Ruff, format check, and strict docs build; see PR2 checks below. | Not required. | Legacy scheduler rows whose hash is `NULL` cannot be cryptographically verified; existing non-submit status guards still prevent automatic resubmission. |
| PR3 | G2 | implemented (CI) | Slurm builder/template, adapter tests, `SLURM_IDENTITY.md`, PR tracking docs | Added deterministic, non-secret Slurm job-name/comment generation and safe optional template directives. | Existing callers that do not provide identity values render the same script. No scheduler search, registry claim, or automatic recovery behavior changes before PR4. | Adapter rendering/identity tests, Ruff, format check, and strict docs build; see PR3 checks below. | Test 1 after PR4 must verify the target cluster preserves name/comment in `squeue` and `sacct`. | Target-specific name/comment limits and accounting visibility remain operational facts to record before enabling recovery. |

## PR0 review notes

- Verified target: `qcsc-prefect` commit `bb97c7f`, branch `slurm`, clean target
  working tree at review start.
- Current SQLite registry has a five-second busy timeout but no external access mutex and no
  compare-and-set submit claim.
- Current submit order remains unchanged in PR0: candidate check → prepare → scheduler
  submit → `mark_submitted`.
- SQLite official guidance makes a shared-NFS, multi-process deployment a design decision,
  not a safe default. The repository owner explicitly accepted that platform assumption
  and selected option A without a sidecar claim or external mutex.
- No ROQUO job, Prefect Cloud action, IBM Runtime request, commit, push, or pull request was
  performed as part of PR0.

## PR1 registry/model contract

PR1 adds persistence vocabulary only. It does not enable a partially recoverable submit
path: no production method transitions a job to `PREPARED` or `AWAITING_OPERATOR`, and the
existing submit, monitor, and cancel behavior remains unchanged.

### Status classification

| Status | Terminal | Active | Submit candidate | Recovery candidate | Meaning in the completed design |
|---|---:|---:|---:|---:|---|
| `PENDING` | no | no | yes | no | Eligible for the existing submit path. |
| `SUBMIT_DEFERRED` | no | no | yes | no | Retryable submit was deferred before scheduler acceptance. |
| `PREPARED` | no | no | no | yes | A durable claim exists but no scheduler job ID is recorded. The process may have stopped either before `sbatch` or after a successful `sbatch`. PR4 will reconcile before any resubmit. |
| `SUBMITTED`, `QUEUED`, `RUNNING` | no | yes | no | no | A scheduler job ID is known and the job is active. |
| `SUCCEEDED`, `FAILED`, `CANCELLED` | yes | no | no | no | Existing terminal states. |
| `UNKNOWN` | no | no | no | yes | Existing uncertain scheduler state. It remains monitorable; marking it recovery-capable records that reconciliation may resolve it. |
| `AWAITING_OPERATOR` | no | no | no | no | Durable hold excluded from automatic processing. Only an explicit operator action may move it in the completed PR4 behavior. |

`PREPARED` is deliberately not an active state because scheduler acceptance has not yet
been associated with a recorded job ID. `AWAITING_OPERATOR` is deliberately neither
terminal nor recoverable: it preserves ambiguity without allowing an automatic submit or
an unbounded polling loop.

### Added fields

| Field | PR1 default | Contract |
|---|---|---|
| `spec_hash` | `NULL` | Versioned canonical resolved-spec fingerprint. PR2 will compute and enforce it. |
| `prepared_at` | `NULL` | UTC claim time used to bound scheduler history lookup. PR4 will write it atomically with `PREPARED`. |
| `job_name`, `job_comment` | `NULL` | Deterministic, non-secret scheduler search identity. PR3 will generate it. |
| `desired_state` | `RUN` | Operator-requested target state. The other modeled value is `CANCEL_REQUESTED`; PR5 will add its behavior. |
| `cancel_requested_at`, `cancel_requested_by`, `cancel_reason` | `NULL` | Audit fields for explicit cancellation intent. PR5 will write and act on them. |

`BulkJobSpec` accepts the three submit-spec values (`spec_hash`, `job_name`, and
`job_comment`) as optional fields. `BulkJobRecord` exposes all persisted values. Existing
callers that do not pass the optional fields therefore retain their previous behavior.

### Schema and row compatibility

- Registry initialization adds missing columns with `ALTER TABLE`; existing rows receive
  `NULL` for optional fields and `RUN` for `desired_state`.
- A newly registered row has the same defaults. Its status is still selected only by the
  existing output-completion check (`PENDING` or `SUCCEEDED`).
- Re-registering an unsubmitted row may fill or update non-`NULL` spec identity fields.
  Passing the legacy defaults does not erase values already persisted by a newer caller.
- Unknown stored lifecycle statuses still decode as `UNKNOWN`. An invalid or absent
  desired-state value decodes conservatively as `RUN` for compatibility; schema-managed
  rows persist a non-`NULL` enum value.
- The SQLite table has no status check constraint, so adding enum values requires no row
  rewrite. Old and new rows are covered by registry round-trip tests.

### PR1 automated checks

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages/qcsc-prefect-core/src:packages/qcsc-prefect-adapters/src:packages/qcsc-prefect-blocks/src:packages/qcsc-prefect-executor/src uv run --with pytest pytest packages/qcsc-prefect-executor/tests/test_bulk_registry.py -q`
  — 30 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages/qcsc-prefect-core/src:packages/qcsc-prefect-adapters/src:packages/qcsc-prefect-blocks/src:packages/qcsc-prefect-executor/src uv run --with pytest --with jinja2 --with prefect --with pydantic pytest packages/qcsc-prefect-executor/tests -q`
  — 116 passed, 2 explicitly opt-in HPC integration tests skipped.
- `uv run --with ruff ruff check packages/qcsc-prefect-executor/src/qcsc_prefect_executor/bulk packages/qcsc-prefect-executor/tests/test_bulk_registry.py`
  — passed.
- `uv run --with ruff ruff format --check packages/qcsc-prefect-executor/src/qcsc_prefect_executor/bulk/__init__.py packages/qcsc-prefect-executor/src/qcsc_prefect_executor/bulk/models.py packages/qcsc-prefect-executor/src/qcsc_prefect_executor/bulk/registry.py packages/qcsc-prefect-executor/tests/test_bulk_registry.py`
  — 4 files already formatted.
- `PYTHONPATH=packages/qcsc-prefect-core/src:packages/qcsc-prefect-adapters/src:packages/qcsc-prefect-blocks/src:packages/qcsc-prefect-executor/src:packages/qcsc-prefect-dice/src:packages/qcsc-prefect-qiskit/src uv run --with mkdocs-material --with 'mkdocstrings[python]' mkdocs build --strict`
  — passed. The existing informational notice that the ROQUO tracking pages and
  `tutorials/tmp.md` are outside `nav` remains.

An earlier executor-suite attempt with only `pytest` as an injected dependency stopped
during collection because `jinja2` was absent; no tests ran in that attempt. The successful
suite command above injects the dependencies needed by the repository's adapter imports.

## PR2 canonical spec hash and immutable guard

PR2 uses schema version `qcsc-prefect-bulk-spec-v1`. A stored value has the form
`qcsc-prefect-bulk-spec-v1:sha256:<hex>`. Canonical JSON sorts mappings, normalizes a
`Path` like its POSIX string, treats tuples and lists as the same ordered sequence, and
rejects non-finite floats or unsupported nondeterministic values.

Every dynamic scalar and string is represented inside the canonical payload by a typed
SHA-256 fingerprint. Command arguments, executable paths, environment values, caller
digests, and other dynamic strings therefore do not occur as plaintext in the payload,
stored hash, mismatch exception, or implementation log. `SpecHashMismatchError` reports
only the `job_key`, stored hash, incoming hash, and the instruction to use a new key.

### Hash field contract

| Included | Excluded |
|---|---|
| Resolved executable and ordered command arguments | `job_key` and workflow/campaign naming rules |
| Resolved execution profile: nodes, MPI/OMP counts, launcher/options, walltime, modules, pre-commands, environment | Registry status and scheduler job/subjob IDs |
| Resolved scheduler target, queue/partition/resource group, account/project, and target-specific CPU/GPU/QPU/memory/resource request | `prepared_at`, submitted/started/finished timestamps, attempts, and errors |
| Optional caller `input_digest`, `code_digest`, and `environment_digest` | Work/attempt directory and generated script filename |
| Command key and scheduler-visible executable mapping | Derived scheduler `job_name` and `job_comment` to avoid a PR3 identity/hash cycle |

The library does not infer ROQUO campaign keys. When a resource, input, code, environment,
or command change is intentional, the caller must retain the old immutable row and issue a
new `job_key`.

### Enforcement and compatibility

- `submit_job_from_blocks` resolves blocks and computes the hash before registry mutation
  and before calling the scheduler. A deferred retry with the same hash is allowed.
- A stored non-`NULL` hash is checked inside the same registry transaction before any row
  update. A mismatch rolls back the whole `upsert_jobs` transaction and preserves the
  original command arguments, digests, status, and hash.
- Bulk resume resolves incoming specs before updating rows that already have hashes.
  Native Fugaku bulk candidates are all resolved and registered with hashes before the
  native bulk scheduler submission.
- Named `command_args` are converted to CLI options in sorted-key order so mapping
  insertion order cannot change either the command or hash.
- `input_digest`, `code_digest`, and `environment_digest` are optional model/registry
  fields. Old databases migrate them to `NULL`, and existing callers need not supply them.
- A legacy submit candidate with `spec_hash=NULL` is backfilled with its resolved hash
  before its next scheduler side effect. A legacy active or terminal row without a hash is
  not automatically resubmitted or treated as a verified match.

### PR2 automated checks

The commands below were run from an exported Git-index snapshot so unrelated
timeout/subprocess work already present in the working tree was not part of the tested PR2
commit:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages/qcsc-prefect-core/src:packages/qcsc-prefect-adapters/src:packages/qcsc-prefect-blocks/src:packages/qcsc-prefect-executor/src uv run --with pytest --with jinja2 --with prefect --with pydantic pytest packages/qcsc-prefect-executor/tests -q`
  — 123 passed, 2 explicitly opt-in HPC integration tests skipped.
- `uv run --with ruff ruff check packages/qcsc-prefect-executor/src/qcsc_prefect_executor/bulk packages/qcsc-prefect-executor/src/qcsc_prefect_executor/from_blocks.py packages/qcsc-prefect-executor/tests/test_bulk_registry.py packages/qcsc-prefect-executor/tests/test_bulk_spec_hash.py`
  — passed.
- `uv run --with ruff ruff format --check packages/qcsc-prefect-executor/src/qcsc_prefect_executor/bulk/__init__.py packages/qcsc-prefect-executor/src/qcsc_prefect_executor/bulk/exceptions.py packages/qcsc-prefect-executor/src/qcsc_prefect_executor/bulk/models.py packages/qcsc-prefect-executor/src/qcsc_prefect_executor/bulk/registry.py packages/qcsc-prefect-executor/src/qcsc_prefect_executor/bulk/spec_hash.py packages/qcsc-prefect-executor/tests/test_bulk_registry.py packages/qcsc-prefect-executor/tests/test_bulk_spec_hash.py`
  — 7 files already formatted. `from_blocks.py` is covered by Ruff check; its full-file
  format check has two pre-existing formatting deltas outside PR2.
- `PYTHONPATH=packages/qcsc-prefect-core/src:packages/qcsc-prefect-adapters/src:packages/qcsc-prefect-blocks/src:packages/qcsc-prefect-executor/src:packages/qcsc-prefect-dice/src:packages/qcsc-prefect-qiskit/src uv run --with mkdocs-material --with 'mkdocstrings[python]' mkdocs build --strict`
  — passed. The existing informational notice that the ROQUO tracking pages and
  `tutorials/tmp.md` are outside `nav` remains.

## PR3 deterministic Slurm identity

PR3 defines `build_slurm_job_identity(job_key, spec_hash)` in the Slurm adapter.
It derives a readable, safe job name with a retained 96-bit digest suffix and a
full SHA-256 identity digest in the Slurm comment. The full comment is the
future recovery match key; the abbreviated job-name suffix is only a practical
human-visible discriminator.

The builder emits `#SBATCH --job-name` and `#SBATCH --comment` only when the
new optional `SlurmJobRequest` fields are set. Values are validated as safe
single-token directive text rather than shell-quoted, because Slurm processes
directives before the batch shell. No existing request gains an identity
implicitly, so ordinary script rendering remains backward compatible.

The conservative default maximum name length is 64 characters. It is a library
default, not an asserted Slurm standard; the target ROQUO cluster's effective
limit and `squeue`/`sacct` retention must be recorded during Test 1 after PR4.
See `SLURM_IDENTITY.md` for the complete contract.

### PR3 automated checks

- `PYTHONDONTWRITEBYTECODE=1 uv run --with pytest --with jinja2 pytest packages/qcsc-prefect-adapters/tests/test_slurm_builder.py -q`
  — 7 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages/qcsc-prefect-core/src:packages/qcsc-prefect-adapters/src:packages/qcsc-prefect-blocks/src:packages/qcsc-prefect-executor/src uv run --with pytest --with jinja2 --with prefect --with pydantic pytest packages/qcsc-prefect-executor/tests/test_run_slurm_job_local.py -q`
  — 1 passed.
- `uv run --with ruff ruff check packages/qcsc-prefect-adapters/src/qcsc_prefect_adapters/slurm/builder.py packages/qcsc-prefect-adapters/tests/test_slurm_builder.py`
  — passed.
- `uv run --with ruff ruff format --check packages/qcsc-prefect-adapters/src/qcsc_prefect_adapters/slurm/builder.py packages/qcsc-prefect-adapters/tests/test_slurm_builder.py`
  — 2 files already formatted.
- `PYTHONPATH=packages/qcsc-prefect-core/src:packages/qcsc-prefect-adapters/src:packages/qcsc-prefect-blocks/src:packages/qcsc-prefect-executor/src:packages/qcsc-prefect-dice/src:packages/qcsc-prefect-qiskit/src uv run --with mkdocs-material --with 'mkdocstrings[python]' mkdocs build --strict`
  — passed. The existing informational notice that the ROQUO tracking pages and
  `tutorials/tmp.md` are outside `nav` now also lists `SLURM_IDENTITY.md`.

These focused checks exercised only the PR3 identity/rendering paths. Unrelated
uncommitted scheduler-timeout work was present in the working tree and is not
part of this PR3 change.

## Future PR entry template

Copy this section for each PR and complete every field.

```text
PR:
G items:
State: implemented (CI) | real-machine verified | hold
Files:
What changed and why:
Backward compatibility:
Tests and exact commands:
Real-machine requirement/result:
Unresolved items:
gb_demo gap-checklist delta note:
```

## gb_demo gap-checklist delta note

PR0 does not change the G1–G9 implementation findings in
`gb_demo_2026/python/docs/QCSC_PREFECT_GAP_CHECKLIST.md`. After the owner approves the
atomicity mechanism, a future gb_demo documentation change should record the chosen
production registry path, filesystem assumptions, and claim primitive. The qcsc-prefect
PR must not modify the gb_demo document directly.
