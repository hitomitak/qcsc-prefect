# ROQUO resumable submit: implementation log

This log is updated in the same pull request as each implementation increment. “Real
machine verified” is used only after a human records an actual scheduler/runtime result in
`REAL_MACHINE_RUNBOOK.md`.

## Progress

| PR | G items | State | Changed files | What and why | Compatibility | Automated checks | Real machine | Open items |
|---|---|---|---|---|---|---|---|---|
| PR0 | Design gate for G1/G4/G5/G6 | approved | `docs/roquo-resumable-submit/{ATOMICITY_DECISION,IMPLEMENTATION_LOG,G_STATUS,REAL_MACHINE_RUNBOOK}.md` | Recorded storage/locking choices and the approved shared-SQLite-only direction before changing scheduler behavior. | Documentation only; no public API, schema, submit, monitor, or cancel behavior changed. | `uv run --with mkdocs-material --with 'mkdocstrings[python]' mkdocs build --strict` passed. No code tests required. | Atomicity preflight waived as a gate. Tests 1–5 remain future human gates. | No sidecar/external mutex. Revisit only if the trusted filesystem assumption fails. |
| PR1 | Foundation for G1/G4/G5/G6 | implemented (CI) | `bulk/{__init__,models,registry}.py`, `tests/test_bulk_registry.py`, PR tracking docs | Added durable schema/model vocabulary required by later immutable-spec, submit-or-attach, hold, and cancel-intent PRs. | Existing constructors keep defaults; old databases migrate in place; submit, monitor, and cancel candidate selection is unchanged. | Registry tests, executor tests, Ruff, and strict docs build; see PR1 checks below. | Not required. | No production transition enters `PREPARED` or `AWAITING_OPERATOR` until PR4; cancel intent behavior waits for PR5. |

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
