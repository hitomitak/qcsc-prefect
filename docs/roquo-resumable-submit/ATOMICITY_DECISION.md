# ROQUO resumable submit: atomicity decision

## Status

**APPROVED: option A (shared-filesystem SQLite only).**

This document records the storage and locking choice for Slurm submit-or-attach. On
2026-07-22, the repository owner approved trusting the HPC shared filesystem and SQLite
locking as an operating condition. PR1 may proceed without a sidecar claim or an external
registry mutex.

## Scope and verified baseline

The PR0 review was performed against `qcsc-prefect` commit `bb97c7f` (v0.2.5). The target
working tree was clean at the start of the review.

The current implementation has the following relevant behavior:

- `BulkJobRegistry` stores state in SQLite, opens a new connection per operation, and sets
  `PRAGMA busy_timeout = 5000`. It does not select WAL mode and does not add an external
  writer lock.
- Registry methods use SQLite transactions, but there is no atomic compare-and-set that
  chooses one submitter.
- `submit_job_from_blocks` checks whether a row is a submit candidate, prepares the script,
  invokes the scheduler, and only then calls `mark_submitted`. Two callers can therefore
  both pass the read-only guard, and a process can also die after `sbatch` succeeds but
  before the scheduler job ID is stored.
- The ROQUO verification log says the previously tested shared filesystem was NFSv4 and
  visible from login and compute nodes. Its O_EXCL/rename behavior was adopted as an
  assumption, not tested. The production registry mount and the set of writer nodes are
  still unknown for this change.

SQLite's own documentation says to avoid a database on NFS when multiple processes may
access it concurrently, because filesystem locking may be unreliable. Its atomic-commit
documentation recommends a secondary locking mechanism if a network filesystem must be
used. `BEGIN IMMEDIATE` acquires a write transaction early, but it still depends on the
underlying filesystem lock implementation. WAL is not an alternative when clients run on
different hosts.

References:

- [SQLite FAQ: NFS and concurrent processes](https://sqlite.org/faq.html#q5)
- [SQLite atomic commit: broken locking implementations](https://sqlite.org/atomiccommit.html#sect_9_1)
- [SQLite transactions: `BEGIN IMMEDIATE`](https://sqlite.org/lang_transaction.html)
- [SQLite database format: WAL on a network filesystem](https://sqlite.org/fileformat.html#the_write_ahead_log)
- [POSIX directory operations are atomic and serializable](https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap04.html#tag_04_04)
- [NFSv4 CREATE behavior](https://www.rfc-editor.org/rfc/rfc8881.html)

These specifications are necessary context, but the production ROQUO mount behavior must
still be checked on the actual filesystem and mount options.

## Safety invariants

Whichever mechanism is approved must preserve all of the following:

1. At most one process may cross the final submission gate for one `(job_key, spec_hash)`.
2. The durable `PREPARED` record exists before the scheduler side effect.
3. A process that cannot prove it owns the submission claim must not call `sbatch`.
4. Loss, corruption, or ambiguity of a lock/claim fails closed. It must not trigger an
   automatic reset, claim steal, submit, attach, or cancel.
5. A submission claim is not a lease. It has no time-based automatic expiry and is not
   removed merely because the creating process disappeared.
6. A scheduler candidate is attached only after identity and spec checks. Zero or multiple
   candidates eventually enter the durable operator hold defined for PR4.
7. All workflow writers use `BulkJobRegistry` transactions and the same conditional-update
   protocol. Ad-hoc direct writes with `sqlite3` are unsupported.
8. WAL mode is not enabled for a database accessed from different hosts.

The design target is not strict exactly-once execution. It is to avoid automatic duplicate
submission and to stop safely for operator judgment whenever the scheduler result cannot
be proved.

## Options

### A. Shared-filesystem SQLite only (approved)

Use one SQLite file on the shared filesystem and implement the state transition with a
single conditional update inside `BEGIN IMMEDIATE`, for example
`PENDING|SUBMIT_DEFERRED -> PREPARED`.

Benefits:

- Smallest implementation and one source of durable state.
- SQLite provides transaction and migration support already used by the registry.

Accepted risks and operating conditions:

- `BEGIN IMMEDIATE` does not repair an unreliable NFS lock implementation.
- The owner explicitly accepts the HPC filesystem and SQLite locking behavior as a trusted
  platform property for this workflow. A separate filesystem claim is not required.
- Cross-node concurrency and `PRAGMA integrity_check` remain useful diagnostics if a lock
  or corruption problem is observed, but they are not a PR1 gate.
- The submission winner will be selected by one SQLite conditional update in a write
  transaction. Only a caller whose update changes one row may call the scheduler.

### B. Single-node local SQLite

Put the registry on local storage and allow only a single designated service/node to own
submission and registry writes.

Benefits:

- Avoids network-filesystem SQLite locking.
- Keeps the implementation close to the current registry.

Risks and acceptance conditions:

- Other login nodes and recovery processes cannot safely attach without routing every
  operation through the owner service.
- Node loss can also lose the registry unless local storage is durable and recovered or
  replicated by an explicitly designed mechanism.
- A shared recovery path is a core requirement, so this option is unsuitable unless the
  owner can guarantee a durable single-writer service and its recovery procedure.

Copying a live SQLite file between local and shared storage is not an acceptable
replication protocol.

### C. Shared SQLite plus filesystem sidecar coordination

Keep the durable registry on the shared filesystem, but add two filesystem mechanisms:

1. A short-lived, registry-wide access mutex serializes every SQLite operation across
   hosts. This avoids relying on byte-range locks for reader/writer or writer/writer
   coordination. Read concurrency can be reconsidered only with filesystem support and
   dedicated tests.
2. A durable per-job submission claim is the final gate before `sbatch`. Its identity is
   derived from stable digests, not raw or secret-bearing `job_key` content.

The initial primitive candidate is atomic `mkdir`: one contender creates a deterministic
claim directory and all others receive an already-exists result. `O_CREAT|O_EXCL` is the
fallback candidate if the site test or filesystem guidance favors NFS exclusive file
creation. The same primitive should not be accepted merely from protocol documentation;
it must pass the cross-node preflight on the production mount.

Benefits:

- The durable claim prevents two callers from submitting the same logical job even if
  SQLite locking alone is not trusted.
- The existing registry remains the queryable state store and migration mechanism.

Risks and acceptance conditions:

- The database and claim are separate stores and cannot be atomically committed together.
  Every crash point must therefore fail closed and be recoverable through deterministic
  scheduler search or explicit operator action.
- An abrupt process death can leave a registry-wide access mutex. Automatic stale-lock
  stealing is not allowed initially; the runbook must require verification of host,
  process, timestamp, and scheduler evidence before an operator removes it.
- A per-job claim remains as an audit record. It is never deleted or reused by normal
  retry logic. `confirm-not-submitted-and-reset` must have a separately reviewed,
  auditable generation/reset procedure.
- All registry access must participate in the global mutex. Protecting only the submission
  path would leave other reads and writes outside the filesystem coordination protocol.

## Approved implementation direction

**Use option A. Do not implement option C's sidecar claim or external registry mutex.**

The intended PR4 ordering is:

1. Resolve the job spec and calculate its canonical `spec_hash` and deterministic Slurm
   identity.
2. In one SQLite write transaction, conditionally transition `PENDING` or
   `SUBMIT_DEFERRED` to `PREPARED`, storing the identity and `prepared_at`. Only the caller
   that changes one row may continue.
3. Call `sbatch` and then store the scheduler job ID in SQLite.
4. If the result or registry update is ambiguous, preserve `PREPARED` and recover by
   `squeue`/`sacct`; do not automatically submit again.

This ordering uses the SQLite compare-and-set as the submit-winner gate and durable
lifecycle state. The transaction is committed before `sbatch` and is not held while the
scheduler command runs.

## Approved operational facts

| Item | Required value | Current evidence |
|---|---|---|
| Production registry absolute path | Caller-configured; must be on the trusted HPC shared filesystem | The library must not hard-code a site path |
| Filesystem type and mount source | ROQUO/HPC shared filesystem | Prior test area reported as NFSv4; exact deployment mount stays environment-specific |
| Relevant mount options | Environment-specific | Trusted as an operational platform property |
| Possible writer hosts | Multiple processes/hosts may access the registry | SQLite transaction selects the submit winner |
| Maximum concurrent writer processes | Configuration-dependent | No library hard-coded limit |
| Can a single durable local-disk owner be guaranteed? | Not required | Shared registry is retained |
| Site support statement for SQLite on this mount | Not required as a PR gate | Repository owner accepted the operating assumption |
| Approved claim primitive | SQLite conditional update only | No `mkdir`/`O_EXCL` sidecar |
| Preflight evidence | Not run; diagnostic runbook retained | Owner waived it as an implementation gate |

## Recorded decision

```text
Decision: A — shared-filesystem SQLite only
Registry path: caller-configured path on the trusted HPC shared filesystem
Filesystem/mount: ROQUO/HPC shared filesystem; environment-specific
Writer hosts/processes: may be multiple
Claim primitive: one SQLite conditional state transition in a write transaction
Sidecar or external mutex: none
Preflight result and date: not run; waived as a PR gate
Approved by: repository owner
Approval date: 2026-07-22
Conditions or follow-up: keep rollback-journal mode; do not enable WAL across hosts. If
locking anomalies or integrity failures occur, stop automatic submission and revisit
option C.
```
