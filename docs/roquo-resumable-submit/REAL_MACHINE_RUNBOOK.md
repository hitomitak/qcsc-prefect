# ROQUO resumable submit: real-machine runbook

All procedures in this document are run by a human operator. Unit tests must mock Slurm,
Prefect blocks, HPC access, and IBM Runtime. Do not mark a feature `実機検証済` until the
corresponding result record below is complete.

## Global safety rules

- Use a dedicated test directory, dedicated test registry, recognizable test job names,
  and the smallest permitted scheduler/runtime workload.
- Never point a fault-injection command at a production registry or an unrelated process.
- Record every scheduler/runtime job ID before inducing a failure.
- Cleanup only the explicitly recorded test job IDs. Never use a broad name/user match for
  cancellation.
- Do not store tokens, account secrets, environment dumps, count dictionaries, QPY files,
  or scientific outputs in Git.
- If the scheduler or filesystem result is ambiguous, stop. Do not reset a claim, resubmit,
  attach, or cancel until an operator has reconciled the evidence.

## PR0 preflight: registry and claim atomicity

### Purpose

Diagnose production filesystem behavior if the trusted shared-SQLite assumption is later
questioned. The repository owner waived this preflight as a PR1 gate. This test does not
submit a Slurm job.

### Inputs to record

```text
Test root (absolute, disposable, on the proposed production mount):
Registry production path (absolute):
Node/process A:
Node/process B:
Filesystem type/source:
Mount options:
Python version and sqlite3.sqlite_version:
Candidate primitive: mkdir | O_EXCL
Iterations:
```

### Procedure

1. On every potential writer host, inspect the proposed path with the site's supported
   equivalents of `findmnt -T <path>` and `stat -f <path>`. Confirm that all hosts resolve
   the same mount and record the source, filesystem type, and options.
2. Ask the filesystem/site owner whether concurrent SQLite locking is supported on this
   mount. Record the response or documentation; do not infer support only from “NFSv4”.
3. In a new disposable directory, launch two independent processes on different potential
   writer hosts. Synchronize each iteration and have both attempt the same atomic `mkdir`
   (or `open` with `O_CREAT|O_EXCL`) claim name.
4. For every iteration, write each process's success/failure to a separate pre-created log
   path. Verify exactly one success, one already-exists result, and immediate visibility of
   the created entry from both hosts. Use at least 1,000 iterations unless site policy
   specifies a stronger test.
5. Repeat with abrupt termination immediately after successful claim creation. Verify the
   claim remains visible and is not treated as expired. Remove test claims only after both
   processes have stopped and the recorded winner is known.
6. Against a separate disposable SQLite database on the proposed mount, run concurrent
   insert/update transactions from both hosts using the current rollback-journal mode.
   After all writers exit, run `PRAGMA integrity_check`, verify it returns `ok`, and verify
   the exact expected row/update counts. Record lock errors and maximum wait time.
7. Treat a SQLite stress-test pass as evidence only. Approve shared SQLite without a
   secondary lock only when the filesystem owner supports it and the actual writer
   topology satisfies option A's conditions.
8. If a test fails, stop automatic submission and reopen `ATOMICITY_DECISION.md` before
   continuing scheduler work.

### Expected diagnostic result

- The production mount and all writer hosts are identified.
- Each same-name claim race has exactly one winner and one loser.
- Created claims are visible from both hosts and survive creator termination.
- The disposable SQLite database remains internally consistent.
- SQLite remains internally consistent. A failure invalidates approved option A and
  requires a new storage/locking decision.

### PR0 result record

```text
Executed by:
Date/time/time zone:
Environment and mount:
Iterations:
Exactly-one-winner result:
Cross-node visibility result:
Creator-kill result:
SQLite integrity/count result:
Site support statement:
Observed anomalies:
Artifacts/log paths (not committed):
Decision approved:
Reviewer:
```

PR0 decision record: **not run; waived as an implementation gate by the repository owner
on 2026-07-22.**

## Test 1: Slurm submit-or-attach and operator hold (after PR4)

### Procedure

1. Submit one short, low-resource test job normally. Record deterministic job-name,
   comment, spec hash, job ID, registry transitions, user/account/partition, and output.
2. Enable the reviewed fault hook that terminates the submitter after `sbatch` accepts the
   job but before `mark_submitted`. Do not emulate this by killing an unscoped process.
3. Start recovery from a new process. Verify it finds exactly one active candidate via
   `squeue`, validates all identity fields, attaches, and stores the job ID without another
   `sbatch` call.
4. Repeat recovery after the job is terminal. Verify `sacct -S` finds the allocation row,
   ignores job-step/duplicate rows, attaches, and marks success only after expected output
   evidence is present.
5. Measure job-name/comment preservation, scheduler/accounting visibility lag, and clock
   skew. Record the configured grace and skew margins.
6. Exercise zero-candidate-after-grace and deliberately constructed multiple-candidate
   cases. Verify both stop in `AWAITING_OPERATOR`, the bulk loop exits, and another retry
   produces no scheduler side effect.
7. Exercise comment/spec mismatch. Verify attach is rejected and the existing registry
   record is not overwritten.
8. Launch two concurrent callers for the same key/spec. Verify exactly one scheduler
   submit occurs.

### Expected result

One logical job produces at most one automatic scheduler submission. Unique candidates are
attached only after full validation; zero, multiple, or mismatched candidates fail closed
without infinite polling.

### Result record

```text
Executed by/date/time zone:
qcsc-prefect commit:
Slurm version/cluster/partition:
Test job IDs:
Name/comment observed:
squeue visibility lag:
sacct visibility lag:
Clock skew:
Concurrent caller submit count:
Active attach result:
Terminal attach/output result:
Zero/multiple/mismatch hold results:
Cleanup confirmation:
Result: PASS | FAIL | BLOCKED
```

## Test 2: explicit cancel intent (after PR5)

### Procedure

1. Start a short job that remains active long enough to observe.
2. Cancel only the Prefect wait/process without recording cancel intent. Verify
   `CancelledError` propagates and no `scancel` is invoked; confirm the Slurm job continues.
3. Restart and attach to the same job.
4. Call the explicit `request_cancel(job_key, requested_by, reason)` API. Verify the
   durable intent and audit fields exist before the cancel executor runs.
5. Run the cancel executor twice. Verify `scancel` is effective at most once and terminal,
   not-found, and transient-failure responses are recorded distinctly.

### Expected result

No cancel intent means zero scheduler cancellations. A durable intent permits an
idempotent, audited cancel, and cancellation of the waiting coroutine is never swallowed.

### Result record

```text
Executed by/date/time zone:
qcsc-prefect commit:
Test job IDs:
No-intent scancel count/job state:
Attach-after-wait-cancel result:
Intent record:
Repeated executor result:
Response classifications:
Cleanup confirmation:
Result: PASS | FAIL | BLOCKED
```

## Test 3: allocation loss, API disconnect, and surviving process (after PR5)

### Procedure

1. For separate low-cost test jobs, simulate allocation walltime expiry, Prefect/API
   disconnection, and a replaced runner while the old submit/wait process remains alive.
2. Confirm that failure/crash/disconnection alone creates no cancel intent and calls no
   `scancel`.
3. Start the replacement run and verify it attaches to the known or uniquely rediscovered
   job. If evidence is ambiguous, verify it enters operator hold instead.
4. Restore connectivity and reconcile old/new observations. Verify there is neither a
   duplicate scheduler submission nor an implicit cancel.

### Expected result

Infrastructure and orchestration failures do not imply user cancellation. Recovery either
attaches safely or holds for an operator.

### Result record

```text
Executed by/date/time zone:
qcsc-prefect commit:
Scenario and job IDs:
Old process state:
Replacement attach/hold result:
scancel count:
Scheduler submit count:
Cleanup confirmation:
Result: PASS | FAIL | BLOCKED
```

## Test 4: IBM Runtime durable submit-or-attach (after PR6)

### Procedure

1. Use an approved test backend or the smallest permitted low-shot job. Pass backend,
   shots, credentials, and journal path through runtime configuration; do not hard-code or
   commit them.
2. Record the durable PREPARED journal and submit-time stable tag before submitting.
3. Trigger the reviewed fault hook after Runtime accepts the job but before its job ID is
   stored.
4. Restart recovery. Verify exactly one tag-matched candidate attaches and the journal is
   repaired without a second submission.
5. Exercise zero candidates after grace, multiple candidates, and spec mismatch. Verify
   all fail closed in operator hold and never select the newest job automatically.
6. Fetch the result by durable job reference after discarding transient Prefect task
   state, demonstrating that recovery does not depend on seven-day retention.

### Expected result

Submit/record failure is recovered from pre-submit tags and the shared journal without a
duplicate QPU job. Ambiguity enters operator hold.

### Result record

```text
Executed by/date/time zone:
qcsc-prefect commit:
Backend category (no credential):
Runtime job IDs:
Submission/tag count:
Attach result:
Zero/multiple/mismatch results:
Durable fetch result:
Cleanup/usage confirmation:
Result: PASS | FAIL | BLOCKED
```

## Test 5: Slurm bounded queue probe (after PR7)

### Procedure

1. Configure an explicit maximum active-job count and safety margin for a dedicated test
   user/account/partition scope.
2. Record the scoped active PENDING/RUNNING count from `squeue` and calculate expected
   available capacity.
3. Run one refill tick and verify it submits no more than that capacity or the configured
   per-refill cap, whichever is smaller.
4. Repeat with jobs outside each user/account/partition filter and verify they do not alter
   the scoped count.
5. Inject a probe command failure and malformed output. Verify capacity fails closed to
   zero and no scheduler submission occurs.

### Expected result

The runner obeys the configured ceiling and filters. It never guesses cluster quota from
`squeue`, and probe uncertainty results in zero submissions.

### Result record

```text
Executed by/date/time zone:
qcsc-prefect commit:
Configured user/account/partition:
Maximum/margin/refill cap:
Observed scoped active count:
Expected/actual submit count:
Filter results:
Probe-failure submit count:
Cleanup confirmation:
Result: PASS | FAIL | BLOCKED
```
