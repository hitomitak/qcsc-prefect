# Slurm submit-or-attach contract

PR4 enables resumable Slurm submission when `submit_job_from_blocks` is called with a
`BulkJobRegistry`. The safety goal is not strict exactly-once delivery across an external
scheduler. It is to prevent automatic duplicate submission and stop for an operator when
the scheduler side effect cannot be identified uniquely.

## State transitions

```text
PENDING / SUBMIT_DEFERRED
  └─ atomic compare-and-set ─> PREPARED
                                ├─ sbatch returns a numeric ID ─> SUBMITTED
                                ├─ unique validated candidate ─> attach ID ─> scheduler state
                                ├─ no candidate within grace ─> PREPARED
                                └─ zero after grace / multiple / mismatch
                                                           ─> AWAITING_OPERATOR
```

Only the process that wins the SQLite compare-and-set may write the Slurm script and call
`sbatch`. A losing process observes `PREPARED`, searches Slurm, and never submits. Saving
the same scheduler job ID is idempotent and cannot regress `QUEUED` or `RUNNING` back to
`SUBMITTED`.

`PREPARED` intentionally covers both “the process stopped before `sbatch`” and “Slurm
accepted the job but the process stopped before saving the ID.” Those cases cannot be
distinguished from the registry alone.

## Submit-result classification

| Observation | Registry action | Automatic retry |
|---|---|---:|
| `sbatch --parsable` returns one numeric allocation ID | Save ID and enter `SUBMITTED` | no |
| Slurm explicitly rejects the request | `FAILED`, or `SUBMIT_DEFERRED` for an explicitly classified retryable rejection | allowed only for the deferred case |
| Local timeout, controller connection loss, cancellation, empty output, unparseable zero-exit output, or unexpected client failure | Preserve `PREPARED` and search | never |
| Registry write fails after Slurm returns an ID | Preserve `PREPARED`; report unknown outcome | never |

Scheduler CLI calls have a bounded command timeout. Timing out or cancelling the local
`sbatch` process does not prove that the controller rejected the request.

## Identity search and validation

Recovery queries active allocations with
[`squeue`](https://slurm.schedmd.com/squeue.html) and accounting history with
[`sacct`](https://slurm.schedmd.com/sacct.html). The history lower bound is
`prepared_at - slurm_clock_skew_margin_seconds`. Search results request explicit fields
for allocation ID, name, comment, user, account, partition, submit time, and state.

Automatic attach requires exactly one allocation-level candidate and exact agreement on:

- deterministic job name;
- the complete spec-derived comment, not only the abbreviated name suffix;
- Slurm user;
- account and partition;
- a parseable submit time inside the recovery window.

Job-step rows such as `.batch`/`.extern`, array IDs, and heterogeneous IDs are excluded.
Rows for the same allocation returned by both commands are deduplicated; multiple distinct
allocations are never reduced by selecting the newest one.

For reliable automatic recovery, configure an explicit Slurm account and partition in the
HPC profile. If the request omits an account but the cluster reports a resolved default
account, the values do not agree and recovery fails closed for operator review.

## Grace, accounting, and terminal output

The default visibility grace is 120 seconds and the default clock-skew allowance is 60
seconds. They can be changed with `slurm_recovery_grace_seconds` and
`slurm_clock_skew_margin_seconds`. `scheduler_command_timeout_seconds` defaults to 300
seconds and may be set to `None` only when an unbounded scheduler command is explicitly
acceptable.

No candidate during the grace keeps the row `PREPARED` and raises `RecoveryPending` from
the single-submit API. The bulk API returns a result with a nonzero `prepared` count
instead of sleeping indefinitely. No candidate after the grace, multiple candidates, or
metadata mismatch enters durable `AWAITING_OPERATOR`.

A uniquely recovered terminal `COMPLETED` allocation first has its job ID attached. It is
marked `SUCCEEDED` only when every registered `expected_outputs` path exists. Missing
output evidence enters operator hold. Output discovery alone cannot bypass a `PREPARED`
claim or `AWAITING_OPERATOR` hold.

## Operator APIs

Inspect `record.last_error`, the immutable identity fields, and the scheduler evidence
before acting. The two explicit recovery operations are:

```python
from qcsc_prefect_executor.bulk import BulkJobRegistry

registry = BulkJobRegistry("/shared/path/bulk.sqlite")

# Use only after verifying this exact allocation belongs to the held job.
registry.operator_attach("logical-job-key", "123456")

# Use only after proving that no scheduler submission occurred.
registry.confirm_not_submitted_and_reset(
    "logical-job-key",
    confirmed_by="operator-identity",
    reason="checked active queue and retained accounting history",
)
```

`operator_attach` accepts only a numeric allocation ID and a row in `PREPARED` or
`AWAITING_OPERATOR`; job-step and array IDs are rejected. Reset additionally requires that
no scheduler ID is stored. A bulk retry that sees a hold returns immediately with
`awaiting_operator` and `operator_action_required_jobs`; it performs no scheduler
submission or identity search.

Target-specific retention, visibility delay, name/comment preservation, account defaults,
and clock skew remain real-cluster facts. Record them in Test 1 of
`REAL_MACHINE_RUNBOOK.md` before treating the feature as real-machine verified.
