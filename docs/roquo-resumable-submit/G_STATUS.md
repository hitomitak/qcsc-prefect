# ROQUO resumable submit: G1–G9 status

Allowed feature states are `未着手`, `実装済(CI)`, `実機検証済`, and `hold`.

PR0 is a design gate and does not implement a G item. The owner approved shared-filesystem
SQLite with a transactional compare-and-set and no sidecar claim. Every feature remains
`未着手` until its implementation PR is complete.

| Gap | Status | Current baseline at `bb97c7f` | Planned PR | Real-machine gate |
|---|---|---|---|---|
| G1 sbatch↔record window | 未着手 | Job ID is stored only after scheduler submit; no PREPARED claim/CAS. | PR1 foundation, completed behavior in PR4 | Test 1 |
| G2 deterministic Slurm identity | 未着手 | Slurm request/template has no deterministic job-name/comment contract. | PR3 | Test 1 |
| G3 search and attach by identity | 未着手 | Monitoring requires known job IDs; no squeue/sacct identity search. | PR4 | Test 1 |
| G4 immutable spec hash | 未着手 | Registry has no canonical `spec_hash` guard. | PR1 fields, PR2 behavior | Test 1 indirectly |
| G5 operator hold | 未着手 | UNKNOWN remains monitorable and can poll indefinitely. | PR1 state, PR4 behavior | Test 1 |
| G6 explicit cancel intent | 未着手 | No durable desired state; Slurm wait cancellation currently calls scancel automatically. | PR1 fields, PR5 behavior | Tests 2 and 3 |
| G7 Slurm bounded probe/runner | 未着手 | Generic bulk runner accepts a probe, but no standard Slurm capacity probe is supplied. | PR7 | Test 5 |
| G8 Cloud log policy | 未着手 | Truncation exists in one path; no shared policy across bulk/reconcile. | PR8 | None required |
| G9 IBM Runtime durable attach | 未着手 | Job-reference fetch exists; no durable PREPARED/tag-search/operator-hold contract. | PR6 | Test 4 |

## Gate history

| Date | Gate | Result | Evidence |
|---|---|---|---|
| 2026-07-22 | PR0 atomicity decision | approved | Option A: trust HPC shared filesystem/SQLite locking; no sidecar or external mutex |
