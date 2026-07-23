# ROQUO resumable submit: G1–G9 status

Allowed feature states are `未着手`, `実装済(CI)`, `実機検証済`, and `hold`.

PR0 is a design gate and does not implement a G item. The owner approved shared-filesystem
SQLite with a transactional compare-and-set and no sidecar claim. PR1 adds the schema and
model foundation for G1/G4/G5/G6 without enabling behavior. PR2 completes G4 in CI and PR3
completes the Slurm identity-generation/template contract in CI; G1/G3/G5/G6 remain
`未着手` until their behavior PRs are complete.

| Gap | Status | Current baseline at `bb97c7f` | Planned PR | Real-machine gate |
|---|---|---|---|---|
| G1 sbatch↔record window | 未着手 | `PREPARED` and `prepared_at` are modeled, but production still stores the job ID only after scheduler submit; no claim/CAS yet. | PR1 foundation complete; behavior in PR4 | Test 1 |
| G2 deterministic Slurm identity | 実装済(CI) | The Slurm request/template accepts optional safe `job_name`/`comment`; the adapter derives deterministic identities from a `job_key` and immutable `spec_hash`. PR4 will persist and use the full comment for recovery. | PR3 complete | Test 1 |
| G3 search and attach by identity | 未着手 | Monitoring requires known job IDs; no squeue/sacct identity search. | PR4 | Test 1 |
| G4 immutable spec hash | 実装済(CI) | Versioned resolved-spec hashes are computed before scheduler submission; a stored mismatch raises `SpecHashMismatchError` before row mutation or external side effects. | PR2 complete | Test 1 indirectly |
| G5 operator hold | 未着手 | `AWAITING_OPERATOR` is modeled and excluded from automatic classifications, but no production transition/operator API exists. | PR1 state complete; behavior in PR4 | Test 1 |
| G6 explicit cancel intent | 未着手 | Desired-state and audit fields are persisted with safe defaults, but Slurm wait cancellation still calls scancel automatically. | PR1 fields complete; behavior in PR5 | Tests 2 and 3 |
| G7 Slurm bounded probe/runner | 未着手 | Generic bulk runner accepts a probe, but no standard Slurm capacity probe is supplied. | PR7 | Test 5 |
| G8 Cloud log policy | 未着手 | Truncation exists in one path; no shared policy across bulk/reconcile. | PR8 | None required |
| G9 IBM Runtime durable attach | 未着手 | Job-reference fetch exists; no durable PREPARED/tag-search/operator-hold contract. | PR6 | Test 4 |

## Gate history

| Date | Gate | Result | Evidence |
|---|---|---|---|
| 2026-07-22 | PR0 atomicity decision | approved | Option A: trust HPC shared filesystem/SQLite locking; no sidecar or external mutex |
| 2026-07-22 | PR1 registry/model foundation | implemented (CI) | Migration, defaults, round-trip, and complete lifecycle classification tests |
| 2026-07-23 | PR2 canonical spec hash | 実装済(CI) | Canonicalization, resolved resource coverage, secret non-exposure, retry match, mismatch preservation |
| 2026-07-23 | PR3 deterministic Slurm identity | 実装済(CI) | Deterministic safe identity construction, suffix-preserving truncation, and Slurm template rendering |
