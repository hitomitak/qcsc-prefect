# Per-Job Execution Profile Block for Bulk Jobs — Implementation Spec (Codex Handoff)

> **Repository:** `qcsc-prefect` (this repo). Self-contained; no other repo needed.
> Keep this file updated with as-built notes and record any deviation here.

## 1. Goal

Let a single `GlobalFugakuBulkRunner` submit different jobs with **different execution
profile blocks** (and optionally different HPC profile blocks), chosen **per job**.

Today the runner applies one `execution_profile_block` / `hpc_profile_block` to
**all** jobs (runner-level). The downstream consumer (gb_demo TrimSQD S/M/L sizing)
needs each job to pick a size-specific execution profile — mainly **node count**
(`ExecutionProfile.num_nodes`), walltime, and MPI layout — without spinning up
separate runners per size.

The layers below the executor already support this: `ExecutionProfile` is a per-call
dataclass and `qcsc_prefect_adapters/fugaku/builder.py::render_script(exec_profile=…)`
takes it per job. Only the executor's `BulkJobSpec` / runner currently hard-wires it
at runner level.

## 2. Pre-change state (verified anchors, 2026-06-26)

- `packages/qcsc-prefect-executor/src/qcsc_prefect_executor/bulk/models.py`
  - `BulkJobSpec` (L64): `job_key / work_dir / command_args / wave_id / target_id /
    stage_id / priority / expected_outputs / max_submit_attempts` — **no block field.**
  - `BulkJobRecord`: persisted job state — **no block field.**
- `.../bulk/global_fugaku_runner.py`
  - `GlobalFugakuBulkRunner` (L108): `command_block` / `execution_profile_block` /
    `hpc_profile_block` are **runner-level** (L117-119).
  - `_submit_once._attempt` (L213-216): `submit_job_from_blocks(...,
    execution_profile_block=self.execution_profile_block,
    hpc_profile_block=self.hpc_profile_block, ...)` — same blocks for every job.
  - `_monitor_once` (L179-193): `monitor_jobs_many(hpc_profile_block=
    self.hpc_profile_block, scheduler_job_ids=[...])` — one hpc block per batch.
- `.../bulk/registry.py`: `bulk_jobs` table with an **`ALTER TABLE … ADD COLUMN`
  migration mechanism** (`existing_columns` map, ~L145-158); `command_args` persisted
  as `command_args_json`; `_row_to_record` (~L762) maps rows → `BulkJobRecord`.

## 3. Decisions

1. Add **optional per-job** `execution_profile_block: str | None = None` and
   `hpc_profile_block: str | None = None` to `BulkJobSpec` (and `BulkJobRecord`).
   `None` ⇒ fall back to the runner/API-level block (fully backward compatible).
   Empty strings are invalid rather than silently falling back.
2. `command_block` stays **runner-level only** (the consumer uses the same command
   for all sizes). Do not add a per-job command block.
3. Persist the per-job blocks in the registry using the **existing `ADD COLUMN`
   migration + `command_args_json` pattern** — do not rewrite the schema. Once a
   scheduler id has been assigned, do not overwrite the persisted block metadata
   on later `upsert_jobs()` calls; monitoring must keep using the block that
   submitted the job.
4. Monitoring must remain correct when per-job `hpc_profile_block` differs. Group
   monitor calls by effective hpc block (`job.hpc_profile_block` if set, otherwise
   the runner/API default). If all jobs leave `hpc_profile_block=None`, monitoring
   is unchanged.
5. The new fields are honored by `GlobalFugakuBulkRunner` and by
   `run_jobs_from_blocks_bulk(..., submit_mode="single")`. `submit_mode="native_bulk"`
   rejects per-job block overrides because one generated script/profile is shared
   by all PJM subjobs in the group.
6. Queue probing remains runner/API-level. If per-job `hpc_profile_block` values
   point at different queues/projects, callers should pass an explicit conservative
   `queue_probe`; the default Fugaku probe is still resolved from the runner/API
   default blocks.

## 4. Instructions
- **Reuse existing patterns; do not duplicate.** New column → mirror
  `command_args_json` (INSERT/UPDATE + `_row_to_record` + `existing_columns`
  migration). Submit path → reuse `submit_job_from_blocks` unchanged, only choose
  which block string to pass.
- **Docs:** update this spec (as-built) and the executor's `README.md` /
  `docs/` where the runner/blocks are described (note the new optional per-job
  fields and that `None` = runner/API default).
- **Backward compatible:** all new fields optional/default `None`; existing callers
  and existing registry DBs keep working (migration adds the column to old DBs).
- **Tests:** unit tests only, mock `submit_job_from_blocks` / `monitor_jobs_many`.
  Add a **registry migration test** (open a DB created before the change, confirm the
  new columns are added and values round-trip).

## 5. Tasks

> As-built note: A1 and A2 should ship together. Allowing per-job
> `hpc_profile_block` at submit time without grouped monitoring leaves a broken
> intermediate state.

### Task A1 — Per-job block fields + registry + submit
- `bulk/models.py`: add `execution_profile_block: str | None = None` and
  `hpc_profile_block: str | None = None` to `BulkJobSpec` **and** `BulkJobRecord`.
- `bulk/registry.py`: add two nullable columns via the `existing_columns` migration
  map; include them in INSERT/UPDATE and in `_row_to_record`. Preserve existing
  persisted block values after `scheduler_job_id` or `scheduler_subjob_id` is set.
- `bulk/global_fugaku_runner.py::_submit_once._attempt` (L213-216): pass
  `job.execution_profile_block if job.execution_profile_block is not None else
  self.execution_profile_block` and the equivalent hpc fallback.
- `from_blocks.py::run_jobs_from_blocks_bulk` in `submit_mode="single"` should use
  the same effective per-job block selection.
- **Acceptance:**
  - A `BulkJobSpec` with a per-job `execution_profile_block` submits with that block;
    `None` ⇒ runner/API default (assert the block passed to a mocked
    `submit_job_from_blocks`).
  - Registry migration test: a pre-existing `bulk_jobs` DB gains the columns; the
    per-job block values round-trip through register → `get_*` → `BulkJobRecord`.
  - Existing tests plus the new per-job block tests pass.

### Task A2 — Monitoring groups by effective HPC block
- `_monitor_once`: instead of one `monitor_jobs_many(hpc_profile_block=
  self.hpc_profile_block, …)` for all jobs, **group monitorable jobs by their
  effective hpc block** (`job.hpc_profile_block` if set, otherwise
  `self.hpc_profile_block`) and call `monitor_jobs_many` **once per group** with
  that group's ids.
- `run_jobs_from_blocks_bulk(..., submit_mode="single")` uses the same monitor
  grouping.
- **Acceptance:** with jobs carrying two different hpc blocks, one
  `monitor_jobs_many` call is made per block with the correct ids (mocked); when all
  jobs share the runner block, exactly one call as before.

### Task A3 — Native bulk guard
- Reject per-job `execution_profile_block` / `hpc_profile_block` overrides when
  `submit_mode="native_bulk"` is selected.
- **Acceptance:** native bulk with an override raises `ValueError` before writing
  unsupported job metadata to the registry.

## 6. Interface contract (for the downstream consumer, gb_demo)
After this lands, callers may set `BulkJobSpec.execution_profile_block` (and
optionally `hpc_profile_block`) per job when using `GlobalFugakuBulkRunner` or
`run_jobs_from_blocks_bulk(..., submit_mode="single")`. `None` preserves current
behaviour. This is the only new surface gb_demo depends on.

## 7. Out of scope
- Per-job `command_block`.
- Per-job block overrides inside native PJM bulk submission mode.
