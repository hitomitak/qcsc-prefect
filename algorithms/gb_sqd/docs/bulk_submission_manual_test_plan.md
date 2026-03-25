# GB-SQD Bulk Mode Manual Test Plan

This document defines the recommended manual test scenarios for running the
bulk GB-SQD flow on a real Fugaku environment.

Use it together with:

- `docs/bulk_submission_flow_runbook.md`
- `docs/bulk_submission_flow_design.md`

## 1. Purpose

The automated tests cover local parameter wiring, retry bookkeeping, queue
parsing, and per-target overrides.

The manual tests below focus on the parts that need a real Fugaku environment:

- actual `pjsub` / `pjstat` interaction
- real queue throttling behavior
- real PJM script generation and runtime environment
- end-to-end `gb-demo` execution on compute nodes

## 2. Common Preconditions

- Prefect blocks already created with `create_blocks.py`
- `gb-demo` already built for Fugaku
- input tree placed on a shared filesystem visible from both:
  - the Prefect execution host
  - Fugaku compute nodes
- output root placed on the same shared filesystem

Recommended baseline parameters for the first runs:

- `max_jobs_in_queue=2`
- `max_prefect_concurrency=2`
- `queue_limit_scope="user_queue"`
- `max_target_task_retries=1`

## 2.1 Manual Runner

The repository now includes a helper CLI that prepares the manual-test
workspace and runs individual scenarios:

- One-shot command sequence script: `manual_bulk_test_commands.sh`

```bash
cd /Users/hitomi/Project/qcsc-prefect/algorithms/gb_sqd

# 1. Create the workspace from one known-good target directory
python manual_bulk_test.py prepare \
  --seed-dir ./data/ligand/13_18MO_Wat/atom_10012 \
  --workspace-root /shared/gb_sqd_manual_bulk

# 2. Inspect one scenario
python manual_bulk_test.py describe \
  --workspace-root /shared/gb_sqd_manual_bulk \
  --scenario scenario1_ext_happy

# 3. Run one scenario
python manual_bulk_test.py run \
  --workspace-root /shared/gb_sqd_manual_bulk \
  --scenario scenario1_ext_happy
```

Useful follow-up commands:

```bash
# List scenarios
python manual_bulk_test.py list

# Repair the intentionally corrupted retry target before scenario 4
python manual_bulk_test.py repair \
  --seed-dir ./data/ligand/13_18MO_Wat/atom_10012 \
  --workspace-root /shared/gb_sqd_manual_bulk \
  --scenario scenario3_retry_runtime \
  --relative-target-path retry_bad_fcidump/atom_0003

# Print the exact bulk flow kwargs without running
python manual_bulk_test.py run \
  --workspace-root /shared/gb_sqd_manual_bulk \
  --scenario scenario4_override_rerun \
  --dry-run
```

## 3. Test Data Layout

Prepare a small dedicated test tree separate from production input:

```text
data/manual_bulk_test/
  success_a/
    atom_0001/
      count_dict.txt
      fci_dump.txt
  success_b/
    atom_0002/
      count_dict.txt
      fci_dump.txt
  retry_bad_fcidump/
    atom_0003/
      count_dict.txt
      fci_dump.txt
  override_target/
    atom_0004/
      count_dict.txt
      fci_dump.txt
```

Use valid files for all directories except when a scenario explicitly asks you
to break one file.

## 4. Required Checks For Every Scenario

After each run, inspect:

- `<output_root_dir>/_bulk_summary/run_summary.json`
- each `<relative_target_path>/target_status.json`
- each `attempt_NNN/` directory
- the generated PJM script
- the scheduler output/error files

Minimum fields to verify:

- `status`
- `latest_attempt`
- `latest_output_dir`
- `latest_job_id`
- `parameter_overrides`
- `energy_log.json` existence for successful targets

## 5. Manual Scenarios

### Scenario 1: ExtSQD happy path with multiple targets

Purpose:
- verify recursive discovery
- verify one target directory becomes one Fugaku job
- verify successful summary aggregation

Setup:
- use two valid target directories
- `mode="ext_sqd"`

Expected result:
- flow succeeds
- `succeeded_targets == 2`
- each target has `attempt_001`
- each target has `energy_log.json`
- each target status is `success`

### Scenario 2: TrimSQD happy path

Purpose:
- verify TrimSQD argument wiring and PJM generation

Setup:
- use one valid target directory
- `mode="trim_sqd"`
- pass the full TrimSQD parameters including combined/final communicator sizes

Expected result:
- flow succeeds
- generated PJM script calls `gb-demo --mode trim_sqd`
- command line contains the expected TrimSQD-only options

### Scenario 3: Retry on runtime failure

Purpose:
- verify failed target retry and `attempt_NNN` creation

Setup:
- keep `fci_dump.txt` present but corrupt its contents for one target
- keep one or more other targets valid
- `max_target_task_retries=1`

Expected result:
- corrupted target runs `attempt_001` and `attempt_002`
- valid targets still succeed
- corrupted target ends as `failed`
- flow ends in failure because at least one target failed
- failed target `target_status.json` contains both attempts

Important note:
- do not remove `fci_dump.txt`
- a missing file is an input validation error, not the intended retry scenario

### Scenario 4: Failed-target-only rerun with `target_overrides`

Purpose:
- verify parameter override support on rerun

Setup:
- start from the failed state produced by Scenario 3
- fix the corrupted `fci_dump.txt`
- rerun with the same `input_root_dir` and `output_root_dir`
- keep `skip_completed=True`
- pass:

```python
target_overrides={
    "retry_bad_fcidump/atom_0003": {
        "max_time": 600,
        "num_samples_per_batch": 500,
    }
}
```

Expected result:
- already successful targets are skipped
- previously failed target gets a new `attempt_NNN`
- `target_status.json` records `latest_parameter_overrides`
- final run succeeds if the fixed input is valid

### Scenario 5: `skip_completed=True` rerun without overrides

Purpose:
- verify completed targets are not resubmitted accidentally

Setup:
- rerun a fully successful bulk output root
- keep `skip_completed=True`

Expected result:
- no new attempt directory is created for successful targets
- task result for each successful target is `skipped`
- `run_summary.json` shows skipped targets

### Scenario 6: Mixed success and failure in one flow

Purpose:
- verify the flow continues across targets and fails only at the end

Setup:
- one valid target
- one intentionally broken target
- `fail_fast=False`

Expected result:
- valid target finishes successfully
- broken target ends in failure after retries
- flow raises at the end
- summary contains both success and failure entries

### Scenario 7: Queue throttling with `queue_limit_scope="user_queue"`

Purpose:
- verify the queue gate counts all active jobs for the same user and queue

Setup:
- before starting the bulk flow, place unrelated jobs for the same user in the
  same Fugaku resource group
- set `max_jobs_in_queue` lower than current active jobs

Expected result:
- the bulk flow waits before submitting
- once external jobs leave the queue, the bulk flow starts submitting
- Prefect logs show delayed queue-slot acquisition

### Scenario 8: Queue throttling with `queue_limit_scope="flow_jobs_only"`

Purpose:
- verify the queue gate ignores unrelated jobs and only counts this flow's jobs

Setup:
- keep unrelated jobs for the same user in the same queue
- run the bulk flow with:
  - `queue_limit_scope="flow_jobs_only"`
  - a distinctive `job_name_prefix`

Expected result:
- unrelated jobs do not block the bulk flow
- only jobs whose PJM job name starts with `job_name_prefix` are counted

### Scenario 9: Invalid override validation

Purpose:
- verify bad `target_overrides` are rejected before submission

Setup:
- pass one of the following:
  - an unknown relative path
  - an unsupported parameter name

Expected result:
- flow fails immediately with `ValueError`
- no jobs are submitted

## 6. Recommended Execution Order

Run the scenarios in this order:

1. Scenario 1
2. Scenario 2
3. Scenario 3
4. Scenario 4
5. Scenario 5
6. Scenario 6
7. Scenario 7
8. Scenario 8
9. Scenario 9

This order keeps the early checks small and functional before moving to queue
and override edge cases.

## 7. Automated Coverage Already Available

Local automated tests currently cover:

- discovery rules
- artifact key normalization
- queue listing parsing
- per-target override validation
- per-target override merging
- bulk task retry bookkeeping
- `skip_completed` behavior
- input validation failure behavior

Run them with:

```bash
cd /Users/hitomi/Project/qcsc-prefect/algorithms/gb_sqd
uv run --extra dev pytest \
  tests/test_target_overrides.py \
  tests/test_bulk_target_run.py \
  tests/test_bulk_target_run_task.py \
  tests/test_discovery.py \
  tests/test_fugaku_queue.py
```
