# GB-SQD Task Splitting Design (Current Implementation)

This document describes the implementation currently in this repository (`algorithms/gb_sqd`).

## 1. Purpose

The workflow was split from a monolithic `gb-demo` execution into task-level orchestration in Prefect so that:
- each major step is visible in the Prefect UI,
- failures are isolated to `init` / specific recovery iteration / `finalize`,
- state hand-off is explicit via `state_iter_*.json` files.

## 2. Implemented Flow Topology

Task-based flows (`ext_sqd_flow`, `trim_sqd_flow`) execute:

1. `initialize_task`
2. `recovery_iteration_task` repeated `num_recovery` times (sequential)
3. `final_diagonalization_task`
4. `output_results_task`

```mermaid
graph LR
    A["gb-demo init"] --> B["init/state_iter_000.json"]
    B --> C["gb-demo recovery (iter 0, --num-iters 1)"]
    C --> D["recovery_0/state_iter_001.json"]
    D --> E["gb-demo recovery (iter 1, --num-iters 1)"]
    E --> F["recovery_1/state_iter_002.json"]
    F --> G["..."]
    G --> H["gb-demo finalize"]
    H --> I["energy_log.json"]
```

Notes:
- Recovery tasks are intentionally sequential because each iteration consumes the previous state.
- Prefect does orchestration and observability; MPI parallelism remains inside `gb-demo`.

## 3. `run_job_from_blocks()` Contract (Actual)

The current function requires all of the following parameters:

```python
await run_job_from_blocks(
    command_block_name=...,                # required
    execution_profile_block_name=...,      # required
    hpc_profile_block_name=...,            # required
    work_dir=Path(...),                    # required
    script_filename="recovery_0.pbs",      # required
    user_args=[...],
)
```

Any call that omits required parameters fails with `TypeError`.

## 4. Command / Block Mapping

`create_blocks.py` creates command blocks and execution profiles with this default mapping:

| Purpose | CommandBlock default | CommandBlock args | ExecutionProfileBlock default |
|---|---|---|---|
| Simple ExtSQD flow | `cmd-gb-sqd-ext` | `--mode ext_sqd` | `exec-gb-sqd-ext-<target>` |
| Simple TrimSQD flow | `cmd-gb-sqd-trim` | `--mode trim_sqd` | `exec-gb-sqd-trim-<target>` |
| Task-based init | `cmd-gb-sqd-init` | `init` | `exec-gb-sqd-init-<target>` |
| Task-based recovery | `cmd-gb-sqd-recovery` | `recovery` | `exec-gb-sqd-recovery-<target>` |
| Task-based finalize | `cmd-gb-sqd-finalize` | `finalize` | `exec-gb-sqd-finalize-<target>` |

### Why separate `ExecutionProfileBlock` per command?

`run_job_from_blocks` validates:

- `ExecutionProfileBlock.command_name == CommandBlock.command_name`

If they differ, it raises `ValueError` and does not submit the job. Because command names differ (`gb-sqd-init`, `gb-sqd-recovery`, `gb-sqd-finalize`), separate execution profile blocks are required for task-based flows.

## 5. Task I/O Contract

### `initialize_task`
- Runs: `gb-demo init`
- Writes: `<work_dir>/init/state_iter_000.json`
- Returns: `init_data["state_file"]` as `str`

### `recovery_iteration_task`
- Runs: `gb-demo recovery`
- Uses input state from previous iteration (or init state)
- Always passes `--num-iters 1`
- Writes: `<work_dir>/recovery_<i>/state_iter_<i+1>.json`
- Returns: `state_file` as `str` (not `Path`)

### TrimSQD special handling
For trim mode, non-final global recovery iterations add:
- `--trim_no_final_carryover_type3`

This keeps the final carryover behavior only on the last iteration.

### `final_diagonalization_task`
- Runs: `gb-demo finalize --state-in <last_state> --output_dir <work_dir>`
- Expects and reads: `<work_dir>/energy_log.json`

### `output_results_task`
- Uses generated `energy_log.json` (or fallback generation path)
- Creates Prefect artifact `gb-sqd-energy-summary`
- Writes summary output metadata

## 6. Filesystem Assumptions (`work_dir`)

`work_dir` is resolved to an absolute path and used by both:
- Prefect worker process (Python tasks read `state_iter_*.json`, `energy_log.json`)
- HPC job scripts (submitted via `run_job_from_blocks`)

Therefore, `work_dir` must be on a filesystem shared and consistently mounted for worker and HPC execution nodes.

## 7. Failure and Restart Semantics (Current)

Current behavior:
- Task retries are enabled (`initialize`: 2, `recovery`: 1, `finalize`: 1).
- Recovery failure can be diagnosed at iteration granularity in Prefect.

Not implemented yet:
- automatic checkpoint-based resume that skips already-completed iterations in a new run.
- cache-policy-based replay control (e.g., `Inputs(...) + RUN_ID`) for GB-SQD tasks.

## 8. Status Against Prior Review Points

1. `run_job_from_blocks()` signature mismatch: addressed in code paths that call it.
2. Subcommand ↔ CommandBlock mapping ambiguity: addressed by explicit init/recovery/finalize blocks.
3. `output_dir`/`work_dir` path model in HPC: still a design constraint; now documented.
4. Local read of `energy_log.json`: still requires shared filesystem; now documented.
5. Cache policy for restart: not implemented yet (known gap).
6. `Path` serialization risk: mitigated by returning state paths as `str` in task outputs.

## 9. References

- Flow definitions: `algorithms/gb_sqd/gb_sqd/main.py`
- Tasks:
  - `algorithms/gb_sqd/gb_sqd/tasks/initialize.py`
  - `algorithms/gb_sqd/gb_sqd/tasks/recovery_iteration.py`
  - `algorithms/gb_sqd/gb_sqd/tasks/final_diagonalization.py`
  - `algorithms/gb_sqd/gb_sqd/tasks/output_results.py`
- Block creation: `algorithms/gb_sqd/create_blocks.py`
- Executor API: `packages/qcsc-prefect-executor/src/qcsc_prefect_executor/from_blocks.py`
