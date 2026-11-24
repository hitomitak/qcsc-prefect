# Run a One-GPU Job on ROQUO

ROQUO runs Slurm. qcsc-prefect sends jobs to it through the shared Slurm executor: the HPC
profile uses `hpc_target="slurm"`, and the ROQUO settings live in saved Blocks. There is no
ROQUO-specific executor, and the workflows need no ROQUO-specific code.

This page runs one small job on one GPU,
[examples/roquo_prefect_gpu_hello_demo](https://github.com/qiskit-community/qcsc-prefect/tree/main/examples/roquo_prefect_gpu_hello_demo).
It checks four things: the job is submitted, its state is followed to the end, its exit status
is reported, and its output is collected.

## Where things run

| What | Where | Needs |
| --- | --- | --- |
| The Prefect flow, which submits and monitors the job | a ROQUO login node | `sbatch`, `scontrol` and `sacct` on `PATH`; the Prefect API |
| The job script | a ROQUO compute node with a GPU | the program it runs, on a file system the node sees |
| `work_dir`: the job script and `output.out`/`output.err` | a file system shared by the two | write access from both |

The flow runs `sbatch --parsable` and reads the job state with `scontrol show job`, falling
back to `sacct` after the job leaves the queue. Slurm writes the job's output to `work_dir`,
and the flow reads it from there, so `work_dir` must be on a shared file system, for example
your home directory.

## How the ROQUO settings map to the Blocks

| ROQUO setting | Block field | Job script |
| --- | --- | --- |
| GPU partition | `HPCProfileBlock.queue_gpu` | `#SBATCH --partition=…` |
| Account | `HPCProfileBlock.project_gpu` (empty: your default account) | `#SBATCH --account=…` |
| Reservation | `HPCProfileBlock.slurm_extra_directives` | `#SBATCH --reservation=…` |
| GPUs per node | `ExecutionProfileBlock.gpus_per_node` with `resource_class="gpu"` | `#SBATCH --gpus-per-node=…` |
| Other directives, for example `--gres=gpu:1` | `HPCProfileBlock.slurm_extra_directives` | `#SBATCH …` |
| Modules | `ExecutionProfileBlock.modules` | `module load …` |
| Launcher | `ExecutionProfileBlock.launcher` (`srun` or `single`) | `srun "<program>"` |

`resource_class="gpu"` selects `queue_gpu` and `project_gpu`.

## ROQUO as checked on 2026-11-24

| Item | Value |
| --- | --- |
| Login node | x86_64; `sbatch`, `scontrol`, `sacct`, `scancel` in `/usr/bin`; Slurm 26.05.6 |
| Partitions | `roquo` (default; 4 GB200 GPUs per node), `roquo_share` (GB200 MIG `1g.23gb` slices), `work` (no GPU) |
| GPU request | `--gpus-per-node=1` and `--gres=gpu:1` are both accepted |
| Size of a one-GPU job | ROQUO adjusts it to a quarter node: 1 GPU, 36 cores, 400000 MB, with the warning `[ROQUO-W003]` from `sbatch`. The warning does not stop the submission |
| Reservation | none was needed |
| Modules | `cuda/12.8`, `cuda/13.2`, `cuda/13.3`, `nvhpc/25.5`, `nvhpc/26.5`, `hpcx/2.50`, `nvhpc-openmpi/25.5`, `nvhpc-openmpi/26.5` |

On that day the example ran with the Blocks of `roquo_gpu.example.toml` and an account, from
a login node with Prefect Cloud:

- a job exiting 0 gave `exit_status` 0, state `COMPLETED`, `gres/gpu=1` in `sacct`, and an
  `output.out` with the GB200 that `CUDA_VISIBLE_DEVICES=0` named;
- a job exiting 3 gave `exit_status` 3 and state `FAILED`, and the flow logged its
  `output.err` as an error.

Not checked yet: cancellation, `roquo_share`, more than one GPU, modules loaded in the job,
and the `--gres` form in a real job.

## Steps

Run these on a ROQUO login node, from a checkout of qcsc-prefect in your home directory.

### 1. Check the scheduler settings

```bash
sinfo -o "%P %a %G %D"                                   # partitions and their GPUs
sacctmgr -n show assoc user="$USER" format=account,partition,qos
scontrol show reservation | grep -E "ReservationName|Users|Accounts"
```

Before you submit through Prefect, test the request (`--test-only` submits nothing):

```bash
sbatch --test-only --partition=<partition> --account=<account> \
  --gpus-per-node=1 --time=00:05:00 --wrap=true
```

Add `--reservation=<name>` if your project runs in a reservation. On ROQUO, `sbatch` may
print `[ROQUO-W003]` and adjust the size: a one-GPU job gets a quarter node.

### 2. Install and connect to Prefect

```bash
uv sync --all-packages
uv run --no-sync prefect config view | grep PREFECT_API_URL   # the server or Cloud you use
```

`--all-packages` installs the workspace packages, Prefect among them. A plain `uv sync`
installs only the development tools, and removes packages that are not in the lock file.

### 3. Create the Blocks

```bash
cp examples/roquo_prefect_gpu_hello_demo/roquo_gpu.example.toml roquo_gpu.toml
# Edit roquo_gpu.toml, or give the account and reservation on the command line:
uv run --no-sync python examples/roquo_prefect_gpu_hello_demo/create_blocks.py \
  --config roquo_gpu.toml --account <account> [--reservation <name>]
```

This saves `cmd-roquo-gpu-hello`, `exec-roquo-gpu-1` and `hpc-roquo-gpu`, replacing any
Blocks of those names. To use other names, set them in the file's `[blocks]` table.

If ROQUO wants the GPU as a generic resource, delete the `gpus_per_node` line and set
`extra_directives = ["--gres=gpu:1"]`.

### 4. Run the job

```bash
uv run --no-sync python -c "import asyncio; from examples.roquo_prefect_gpu_hello_demo.flow import roquo_gpu_hello_flow; print(asyncio.run(roquo_gpu_hello_flow()))"
```

The flow returns the job record:

```text
{'job_id': '…', 'exit_status': 0, 'state': 'COMPLETED', 'node_list': '…', 'stdout_file': '…/work/roquo_gpu_hello/output.out', 'stderr_file': '…/output.err'}
```

`output.out` holds the node name, `SLURM_JOB_ID`, `CUDA_VISIBLE_DEVICES`, and the GPU from
`nvidia-smi`. The flow also logs it, and saves the job's state, exit code and node in the
table artifact `roquo-gpu-hello-demo-metrics`.

To check that a failing job is reported, run it with `exit_code=3`:

```bash
uv run --no-sync python -c "import asyncio; from examples.roquo_prefect_gpu_hello_demo.flow import roquo_gpu_hello_flow; print(asyncio.run(roquo_gpu_hello_flow(exit_code=3)))"
```

`exit_status` is then 3 and `state` is `FAILED`. The flow itself still completes: it
returns the job record, and the caller checks `exit_status`. The script also exits with 2
when `nvidia-smi` cannot list a GPU, for example when the job got none.

## Troubleshooting

- **`sbatch: error: Batch job submission failed: Invalid account or account/partition
  combination specified`**: check the account and partition with `sacctmgr` (step 1).
- **The job waits with `Reason=ReqNodeNotAvail` or `Reservation`**: the partition may need a
  reservation; set `reservation`.
- **`QCSC Prefect preflight failed: executable … does not exist`**: the program is not on a
  file system the compute node sees. Set `executable` in the config to a shared path.
- **Exit status 2 and `No GPU could be listed`**: the job got no GPU, or `nvidia-smi` is not
  on `PATH` on the compute node. Check the GPU request, or add the module that provides it.
