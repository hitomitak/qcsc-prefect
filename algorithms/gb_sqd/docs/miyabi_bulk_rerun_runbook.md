# GB-SQD Miyabi Bulk Rerun Runbook

This document explains how to build `gb-demo` on Miyabi and run only
`bulk_gb_sqd_flow_with_failed_target_rerun_plan` with Prefect.

It covers both:

- Miyabi CPU (`miyabi-c`, `regular-c`)
- Miyabi GPU (`miyabi-g`, `regular-g`)

The flow scans a parent directory, finds leaf directories that contain both
`count_dict.txt` and `fci_dump.txt`, runs one GB-SQD job per discovered target,
and reruns only failed targets with staged override parameters.

## 1. Preconditions

- You can log in to Miyabi-C.
- For GPU builds, you can enter an interactive Miyabi-G compute node.
- Prefect is already configured to talk to your self-hosted server or Cloud workspace.
- The `input_root_dir` and `output_root_dir` are on a filesystem visible from both:
  - the Prefect worker host
  - Miyabi compute nodes
- This repository is already available on Miyabi.

Relevant implementation files:

- `algorithms/gb_sqd/gb_sqd/bulk.py`
- `algorithms/gb_sqd/gb_sqd/bulk_rerun.py`
- `algorithms/gb_sqd/gb_sqd/tasks/bulk_target_run.py`
- `algorithms/gb_sqd/create_blocks.py`

## 2. Prepare the input tree

The flow expects a parent directory like:

```text
data/
  ligand/
    19_26MO_Lig/
      atom_1/
        count_dict.txt
        fci_dump.txt
      atom_2/
        count_dict.txt
        fci_dump.txt
    27_35MO_Lig/
      atom_0/
        count_dict.txt
        fci_dump.txt
```

Each deepest directory that contains both files becomes one GB-SQD target.

## 3. Build `gb-demo`

### 3.1 Miyabi CPU build

Log in to Miyabi-C and load the required modules:

```bash
ssh -A z12345@miyabi-c.example.org
cd /work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd/gb_demo_2026

module load intel
module load impi
mkdir -p build
cd build
cmake .. -DMIYABI=ON -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc
cmake --build .
```

Expected executable:

```text
/work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd/gb_demo_2026/build/gb-demo
```

This matches the upstream Miyabi CPU build using `mpiicc` / `mpiicpc`.

### 3.2 Miyabi GPU build

First enter a Miyabi-G interactive node. Example:

```bash
ssh -A z12345@miyabi-c.example.org
qsub -I -W group_list=gz00 -q interact-g -l select=1 -l walltime=1:00:00
```

Then build on the GPU node:

```bash
cd /work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd/gb_demo_2026
mkdir -p build-miyabi-gpu
cd build-miyabi-gpu
cmake .. -DMIYABI_GPU=ON -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpic++
cmake --build .
```

Expected executable:

```text
/work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd/gb_demo_2026/build-miyabi-gpu/gb-demo
```

This matches the upstream Miyabi GPU build using `mpicc` / `mpic++`.

## 4. Create Prefect blocks

Run block creation from the `algorithms/gb_sqd` directory.

### 4.1 Miyabi CPU blocks

```bash
cd /work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd

cat > gb_sqd_blocks_miyabi_cpu.toml <<'EOF'
hpc_target = "miyabi"
resource_class = "cpu"
project = "gz00"
queue = "regular-c"
work_dir = "/work/gz00/z12345/gb_sqd_runs"

num_nodes = 1
mpiprocs = 1
walltime = "01:00:00"
launcher = "mpiexec.hydra"
modules = ["intel", "impi"]
mpi_options = []

executable = "/work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd/gb_demo_2026/build/gb-demo"
EOF

uv run python create_blocks.py --config gb_sqd_blocks_miyabi_cpu.toml
```

Expected default block names:

- `cmd-gb-sqd-ext`
- `cmd-gb-sqd-trim`
- `exec-gb-sqd-ext-miyabi`
- `exec-gb-sqd-trim-miyabi`
- `hpc-miyabi-gb-sqd`

### 4.2 Miyabi GPU blocks

```bash
cd /work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd

cat > gb_sqd_blocks_miyabi_gpu.toml <<'EOF'
hpc_target = "miyabi"
resource_class = "gpu"
project = "gz00"
queue = "regular-g"
work_dir = "/work/gz00/z12345/gb_sqd_runs"

num_nodes = 1
mpiprocs = 1
walltime = "01:00:00"
launcher = "mpirun"
modules = []
mpi_options = ["-n", "1"]

executable = "/work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd/gb_demo_2026/build-miyabi-gpu/gb-demo"
EOF

uv run python create_blocks.py --config gb_sqd_blocks_miyabi_gpu.toml
```

Expected default block names:

- `cmd-gb-sqd-ext`
- `cmd-gb-sqd-trim`
- `exec-gb-sqd-ext-miyabi-gpu`
- `exec-gb-sqd-trim-miyabi-gpu`
- `hpc-miyabi-gpu-gb-sqd`

If you want `:ompthreads=...` in the generated PBS script, add `ompthreads = N`
explicitly. If omitted, Miyabi blocks now leave `ompthreads` unset.

## 5. Sanity-check the blocks

### 5.1 CPU

```bash
uv run prefect block inspect execution_profile/exec-gb-sqd-ext-miyabi
uv run prefect block inspect hpc_profile/hpc-miyabi-gb-sqd
```

Check:

- `resource_class = "cpu"`
- `queue_cpu = "regular-c"`
- `executable_map["gb_sqd"]` points to the CPU binary

### 5.2 GPU

```bash
uv run prefect block inspect execution_profile/exec-gb-sqd-trim-miyabi-gpu
uv run prefect block inspect hpc_profile/hpc-miyabi-gpu-gb-sqd
```

Check:

- `resource_class = "gpu"`
- `queue_gpu = "regular-g"`
- `executable_map["gb_sqd"]` points to the GPU binary

## 6. Run `bulk_gb_sqd_flow_with_failed_target_rerun_plan`

### 6.1 Miyabi CPU example

This example corresponds to the CPU-style Miyabi job script:

- queue: `regular-c`
- launcher: `mpiexec.hydra`
- binary: CPU build

```bash
cd /work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd

PYTHONPATH=/work/gz00/z12345/qcsc-prefect/packages/qcsc-prefect-core/src:/work/gz00/z12345/qcsc-prefect/packages/qcsc-prefect-adapters/src:/work/gz00/z12345/qcsc-prefect/packages/qcsc-prefect-blocks/src:/work/gz00/z12345/qcsc-prefect/packages/qcsc-prefect-executor/src:/work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd \
uv run --no-project python - <<'PY'
from gb_sqd import bulk_gb_sqd_flow_with_failed_target_rerun_plan

result = bulk_gb_sqd_flow_with_failed_target_rerun_plan(
    mode="ext_sqd",
    hpc_target="miyabi",
    resource_class="cpu",
    input_root_dir="./data/ligand",
    output_root_dir="/work/gz00/z12345/gb_sqd_runs/ligand_ext_miyabi_cpu",
    command_block_name="cmd-gb-sqd-ext",
    execution_profile_block_name="exec-gb-sqd-ext-miyabi",
    hpc_profile_block_name="hpc-miyabi-gb-sqd",
    max_jobs_in_queue=8,
    max_prefect_concurrency=8,
    num_recovery=2,
    num_batches=2,
    num_samples_per_batch=1000,
    iteration=1,
    adet_comm_size=1,
    bdet_comm_size=1,
    task_comm_size=1,
    adet_comm_size_final=1,
    bdet_comm_size_final=1,
    task_comm_size_final=1,
    do_carryover_in_recovery=True,
    carryover_ratio=0.50,
    carryover_threshold=5e-6,
    max_time=60,
    verbose=True,
    failed_target_override_sequence=[
        {"carryover_threshold": 1e-3},
        {"carryover_threshold": 1e-2, "max_time": 1800},
        {"carryover_threshold": 1e-1, "max_time": 2400},
    ],
)

print(result)
PY
```

### 6.2 Miyabi GPU example

This example corresponds to the GPU-style Miyabi job script:

- queue: `regular-g`
- launcher: `mpirun`
- binary: GPU build

```bash
cd /work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd

PYTHONPATH=/work/gz00/z12345/qcsc-prefect/packages/qcsc-prefect-core/src:/work/gz00/z12345/qcsc-prefect/packages/qcsc-prefect-adapters/src:/work/gz00/z12345/qcsc-prefect/packages/qcsc-prefect-blocks/src:/work/gz00/z12345/qcsc-prefect/packages/qcsc-prefect-executor/src:/work/gz00/z12345/qcsc-prefect/algorithms/gb_sqd \
uv run --no-project python - <<'PY'
from gb_sqd import bulk_gb_sqd_flow_with_failed_target_rerun_plan

result = bulk_gb_sqd_flow_with_failed_target_rerun_plan(
    mode="trim_sqd",
    hpc_target="miyabi",
    resource_class="gpu",
    input_root_dir="./data/ligand",
    output_root_dir="/work/gz00/z12345/gb_sqd_runs/ligand_trim_miyabi_gpu",
    command_block_name="cmd-gb-sqd-trim",
    execution_profile_block_name="exec-gb-sqd-trim-miyabi-gpu",
    hpc_profile_block_name="hpc-miyabi-gpu-gb-sqd",
    max_jobs_in_queue=8,
    max_prefect_concurrency=8,
    num_recovery=1,
    num_batches=1,
    num_samples_per_recovery=100,
    iteration=1,
    adet_comm_size=1,
    bdet_comm_size=1,
    task_comm_size=1,
    adet_comm_size_combined=1,
    bdet_comm_size_combined=1,
    task_comm_size_combined=1,
    adet_comm_size_final=1,
    bdet_comm_size_final=1,
    task_comm_size_final=1,
    carryover_ratio_batch=0.10,
    carryover_ratio_combined=0.50,
    carryover_threshold=5e-6,
    max_time=60,
    verbose=True,
    failed_target_override_sequence=[
        {"carryover_threshold": 1e-3},
        {"carryover_threshold": 1e-2, "max_time": 1800},
        {"carryover_threshold": 1e-1, "max_time": 2400},
    ],
)

print(result)
PY
```

## 7. What the generated job scripts should look like

The block settings above should lead to scripts that are conceptually close to:

### 7.1 Miyabi CPU

```bash
#!/bin/bash
#PBS -q regular-c
#PBS -l select=1
#PBS -W group_list=gz00

module load intel
module load impi

export KMP_AFFINITY=granularity=fine,compact,1,0

cd <attempt_dir>
mpiexec.hydra ./path/to/cpu/gb-demo ...
```

### 7.2 Miyabi GPU

```bash
#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -W group_list=gz00

export KMP_AFFINITY=granularity=fine,compact,1,0

cd <attempt_dir>
mpirun -n 1 ./path/to/gpu/gb-demo ...
```

`create_blocks.py` fills in the rest of the arguments from the flow parameters.

## 8. Output layout

For each discovered target directory, the flow creates:

```text
<output_root_dir>/
  _bulk_summary/
    run_summary.json
  <relative_target_path>/
    target_status.json
    attempt_001/
      gb_sqd_ext.pbs or gb_sqd_trim.pbs
      output.out
      output.err
      energy_log.json
    attempt_002/
      ...
```

Important files:

- `_bulk_summary/run_summary.json`
  - flow-level summary across all targets
- `<relative_target_path>/target_status.json`
  - latest status for that target
- `<relative_target_path>/attempt_NNN/output.out`
  - scheduler stdout
- `<relative_target_path>/attempt_NNN/output.err`
  - scheduler stderr

## 9. Troubleshooting

### 9.1 CPU build fails on Miyabi-C

Check:

- `module load intel`
- `module load impi`
- `mpiicc` and `mpiicpc` are available in `PATH`

### 9.2 GPU build fails on Miyabi-G

Check:

- you are on a Miyabi-G compute node
- `mpicc` and `mpic++` are available
- the build directory is separate from the CPU build directory

### 9.3 Prefect submits to the wrong queue

Inspect the block:

```bash
uv run prefect block inspect hpc_profile/hpc-miyabi-gb-sqd
uv run prefect block inspect hpc_profile/hpc-miyabi-gpu-gb-sqd
```

Verify:

- CPU block uses `queue_cpu = regular-c`
- GPU block uses `queue_gpu = regular-g`

### 9.4 Successful targets rerun unexpectedly

Make sure the rerun helper is called with the same:

- `input_root_dir`
- `output_root_dir`

and leaves `skip_completed=True` through the helper's internal reruns.
