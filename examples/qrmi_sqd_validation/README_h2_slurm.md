# Run H2 LUCJ through the QRMI/Slurm library

This guide uses the validated `test_heron` resource, the `debug` partition,
`/shared/qcsc-prefect` and `/shared/venv/qrmi`. It runs quantum sampling only.
For the algorithm-facing BitArray return value, use the
[local integration entrypoint](README_integration.md).

```text
Workflow host: H2/STO-3G CCSD -> pinned LUCJ
  -> ffSIM PRE_INIT and gate expansion -> QPY and JSON
  -> existing Slurm executor -> sbatch --qpu=test_heron
Compute node: QPY -> QRMI Target -> pinned specialized compiler
  -> task_runner -> IBM Quantum Batch -> result.json
Workflow host: PrimitiveResult -> validate 4-bit meas and shots -> summary.json
```

## 1. Copy the selected source files

From the local repository root, replace `VM1_SSH_HOST` with the usual SSH host.
The checkout includes uncommitted files; no GitHub operation is needed.

```bash
rsync -avR --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude='build/' --exclude='*.egg-info/' --exclude='.env' \
  packages/qcsc-prefect-qiskit \
  examples/qrmi_sqd_validation \
  examples/prefect_bitcount_demo/qrmi_slurm_runtime.py \
  examples/prefect_bitcount_demo/qrmi_task_runner.py \
  algorithms/qcsc_workflow_utility/src \
  packages/qcsc-prefect-executor/tests/test_qrmi_sqd_slurm_poc.py \
  hitomi@VM1_SSH_HOST:/shared/qcsc-prefect/
```

The existing core, blocks, adapters and executor packages must also be installed.
For this branch, deploy its `packages/qcsc-prefect-adapters` directory as well:
its monitoring fix lets running jobs be tracked without accounting storage.
Shared source and the shared venv must be visible at the same paths on every
eligible compute node.

## 2. Update and verify the installed code

On the workflow host, retain the previous successful Prefect Cloud configuration:

```bash
cd /shared/qcsc-prefect
source /shared/venv/qrmi/bin/activate
export QRMI_CHECK_DIR="$(mktemp -d /shared/qrmi-h2-library-check.XXXXXX)"
python -m pip freeze > "$QRMI_CHECK_DIR/packages-before.txt"
python -m pip install --no-deps \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-qiskit
python -m pip check
python -m pip freeze > "$QRMI_CHECK_DIR/packages-after.txt"
```

Use the already validated QRMI and chemistry dependencies. The validation used
Python 3.12, Qiskit 2.4.0, qiskit-ibm-runtime 0.46.1, ffSIM 0.0.84,
PySCF 2.13.0 and pydantic-numpy 9.1.0. `requirements.txt` records the separate
offline environment; do not use it to blindly upgrade a working QRMI installation.
If dependencies are missing, resolve them before submitting a quantum job.

On the workflow host **and every eligible compute node**, verify imports outside
the checkout so source paths cannot hide a missing installation:

```bash
(
  cd /tmp
  /shared/venv/qrmi/bin/python - <<'PY'
import sys
from pathlib import Path
from qrmi import QRMIService
from qcsc_prefect.integrations.qiskit.qrmi import runtime, worker
expected = Path('/shared/qcsc-prefect/packages/qcsc-prefect-qiskit/src').resolve()
print('Python:', sys.executable)
for module in (runtime, worker):
    path = Path(module.__file__).resolve()
    print(module.__name__, path)
    assert path.is_relative_to(expected), path
print('Installed library imports OK')
PY
)
/shared/venv/qrmi/bin/python \
  /shared/qcsc-prefect/examples/prefect_bitcount_demo/qrmi_task_runner.py --help
```

The worker help must include `--request` and `--task-runner`. Importing these
modules and requesting help do not acquire a QPU. A version number alone cannot
identify this unpublished revision; check the module paths and deployed source.

## 3. Verify Batch configuration and three quantum blocks

Retain the working SPANK configuration. Each node must configure the resource
with `QRMI_IBM_QCS_SESSION_MODE="batch"`. SPANK acquires the Batch before Python
starts and supplies the resource-prefixed acquisition token. Exporting the mode
inside a running Python process cannot change an already acquired allocation.
The worker checks the mode and token presence before invoking `task_runner`.

On the workflow host:

```bash
prefect config view
python - <<'PY'
from qcsc_prefect_blocks.common.blocks import (
    CommandBlock, ExecutionProfileBlock, HPCProfileBlock,
)
command = CommandBlock.load('poc-qrmi-command')
execution = ExecutionProfileBlock.load('poc-qrmi-execution')
hpc = HPCProfileBlock.load('poc-qrmi-hpc')
assert hpc.hpc_target == 'slurm'
assert hpc.slurm_qpu == 'test_heron'
assert hpc.executable_map['python'] == '/shared/venv/qrmi/bin/python'
assert execution.launcher == 'single'
print('Command:', command.default_args)
print('Quantum blocks OK')
PY
```

The command should reference the deployed `qrmi_task_runner.py`, which adds the
shared checkout to the module path for the example hook. The existing three
blocks can be reused unchanged. No MPI executable or classical BitCount blocks
are needed.

For a new setup only, the following registers the three dedicated blocks in the
currently selected Prefect workspace:

```python
from qcsc_prefect_blocks.common.blocks import (
    CommandBlock, ExecutionProfileBlock, HPCProfileBlock,
)
CommandBlock(
    command_name="poc-qrmi",
    executable_key="python",
    default_args=["/shared/qcsc-prefect/examples/prefect_bitcount_demo/qrmi_task_runner.py"],
).save("poc-qrmi-command", overwrite=True)
ExecutionProfileBlock(
    profile_name="poc-qrmi", command_name="poc-qrmi",
    num_nodes=1, mpiprocs=1, ompthreads=1,
    walltime="00:10:00", launcher="single",
).save("poc-qrmi-execution", overwrite=True)
HPCProfileBlock(
    hpc_target="slurm", queue_cpu="debug",
    slurm_qpu="test_heron", slurm_ntasks=1,
    executable_map={"python": "/shared/venv/qrmi/bin/python"},
).save("poc-qrmi-hpc", overwrite=True)
```

## 4. Run offline checks before submission

From the workflow host's source root:

```bash
PYTHONPATH=packages/qcsc-prefect-blocks/src \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_slurm_runtime.py \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_sqd_qpy_transpile.py \
  packages/qcsc-prefect-executor/tests/test_qrmi_sqd_slurm_poc.py \
  packages/qcsc-prefect-adapters/tests/test_slurm_runtime.py -q --tb=short
```

These tests do not submit quantum work. Optional chemistry dependencies must be
installed for the SQD checks to run. Two strict xfails document the raw-LUCJ QPY
limitation; skipped modules are not evidence of validation.

## 5. Submit H2 with 100 shots and meas only

The following creates a real Slurm and quantum job. Omit `--reset-mitigation`.
Keep the workflow process running until it observes completion.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m examples.qrmi_sqd_validation.flow \
    --shots 100 --work-dir "$QRMI_CHECK_DIR/runs"
```

The circuit is H2 at 0.74 angstrom, STO-3G, four logical qubits, one LUCJ layer.
The specialized compiler uses optimization level 3, seed 24, SABRE layout trials
4, swap trials 2 and iterations 2. Automatic retries are disabled. Each explicit
rerun creates a new allocation and quantum job.

## 6. Inspect the result

The flow prints a summary. The quantum job directory is `slurm.work_dir` in
`summary.json` and contains:

```text
circuit.qpy              # ffSIM-prepared transport circuit
sampler_request.json     # schema, resource, shots, registers, hook
transpile_config.json    # fixed source provenance and circuit parameters
sampler.slurm
output.out / output.err
isa.qpy                  # circuit compiled for the actual QRMI Target
transpile_report.json    # operations, Target width, registers, host and mode
sampler_input.json       # actual CLI payload
result.json              # raw encoded CLI result
slurm_job.json           # scheduler state and execution record
```

Require all of the following:

- `slurm_job.json`: state `COMPLETED`, exit status 0.
- Request: schema 1, resource `test_heron`, shots 100, registers `{"meas": 4}`.
- Config: pinned commit, transport `ffsim-pre-init-decomposed`, reset mitigation false.
- Report: strategy `sqd-pr42`, specialized hook, mode `batch`, registers `{"meas": 4}`.
- Summary: four logical qubits, 100 shots and counts summing to 100.
- Raw result: one PUB, four-bit meas with 100 shots, no test register.

ISA width may exceed four because the Target can include unused physical qubits.
Matching an ideal molecular distribution is not a success condition for this
transport check. For independent IBM Batch membership confirmation, use the
Task ID in `output.out` to inspect the corresponding provider-side record.

A user-reported library run passed as Slurm job `16` on `slurm-vm1`, before this
focused branch was prepared on latest main. The BitArray entrypoint also passed
as Prefect run `uber-sloth` with four bits and 100 shots. The exact focused branch
has not been rerun on the cluster.

If a run fails, inspect its logs before submitting again. `--request` being
unrecognized indicates an old worker; `ModuleNotFoundError` indicates installation
or path issues; `norb` errors indicate raw ffSIM gates reached the QPY boundary;
source mismatch indicates inconsistent pinned snapshots. Do not publish
acquisition tokens or service credentials when sharing diagnostics.
