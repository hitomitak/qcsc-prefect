# PR #42 sampling integration against QCSC Prefect 0.3.0

This source-only application adapts the sampling stage of PR #42 at immutable
commit `2d435b19af4648cbbc5fbc6a27ff258c93817d1a` to the public
`qcsc_prefect.integrations.qiskit.qrmi.QrmiRuntime` API with
`scheduler="slurm"`. It returns an ordered native `BitArray` for the next stage.
The application and its fixtures belong to this integration-validation branch.
The generic library is unchanged from tag `v0.3.0`.

## Execution boundary

```text
Pinned LUCJ circuit construction
  -> application-owned ffSIM preparation and gate expansion
  -> QrmiRuntime.sampler -> QPY/JSON transport
  -> compute-node Target and pinned transpilation hook
  -> task_runner -> PrimitiveResult
  -> meas extraction and optional test-based shot selection -> BitArray
```

The injected runtime supplies the resource, shared directory, three block names,
polling interval, timeout and scheduler. The application configures its
preparation and transpilation hooks on a dataclass copy, leaving the caller's
runtime unchanged. A runtime with existing compilation hooks/config is rejected.

The fixed PR's `QuantumRuntime.load`, Prefect Variable lookup and workflow-host
Target acquisition are replaced by explicit runtime/options arguments and the
compute-node compilation hook. The original PR remains unchanged. This is a
sampling-stage adapter; configuration recovery, Fulqrum execution and energy
validation remain outside its scope.

## Preserved sampling contract

- Only `options["params"]["shots"]` is supported. Invalid options fail before
  circuit construction or Slurm submission.
- The application preserves the pinned PR's default of **100,000 shots** when
  shots are omitted. The validation CLI explicitly defaults to **100 shots**.
- Without reset mitigation, return `result[0].data.meas` directly.
- With reset mitigation, retain the `meas` shots whose corresponding `test`
  bitcount is zero, preserving width, shot order and bit order.
- Reject an empty retained sample set explicitly. No random fallback is used.
- Disable automatic retries and result persistence for the H2 validation flow.
  The library's sampling task also disables caching and automatic retries.

`pr42.py` verifies the SHA-256 of the unchanged source snapshot before loading
`CircuitParameters`, `create_ansatz_circuits` and `transpile_circuit`. Both hosts
must have this checkout available and use the same source snapshot.

Raw ffSIM LUCJ gates lose Python attributes during QPY transport. The preparation
hook runs `PRE_INIT` and expands the gates before serialization; this known
unsupported raw-QPY path remains covered by two strict expected-failure tests.

## Local environment

From this branch's repository root, create a separate environment. These commands
install dependencies and do not submit Slurm or quantum workloads:

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/qrmi-pr42-v030-venv
uv sync --all-packages --python 3.12
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  -r examples/qrmi_sqd_validation/requirements.txt \
  -e algorithms/qcsc_workflow_utility \
  -e examples/qrmi_sqd_validation
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
```

The workspace packages are installed from the `v0.3.0` checkout. The application's
scientific dependencies are pinned separately; `--no-sync` retains those pins
during the test runs below. This setup leaves the working cluster environment
and other local virtual environments untouched. The application is installed in
editable mode so the compute-node worker can import its hook from any job
directory. Its pinned fixture remains in this checkout; keep the full checkout
available on both hosts rather than distributing the application as a wheel.

## Local tests

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_pr42_integration.py \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_sqd_qpy_transpile.py -q

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

The integration tests block network access and replace Slurm submission and QRMI
resource discovery. They exercise real H2/STO-3G integrals, LUCJ construction,
QPY round trips, the pinned transpiler, QASM export/import, a fake CLI subprocess
and result decoding. A fresh interpreter also checks hook imports from outside
the checkout, without `PYTHONPATH`. The original sampling function is loaded separately as an
oracle to compare ordered results, including asymmetric bitstrings and nonzero
`test` shots. Both CZ-only and CZ+RZZ Targets are checked.

These tests run the sampling task body directly to remain offline. The unchanged
library suite also includes real Prefect flow/task execution against a temporary
local server. No cluster or QPU execution is established by these local tests.

## Application API

```python
from pathlib import Path
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime
from examples.qrmi_sqd_validation.integration import sample_bitstrings

runtime = QrmiRuntime(
    root_dir=Path("/shared/qrmi-pr42-bitarray/quantum"),
    command_block_name="poc-qrmi-command",
    execution_profile_block_name="poc-qrmi-execution",
    hpc_profile_block_name="poc-qrmi-hpc",
    resource="test_heron",
    scheduler="slurm",
)
bits = await sample_bitstrings(
    circuit_params=params, elec_props=properties, runtime=runtime,
    options={"params": {"shots": 100}},
)
```

Calling the API or running `python -m examples.qrmi_sqd_validation.integration`
submits a real workload with a configured environment. Do this only in the
subsequent hardware-validation stage. Running the module with `--help` is a
read-only CLI check.

The H2 entrypoint saves ordered `bitstrings.txt` and `sampling_summary.json`,
including the source provenance, library version, scheduler, requested/retained
shots and counts. The library retains QPY, request/config, compiler report, raw
result and Slurm metadata under the quantum job directory.

## Validation status

Earlier PoC runs returned four-bit, 100-shot `meas` results on `test_heron`,
including the previous BitArray adapter. Those runs used the earlier runtime API.
Hardware execution against `QrmiRuntime` 0.3.0, hardware acceptance of the `test`
register, the full Fulqrum workflow and energy accuracy are separate, pending
validation stages.
