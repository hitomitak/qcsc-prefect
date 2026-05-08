# Native Qiskit Primitives Live Test

This example submits real native Qiskit Runtime jobs through Prefect.

Native Qiskit performs quantum execution. Prefect provides orchestration, logs,
retries, and artifacts. qcsc-prefect does not reimplement Qiskit Runtime APIs
and does not introduce `PrefectBackendV2`, `PrefectSamplerV2`, or
`PrefectEstimatorV2`.

For design details and when to use each style, see
[`docs/howto/howto_use_native_qiskit_prefect.md`](../../docs/howto/howto_use_native_qiskit_prefect.md).

## Minimal conversion patterns

Original native Qiskit code:

```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

service = QiskitRuntimeService()
backend = service.backend("ibm_fez")

pubs = [isa_circuit]
sampler = SamplerV2(mode=backend)
job = sampler.run(pubs, shots=100)
result = job.result()
```

### 1. Wrap the existing program

Keep `QiskitRuntimeService()`, `service.backend(...)`, and `sampler.run(...)`
inside your code. Add only a Prefect task boundary.

```python
from prefect import flow, task
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


@task
def run_existing_qiskit_program(pubs, shots):
    service = QiskitRuntimeService()
    backend = service.backend("ibm_fez")
    sampler = SamplerV2(mode=backend)
    job = sampler.run(pubs, shots=shots)
    return {"job_id": job.job_id(), "result": job.result()}


@flow
def qiskit_flow(pubs):
    return run_existing_qiskit_program(pubs, 100)
```

Runnable file: `existing_program_task.py`.

### 2. Use `run_sampler_task`

Keep `pubs` as native Qiskit input. Replace manual `SamplerV2(...).run(...)`
and `job.result()` with the helper task.

```python
from qcsc_prefect.integrations.qiskit import QiskitRuntimeConfig, run_sampler_task

runtime_config = QiskitRuntimeConfig(backend_name="ibm_fez")

result = await run_sampler_task(
    pubs,
    runtime_config=runtime_config,
    shots=100,
)
```

No saved Prefect Block is required for this form. If the config has no token,
native Qiskit discovery is used.

You can also use a saved Block:

```python
result = await run_sampler_task(
    pubs,
    runtime_block_name="ibm-runtime",
    shots=100,
)
```

### 3. Use robust submit/fetch mode

Use this when a retry after job submission must not submit a duplicate job.

```python
from qcsc_prefect.integrations.qiskit import (
    QiskitRuntimeConfig,
    fetch_qiskit_job_result_task,
    submit_sampler_job_task,
)

runtime_config = QiskitRuntimeConfig(backend_name="ibm_fez")

job_ref = await submit_sampler_job_task(
    pubs,
    runtime_config=runtime_config,
    shots=100,
)
result = await fetch_qiskit_job_result_task(
    runtime_config=runtime_config,
    job_reference=job_ref,
    pubs=pubs,
)
```

Estimator uses the same pattern with `run_estimator_task` and
`submit_estimator_job_task`.

## Prerequisites

- Prefect is configured and reachable.
- `qcsc-prefect-qiskit` dependencies are installed.
- IBM Quantum credentials are available through a saved Qiskit account,
  environment discovery, or a `QiskitRuntimeConfig` Prefect Block.

## Run one wrapper example

This path keeps existing Qiskit authentication and does not require a runtime
Block:

```bash
cd /Users/hitomi/Project/qcsc-prefect

uv run --package qcsc-prefect-qiskit python \
  examples/native_qiskit_primitives_demo/existing_program_task.py \
  --backend-name ibm_fez \
  --shots 100
```

## Optional: create a runtime Block

The full live-test script uses `runtime_block_name`. Create the Block first if
you want to run that script. Token storage is optional; omit `--token-env` to
let Qiskit use saved account or environment discovery.

```bash
cd /Users/hitomi/Project/qcsc-prefect
export IBM_QUANTUM_TOKEN=your_token

PYTHONPATH=packages/qcsc-prefect-qiskit/src \
uv run --package qcsc-prefect-qiskit python examples/native_qiskit_primitives_demo/flow.py \
  --save-runtime-block \
  --runtime-block ibm-runtime \
  --backend-name ibm_fez \
  --channel ibm_quantum_platform \
  --instance your_instance_or_crn \
  --token-env IBM_QUANTUM_TOKEN \
  --primitive sampler \
  --mode simple
```

## Run all live checks

This submits four real Qiskit Runtime jobs: Sampler simple, Sampler robust,
Estimator simple, and Estimator robust.

```bash
cd /Users/hitomi/Project/qcsc-prefect

PYTHONPATH=packages/qcsc-prefect-qiskit/src \
uv run --package qcsc-prefect-qiskit python examples/native_qiskit_primitives_demo/flow.py \
  --runtime-block ibm-runtime \
  --primitive all \
  --mode all \
  --shots 100 \
  --precision 0.2
```

The script prints a compact JSON summary containing submitted job IDs. Detailed
metadata and result artifacts are registered in Prefect.
