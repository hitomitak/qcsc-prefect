# Native Qiskit Execution with Prefect

Use this integration when you want native Qiskit Runtime execution with Prefect
orchestration around it.

- Qiskit and `qiskit-ibm-runtime` submit and manage quantum jobs.
- Prefect provides flow/task orchestration, logs, retries, and artifacts.
- qcsc-prefect adds thin configuration and task helpers around native Qiskit
  objects.

## Non-goals

This integration does not:

- reimplement Qiskit Runtime APIs
- call IBM Runtime REST APIs directly
- introduce `PrefectBackendV2`
- introduce `PrefectSamplerV2`
- introduce `PrefectEstimatorV2`

## Why `QiskitRuntimeConfig` exists

`QiskitRuntimeConfig` is not a replacement for `QiskitRuntimeService`.
It is a thin Prefect-friendly configuration object for the qcsc-prefect helper
tasks.

Internally it still does native Qiskit work:

```python
service = QiskitRuntimeService(...)
backend = service.backend(backend_name)
```

Use it when you want `run_sampler_task`, `run_estimator_task`, or robust
submit/fetch tasks to create the native backend for you.

You can pass it directly:

```python
from qcsc_prefect.integrations.qiskit import QiskitRuntimeConfig

runtime_config = QiskitRuntimeConfig(backend_name="ibm_fez")
```

If no token is stored in the config, `qiskit-ibm-runtime` uses its normal
account discovery, such as saved account files or environment variables.

You can also save it as a Prefect Block and pass `runtime_block_name`:

```python
result = await run_sampler_task(
    pubs,
    runtime_block_name="ibm-runtime",
    shots=100,
)
```

## When a Block is required

A saved `QiskitRuntimeConfig` Block is not always required.

Use no Block when:

- you wrap existing Qiskit code and keep `QiskitRuntimeService()` directly
- you pass `runtime_config=QiskitRuntimeConfig(backend_name="...")` directly
- credentials are already handled by Qiskit saved account or environment
  discovery

Use a saved Block when:

- you want to reference runtime settings by name in a deployment
- you want to manage backend/account settings from Prefect UI or CLI
- you want Prefect to store optional credentials or account metadata
- multiple flows should share the same runtime configuration

Avoid passing token-bearing config objects directly as task parameters. Prefer a
saved Prefect Block for secrets.

## How to convert an existing Qiskit program

Suppose your current native Qiskit program looks like this:

```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

service = QiskitRuntimeService()
backend = service.backend("ibm_fez")

pubs = [isa_circuit]
sampler = SamplerV2(mode=backend)
job = sampler.run(pubs, shots=100)
result = job.result()
```

Keep these parts unchanged in all styles:

- circuit construction
- transpilation / ISA circuit creation
- `pubs` creation
- native Qiskit input/output objects

Then choose how much of the execution block to replace.

| Existing Qiskit line | Example 1: wrap existing code | Example 2: `run_sampler_task` | Example 3: robust submit/fetch |
| --- | --- | --- | --- |
| `service = QiskitRuntimeService()` | Keep as-is inside the task/helper. | Replace with `QiskitRuntimeConfig`; the helper task creates the service internally. | Replace with `QiskitRuntimeConfig`; submit/fetch tasks create the service internally. |
| `backend = service.backend("ibm_fez")` | Keep as-is. | Backend name moves to `QiskitRuntimeConfig(backend_name="ibm_fez")` or a saved Block. | Same as Example 2. |
| `sampler = SamplerV2(mode=backend)` | Keep as-is. | Remove it; `run_sampler_task` creates native `SamplerV2`. | Remove it; `submit_sampler_job_task` creates native `SamplerV2`. |
| `job = sampler.run(pubs, shots=100)` | Keep as-is. | Replace with `await run_sampler_task(...)`. | Replace with `await submit_sampler_job_task(...)`. |
| `result = job.result()` | Keep as-is. | Handled by `run_sampler_task`. | Replace with `await fetch_qiskit_job_result_task(...)`. |

For Estimator, the same idea applies. Replace `SamplerV2` with `EstimatorV2`,
`shots` with `precision`, and use the Estimator helper tasks.

## Example 1: wrap an existing Qiskit program

Choose this when you want the smallest change to an existing program.
Keep `QiskitRuntimeService()`, `service.backend(...)`, `SamplerV2`, and
`job.result()` in your code. Add a Prefect task boundary around it.

Before:

```python
service = QiskitRuntimeService()
backend = service.backend("ibm_fez")
sampler = SamplerV2(mode=backend)
job = sampler.run(pubs, shots=100)
result = job.result()
```

After:

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
def my_flow(pubs):
    return run_existing_qiskit_program(pubs, 100)
```

This style does not require `QiskitRuntimeConfig`.

## Example 2: use `run_sampler_task`

Choose this when you want qcsc-prefect to create `SamplerV2`, wait for the
result, and create Prefect artifacts.

Before:

```python
service = QiskitRuntimeService()
backend = service.backend("ibm_fez")
sampler = SamplerV2(mode=backend)
job = sampler.run(pubs, shots=100)
result = job.result()
```

After, with an inline runtime config:

```python
from prefect import flow
from qcsc_prefect.integrations.qiskit import QiskitRuntimeConfig, run_sampler_task


@flow
async def sampler_flow(pubs):
    runtime_config = QiskitRuntimeConfig(backend_name="ibm_fez")
    return await run_sampler_task(
        pubs,
        runtime_config=runtime_config,
        shots=100,
        artifact_key="native-qiskit-sampler",
    )
```

This keeps credentials in native Qiskit discovery if the config has no token.

The saved-Block form is equivalent from the task's point of view:

```python
return await run_sampler_task(
    pubs,
    runtime_block_name="ibm-runtime",
    shots=100,
)
```

Estimator uses the same pattern with `run_estimator_task`.

## Example 3: robust submit/fetch mode

Choose this when a task retry after submission must not submit a duplicate
Qiskit Runtime job. Submit returns a job reference. Fetch retrieves the existing
job by ID and records artifacts.

Before:

```python
sampler = SamplerV2(mode=backend)
job = sampler.run(pubs, shots=100)
result = job.result()
```

After:

```python
from prefect import flow
from qcsc_prefect.integrations.qiskit import (
    QiskitRuntimeConfig,
    fetch_qiskit_job_result_task,
    submit_sampler_job_task,
)


@flow
async def robust_sampler_flow(pubs):
    runtime_config = QiskitRuntimeConfig(backend_name="ibm_fez")
    job_ref = await submit_sampler_job_task(
        pubs,
        runtime_config=runtime_config,
        shots=100,
    )
    return await fetch_qiskit_job_result_task(
        runtime_config=runtime_config,
        job_reference=job_ref,
        pubs=pubs,
        artifact_key="native-qiskit-sampler-robust",
    )
```

The same pattern is available for Estimator with `submit_estimator_job_task`.

## Live example

Runnable examples are in `examples/native_qiskit_primitives_demo/`.

```bash
PYTHONPATH=packages/qcsc-prefect-qiskit/src \
uv run --package qcsc-prefect-qiskit python examples/native_qiskit_primitives_demo/flow.py \
  --runtime-block ibm-runtime \
  --primitive all \
  --mode all \
  --shots 100 \
  --precision 0.2
```

This command submits real Qiskit Runtime jobs. No IBM Quantum token is included
in the repository or examples.
