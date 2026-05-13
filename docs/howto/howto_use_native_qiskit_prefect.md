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

## Example 4: add submit cache and fetch retry

Choose this when you want reruns to avoid duplicate Qiskit Runtime submissions,
while still keeping raw result fetching explicit and retryable.

The important split is:

- cache the submit task, because it returns a lightweight job reference
- retry the fetch task, because it fetches an existing `job_id`
- do not retry submit by default, because duplicate Qiskit jobs can be created
- do not cache raw Qiskit results by default

Starting from native Qiskit:

```python
sampler = SamplerV2(mode=backend)
job = sampler.run(pubs, shots=100)
result = job.result()
```

Add Prefect submit cache and fetch retry:

```python
from datetime import timedelta

from prefect import flow
from qcsc_prefect.integrations.qiskit import (
    QiskitRuntimeConfig,
    build_qiskit_sampler_input_digest,
    fetch_qiskit_job_result_task,
    qiskit_result_fetch_cache_key,
    qiskit_retry_delays,
    qiskit_sampler_submit_cache_key,
    should_retry_qiskit_fetch_failure,
    submit_sampler_job_task,
)


@flow
async def cached_robust_sampler_flow(pubs):
    runtime_config = QiskitRuntimeConfig(backend_name="ibm_fez")

    input_digest = build_qiskit_sampler_input_digest(
        pubs,
        backend_name="ibm_fez",
        shots=100,
        cache_scope="flow",
    )

    job_ref = await submit_sampler_job_task.with_options(
        cache_key_fn=qiskit_sampler_submit_cache_key,
        cache_expiration=timedelta(days=7),
        persist_result=True,
    )(
        pubs,
        runtime_config=runtime_config,
        shots=100,
        input_digest=input_digest,
    )
    return await fetch_qiskit_job_result_task.with_options(
        cache_key_fn=qiskit_result_fetch_cache_key,
        persist_result=True,
        result_serializer="compressed/pickle",
        retries=len(qiskit_retry_delays()),
        retry_delay_seconds=qiskit_retry_delays(),
        retry_condition_fn=should_retry_qiskit_fetch_failure,
    )(
        runtime_config=runtime_config,
        job_reference=job_ref,
        pubs=pubs,
        artifact_key="native-qiskit-sampler-cached-robust",
    )
```

`input_digest` is the stable identity of the Qiskit input and execution
settings. It is passed as task metadata and used by
`qiskit_sampler_submit_cache_key`. The cache key helper intentionally does not
serialize `pubs`, circuits, backends, services, or RuntimeJob objects.

Use `cache_scope="flow"` for the default behavior: reruns of the same Flow can
reuse the submit cache. Use `cache_scope="global"` when different Flows should
share submit cache entries for the same circuit, backend, shots, and options.

For Estimator, use the Estimator variants:

```python
from datetime import timedelta

from prefect import flow
from qcsc_prefect.integrations.qiskit import (
    build_qiskit_estimator_input_digest,
    qiskit_estimator_submit_cache_key,
    submit_estimator_job_task,
)


@flow
async def cached_robust_estimator_flow(pubs):
    input_digest = build_qiskit_estimator_input_digest(
        pubs,
        backend_name="ibm_fez",
        precision=0.2,
        cache_scope="flow",
    )
    return await submit_estimator_job_task.with_options(
        cache_key_fn=qiskit_estimator_submit_cache_key,
        cache_expiration=timedelta(days=7),
        persist_result=True,
    )(
        pubs,
        runtime_block_name="ibm-runtime",
        precision=0.2,
        input_digest=input_digest,
    )
```

When checking this in Prefect UI, the second identical run should show the
submit task as `Cached`. If the same `job_id` has already been fetched
successfully, the fetch task can also restore its result from Prefect's result
cache instead of calling IBM Quantum Platform again.

## Restore with Prefect result cache

IBM Quantum Platform may stop retaining a job or its result after some time.
For automatic retry/rerun recovery, use the same approach as prefect-qiskit:
let Prefect persist the fetch task result with
`result_serializer="compressed/pickle"` and a stable cache key.

Use the same `with_options(...)(...)` style as the robust example:

```python
from qcsc_prefect.integrations.qiskit import (
    fetch_qiskit_job_result_task,
    qiskit_result_fetch_cache_key,
)

result = await fetch_qiskit_job_result_task.with_options(
    cache_key_fn=qiskit_result_fetch_cache_key,
    persist_result=True,
    result_serializer="compressed/pickle",
)(
    runtime_config=runtime_config,
    job_reference=job_ref,
    pubs=pubs,
)
```

On the first successful fetch, Prefect persists the returned dictionary,
including the native Qiskit result object. When the same `job_id` is fetched
again and the cache is still available, Prefect restores that stored result and
does not call IBM Quantum Platform again.

This cache helps after a successful fetch has already been persisted. If the
first fetch never completed, there is no local result to restore.

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

Add `--enable-result-cache` to let Prefect restore a previously fetched result
by Qiskit job ID.

This command submits real Qiskit Runtime jobs. No IBM Quantum token is included
in the repository or examples.

To test submit cache and fetch retry with a real backend, run robust Sampler
mode twice:

```bash
PYTHONPATH=packages/qcsc-prefect-qiskit/src \
uv run --package qcsc-prefect-qiskit python examples/native_qiskit_primitives_demo/flow.py \
  --runtime-block ibm-runtime \
  --primitive sampler \
  --mode robust \
  --shots 100 \
  --enable-submit-cache \
  --enable-fetch-retry \
  --enable-result-cache \
  --cache-scope flow
```

Use `--cache-scope global` when you intentionally want different Flows with the
same Qiskit inputs and runtime settings to share submit cache entries.
