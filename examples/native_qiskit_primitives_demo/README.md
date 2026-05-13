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

### 3. Use wrapper classes

Use this when you want a small object that keeps runtime settings and delegates
to the same Prefect tasks internally.

```python
from qcsc_prefect.integrations.qiskit import QCSCSamplerV2, QiskitRuntimeConfig

sampler = QCSCSamplerV2(
    runtime_config=QiskitRuntimeConfig(backend_name="ibm_fez"),
)
job = await sampler.run(
    pubs,
    shots=100,
    cache_submit=True,
    cache_result=True,
    retry_fetch=True,
)
result = await job.result()
```

If you want the full structured dictionary returned by the underlying fetch
task, use `run_and_fetch(...)`:

```python
output = await sampler.run_and_fetch(
    pubs,
    shots=100,
    cache_submit=True,
    cache_result=True,
    retry_fetch=True,
)
result = output["result"]
```

Estimator uses `QCSCEstimatorV2(...).run(pubs, precision=...)`.

Cache and fetch behavior with wrappers:

- `cache_submit=True` lets an identical rerun reuse the cached `job_id` instead
  of submitting a duplicate Runtime job.
- `job.result()` fetches the native Qiskit result for that `job_id`.
- `job.output()` fetches the full qcsc-prefect output dictionary, including
  result metadata and artifact fields.
- `cache_result=True` stores the fetched output in Prefect result storage with
  `compressed/pickle`; if that local cache exists, the same `job_id` can be
  restored without asking IBM Quantum Platform again.
- `retry_fetch=True` retries the fetch step only.

```python
job = await sampler.run(
    pubs,
    shots=100,
    cache_submit=True,
    cache_result=True,
    retry_fetch=True,
)

output = await job.output()
result = output["result"]
```

### 4. Use robust submit/fetch mode

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

### 5. Add submit cache and fetch retry

Submitting a Qiskit Runtime job is not safely retryable by default: if the
client loses the response after the job was accepted, a retry can create a
duplicate quantum job. For robust execution, cache the submit task and retry
only the fetch task. If you also want repeated fetches of the same `job_id` to
return the saved result, add Prefect result caching to the fetch task.

Starting from the robust native pattern:

```python
job = sampler.run(pubs, shots=100)
result = job.result()
```

Add these Prefect pieces:

```python
from datetime import timedelta

from prefect import flow
from qcsc_prefect.integrations.qiskit import (
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
        runtime_block_name="ibm-runtime",
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
        runtime_block_name="ibm-runtime",
        job_reference=job_ref,
        pubs=pubs,
    )
```

The cache key uses only stable execution identity: `input_digest`,
`runtime_block_name`, `shots`, and safe options. It does not serialize raw
circuits inside Prefect's `cache_key_fn`. `input_digest` is generated before
the task call.

Use `cache_scope="flow"` for the default behavior: reruns of the same Flow can
reuse the submit cache. Use `cache_scope="global"` only when different Flows
with the same circuit, backend, shots, and options should share cached submit
references.

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

## Run the cache/retry live check

This submits one real Sampler job on the first run. Run the same command again
to confirm that `submit-qiskit-sampler-job` becomes `Cached` in Prefect and the
same `job_id` is reused. With `--enable-result-cache`, the fetch result is also
persisted by Prefect and can be restored by job ID.

```bash
cd /Users/hitomi/Project/qcsc-prefect

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

For cross-Flow submit cache reuse, change `--cache-scope flow` to
`--cache-scope global`. The runtime Block name is still part of the submit cache
key, so use the same `--runtime-block` when checking reuse with this example.
`--enable-result-cache` lets Prefect restore the fetched result locally if the
same job ID is requested again.
