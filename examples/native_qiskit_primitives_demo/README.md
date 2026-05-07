# Native Qiskit Primitives Live Test

This example submits real native Qiskit Runtime jobs through the
`qcsc_prefect.integrations.qiskit` Prefect tasks.

It can test:

- Sampler simple mode: `run_sampler_task`
- Sampler robust mode: `submit_sampler_job_task` then `fetch_qiskit_job_result_task`
- Estimator simple mode: `run_estimator_task`
- Estimator robust mode: `submit_estimator_job_task` then `fetch_qiskit_job_result_task`

Running with `--primitive all --mode all` submits four real Qiskit Runtime jobs.

## Prerequisites

- Prefect is configured and reachable.
- `qcsc-prefect-qiskit` dependencies are installed.
- IBM Quantum credentials are available either from a saved Qiskit account,
  environment discovery, or a `QiskitRuntimeConfig` Prefect Block.

## Optional: create the runtime block

This stores the token in a Prefect Block from the named environment variable.
If you want Qiskit to use a saved account file or environment discovery instead,
omit `--token-env` or leave the environment variable unset.

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

## Run only one case

Sampler robust mode:

```bash
PYTHONPATH=packages/qcsc-prefect-qiskit/src \
uv run --package qcsc-prefect-qiskit python examples/native_qiskit_primitives_demo/flow.py \
  --runtime-block ibm-runtime \
  --primitive sampler \
  --mode robust
```

Estimator simple mode:

```bash
PYTHONPATH=packages/qcsc-prefect-qiskit/src \
uv run --package qcsc-prefect-qiskit python examples/native_qiskit_primitives_demo/flow.py \
  --runtime-block ibm-runtime \
  --primitive estimator \
  --mode simple
```

The script prints a compact JSON summary containing the submitted job IDs.
Detailed metadata and result artifacts are registered in Prefect.
