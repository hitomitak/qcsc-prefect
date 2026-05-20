# Native Qiskit + HPC Workflow

This Level 2 tutorial connects native Qiskit Runtime execution and HPC
execution in one Prefect workflow.

Use Qiskit as usual for quantum execution. Use `qcsc-prefect` only where you
want Prefect orchestration, artifacts, retries, caching, or HPC submission.

## What you will learn

- How native Qiskit execution and an HPC solver task fit into one flow
- Which `qcsc-prefect` helpers are used at the Prefect boundary
- How this differs from the older `prefect-qiskit` style

## Prerequisites

You need:

- IBM Quantum credentials for real-device Qiskit Runtime execution
- Access to a real HPC environment, such as Miyabi or Fugaku
- A reachable Prefect backend for blocks, variables, logs, and artifacts
- The BitCount executable and Prefect blocks from one of the Level 1 tutorials

Start from one of these tutorials and complete the random-source run first:

- [Miyabi Workflow](create_qcsc_workflow_for_miyabi.md)
- [Fugaku Workflow](create_qcsc_workflow_for_fugaku.md)

Then configure native Qiskit Runtime using:

- [Native Qiskit on Prefect](../howto/howto_use_native_qiskit_prefect.md)

## Installation

Install the package with the native Qiskit integration extra:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "qcsc-prefect[qiskit]"
```

The `qiskit` extra installs the `qcsc-prefect-qiskit` integration. It is not
the external `prefect-qiskit` package.

For unreleased tutorial changes, use the repository development setup instead
of the released package.

## Workflow shape

The integrated BitCount flow uses this shape:

1. Build a normal Qiskit circuit and transpile it for the selected backend.
2. Submit the native Qiskit Runtime job through a `qcsc-prefect` helper task.
3. Convert the sampler bitstrings into the binary input expected by the HPC
   BitCount executable.
4. Submit and monitor the HPC job through block-driven `qcsc-prefect` executor
   helpers.
5. Read the HPC output and publish Prefect artifacts.

The runnable implementation is:

- `examples/prefect_bitcount_demo/flow_optimized.py`
- `examples/prefect_bitcount_demo/quantum_sampling.py`

Important API boundaries:

- `QiskitRuntimeConfig` stores or loads Prefect-friendly runtime settings.
- `run_sampler_task` submits native Qiskit Runtime sampler work and records
  Prefect metadata/artifacts.
- `run_job_from_blocks` resolves `CommandBlock`, `ExecutionProfileBlock`, and
  `HPCProfileBlock`, then submits the HPC job.

## Run on Miyabi

After the Miyabi random-source BitCount run succeeds, verify that the Qiskit
Runtime block exists:

```bash
prefect block inspect qiskit_runtime_config/ibm-runner
```

Then run the same flow with the real-device quantum source:

```bash
python examples/prefect_bitcount_demo/flow_optimized.py \
  --quantum-source real-device \
  --runtime-block ibm-runner \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-mpi \
  --hpc-profile-block hpc-miyabi-bitcount \
  --options-variable miyabi-bitcount-options
```

## Run on Fugaku

After the Fugaku random-source BitCount run succeeds, configure the certificate
environment needed for IBM Quantum access on Fugaku, then verify the Qiskit
Runtime block:

```bash
export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')
prefect block inspect qiskit_runtime_config/ibm-runner
```

Then run the same flow with Fugaku block names:

```bash
python examples/prefect_bitcount_demo/flow_optimized.py \
  --quantum-source real-device \
  --runtime-block ibm-runner \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-fugaku \
  --hpc-profile-block hpc-fugaku-bitcount \
  --options-variable fugaku-bitcount-options \
  --script-filename bitcount_optimized.pjm
```

## Where to see results

Use the Prefect UI or console logs to inspect:

- the Qiskit sampler task
- the HPC BitCount task
- the HPC metrics artifact
- the final sampler count dictionary artifact

The HPC working directory also contains the generated scheduler script,
`input.bin`, and `hist_u64.bin`.

## How this differs from older `prefect-qiskit` style

This workflow does not use `PrefectBackendV2`, `PrefectSamplerV2`, or
`PrefectEstimatorV2`.

Qiskit remains responsible for Runtime execution. `qcsc-prefect` adds Prefect
task boundaries, artifacts, retry/caching helpers, and HPC orchestration around
that native Qiskit execution.

## What to read next

- [Native Qiskit on Prefect](../howto/howto_use_native_qiskit_prefect.md) for
  lower-level Qiskit helper patterns
- [Miyabi Workflow](create_qcsc_workflow_for_miyabi.md) for block setup on
  Miyabi
- [Fugaku Workflow](create_qcsc_workflow_for_fugaku.md) for block setup on
  Fugaku
- [Closed Loop Workflow](run_sbd_closed_loop_workflow.md) when you are ready
  for the Level 3 SBD workflow
