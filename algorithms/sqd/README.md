# Sample-based Quantum Diagonalization

This package provides implementation for the QCSC algorithm demonstrated in the paper [Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068).

## 🚀 Getting Started

Before starting, make sure:

- You have completed [How to Set Up IBM Quantum Access Credentials for Prefect](../howto/setup_prefect_qiskit.md).
- You have completed [How to Set Up the MDX Workflow Server for QCSC Execution](../howto/setup_mdx_server.md).

You can install the SQD workflow with the uv package manager:

```bash
uv pip install -e ./qii-miyabi-kawasaki/algorithms/sqd
```

After installation, you can deploy the workflow in a Prefect server:

```bash
sqd-deploy
```

See the [Run Hybrid Workflow](../../docs/tutorials/run_sqd_workflow.md) tutorial for the end-to-end steps.

## 🧭 MPI Parameter Tuning

The [perf](./perf/) module also includes example code for implementing a meta workflow.
This workflow internally runs SQD workflows with different flow parameters, and measures the weak and strong scaling of the DICE solver.

## 🧑‍💻 Contribution Guidelines

This package is a reference implementation of the SQD experiment described in the publication.
It is considered feature-complete and is now in maintenance mode.

Further contributions are limited to bug fixes and improvements to Prefect usage patterns.  
Workflow developers may use this as a testbed for workflow technology research.
