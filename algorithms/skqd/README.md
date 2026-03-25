# Sample-based Krylov Quantum Diagonalization

This package provides implementation for the QCSC algorithm demonstrated in the paper  [Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization](https://arxiv.org/abs/2501.09702).

## 🚀 Getting Started

Before starting, make sure:

- You have completed [How to Set Up IBM Quantum Access Credentials for Prefect](../howto/setup_prefect_qiskit.md).
- You have completed [How to Set Up the MDX Workflow Server for QCSC Execution](../howto/setup_mdx_server.md).

You can install the SKQD workflow with the uv package manager:

```bash
uv pip install -e ./qii-miyabi-kawasaki/algorithms/skqd
```

After installation, you can deploy the workflow in a Prefect server:

```bash
skqd-deploy
```
