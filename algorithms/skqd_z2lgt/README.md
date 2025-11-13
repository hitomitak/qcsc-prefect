# Sample-based Krylov quantum diagonalization of two-dimensional Z2 lattice gauge theory

This package provides an implementation of sample-based Krylov quantum diagonalization (SKQD)
applied to pure two-dimensional Z2 lattice gauge theory (LGT) problems. Samples are drawn from
Trotterized time evolution of the LGT model, and its Hamiltonian is projected and diagonalized onto
the sub-Hilbert space spanned by the sample bitstrings.

## 🚀 Getting Started

Before starting, make sure:

- You have completed [How to Set Up IBM Quantum Access Credentials for Prefect](../../docs/howto/howto_setup_prefect_qiskit.md).
- You have completed [How to Set Up the MDX Workflow Server for QCSC Execution](../../docs/howto/howto_setup_mdx_server.md).

The workflow for SKQD utilizes both Miyabi-C and Miyabi-G resources (in principle we only need the
latter, but using Miyabi-C is more economical if GPU is not required). You would therefore need to
set up a python environment for each architecture separately:

```bash
ssh miyabi-g.jcahpc.jp
uv venv ~/venv/skqd_z2lgt_aarch64 -p 3.12
source ~/venv/skqd_z2lgt_aarch64/bin/activate
uv pip install -e ./qii-miyabi-kawasaki/algorithms/skqd_z2lgt
```

```bash
ssh miyabi-c.jcahpc.jp
uv venv ~/venv/skqd_z2lgt_x86_64 -p 3.12
source ~/venv/skqd_z2lgt_x86_64/bin/activate
uv pip install -e ./qii-miyabi-kawasaki/algorithms/skqd_z2lgt
uv pip install -e ./qii-miyabi-kawasaki/framework/prefect-miyabi
```

Prefect-miyabi is required only in the architecture where the Prefect flow is deployed. Presumably
this would be x86_64, as both the MDX Workflow Server and Miyabi prepost nodes are x86_64.

You can deploy the workflow in a Prefect server with

```bash
skqd-z2lgt-deploy
```

Check [Run Hybrid Workflow](../../docs/tutorials/run_sqd_workflow.md) to learn how to submit Prefect
flow runs.

## 🧑‍💻 Contribution Guidelines

This package is still under active development. Contributions are welcome.
