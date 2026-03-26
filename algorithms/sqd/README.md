# Sample-based Quantum Diagonalization

This package provides implementation for the QCSC algorithm demonstrated in the paper [Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068).

## 🚀 Getting Started

Before starting, make sure:

- You have completed [How to Set Up IBM Quantum Access Credentials for Prefect](../../docs/howto/howto_setup_prefect_qiskit.md).
- You have completed [How to Set Up the MDX Workflow Server for QCSC Execution](../../docs/howto/howto_setup_mdx_server.md).
- You have built the DICE executable under `packages/qcsc-prefect-dice/native` and know its absolute path.

From the repository root, install the local qcsc-prefect packages and the SQD workflow:

```bash
cd /path/to/qcsc-prefect

uv pip install --no-deps \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-executor \
  -e packages/qcsc-prefect-dice

uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/sqd
```

After installation, you can deploy the workflow in a Prefect server:

```bash
sqd-deploy
```

Before running on Miyabi or Fugaku, create the DICE-related blocks and
`sampler_options` variable:

```bash
cp algorithms/sqd/sqd_blocks.example.toml algorithms/sqd/sqd_blocks.toml
python algorithms/sqd/create_blocks.py --config algorithms/sqd/sqd_blocks.toml --hpc-target miyabi
```

For Fugaku, start from `algorithms/sqd/sqd_blocks.fugaku.example.toml` and pass
`--hpc-target fugaku` instead.

Set `dice_executable` in the TOML file to the built binary, for example:

```toml
dice_executable = "/path/to/qcsc-prefect/packages/qcsc-prefect-dice/native/bin/Dice"
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
