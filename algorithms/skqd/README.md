# Sample-based Krylov Quantum Diagonalization

This package provides implementation for the QCSC algorithm demonstrated in the paper  [Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization](https://arxiv.org/abs/2501.09702).

## 🚀 Getting Started

Before starting, make sure:

- You have completed [How to Set Up IBM Quantum Access Credentials for Prefect](../../docs/howto/howto_setup_prefect_qiskit.md).
- You have completed [How to Set Up the MDX Workflow Server for QCSC Execution](../../docs/howto/howto_setup_mdx_server.md).
- You have built the DICE executable under `packages/qcsc-prefect-dice/native` and know its absolute path.

From the repository root, install the local qcsc-prefect packages and the SKQD workflow:

```bash
cd /path/to/qcsc-prefect

uv pip install --no-deps \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-executor \
  -e packages/qcsc-prefect-dice

uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/skqd
```

After installation, you can deploy the workflow in a Prefect server:

```bash
skqd-deploy
```

Before running on Miyabi or Fugaku, create the DICE-related blocks and
`sampler_options` variable:

```bash
cp algorithms/skqd/skqd_blocks.example.toml algorithms/skqd/skqd_blocks.toml
python algorithms/skqd/create_blocks.py --config algorithms/skqd/skqd_blocks.toml --hpc-target miyabi
```

For Fugaku, start from `algorithms/skqd/skqd_blocks.fugaku.example.toml` and pass
`--hpc-target fugaku` instead.

Set `dice_executable` in the TOML file to the built binary, for example:

```toml
dice_executable = "/path/to/qcsc-prefect/packages/qcsc-prefect-dice/native/bin/Dice"
```
