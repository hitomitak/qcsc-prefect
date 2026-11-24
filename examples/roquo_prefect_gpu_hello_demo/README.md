# ROQUO GPU Prefect Block Hello Demo

This example runs one small job on one ROQUO GPU through the shared Slurm executor
(`hpc_target="slurm"`). It is the check described in
[Run a One-GPU Job on ROQUO](../../docs/howto/howto_run_gpu_job_on_roquo.md).

| File | Content |
| --- | --- |
| `roquo_gpu.example.toml` | account, partition, reservation, GPU request and Block names |
| `create_blocks.py` | saves `cmd-roquo-gpu-hello`, `exec-roquo-gpu-1` and `hpc-roquo-gpu` from that file |
| `flow.py` | submits `hello_gpu.sh` with `run_job_from_blocks` and returns the job record |
| `hello_gpu.sh` | prints the node and the GPU, then exits with the status it is given |

On a ROQUO login node:

```bash
cp examples/roquo_prefect_gpu_hello_demo/roquo_gpu.example.toml roquo_gpu.toml
uv run python examples/roquo_prefect_gpu_hello_demo/create_blocks.py \
  --config roquo_gpu.toml --account <account> [--reservation <name>]
uv run python -c "import asyncio; from examples.roquo_prefect_gpu_hello_demo.flow import roquo_gpu_hello_flow; print(asyncio.run(roquo_gpu_hello_flow()))"
```

The job scripts that these Blocks give are tested in
`packages/qcsc-prefect-executor/tests/test_roquo_gpu_profile.py`.
