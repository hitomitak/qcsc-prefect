# Miyabi Prefect Block Hello Demo

This example uses `run_miyabi_job` to submit a real `qsub` job from Prefect Blocks.

## Prerequisites

- Run on a Miyabi login node
- `qsub` and `qstat` are available in `PATH`
- Prefect API is reachable

## 1) Sync dependencies

```bash
cd /Users/hitomi/Project/hpc-prefect
uv sync
```

## 2) Register block types

```bash
cd /Users/hitomi/Project/hpc-prefect
uv run prefect block register -m hpc_prefect_blocks.miyabi.blocks
```

## 3) Create demo Blocks

```bash
cd /Users/hitomi/Project/hpc-prefect
export MIYABI_PBS_PROJECT=gz09
export MIYABI_PBS_QUEUE=regular-c
uv run python examples/miyabi_prefect_hello_demo/create_blocks.py
```

This script creates:
- `cmd-hello-demo`
- `exec-hello-single`
- `hpc-miyabi` (instance of `HPCProfileBlock`)

To use an executable other than `hello_demo.sh`:

```bash
export MIYABI_DEMO_EXECUTABLE=/path/to/your/executable
```

## 4) Run the flow

```bash
cd /Users/hitomi/Project/hpc-prefect
uv run python -c "import asyncio; from examples.miyabi_prefect_hello_demo.flow import miyabi_prefect_block_hello_flow; print(asyncio.run(miyabi_prefect_block_hello_flow()))"
```

Example return value:

```text
{'job_id': '12345.miyabi', 'exit_status': 0, 'work_dir': '/work/.../miyabi_prefect_block_hello'}
```
