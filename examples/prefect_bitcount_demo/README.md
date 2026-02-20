# Prefect BitCount Demo

This example provides two execution styles:

- `flow_optimized.py`: block-driven execution (works on Miyabi and Fugaku)
- `flow_tutorial_style.py`: legacy `counter.get(bitstrings)` style (Miyabi only)

## Files

- `/Users/hitomi/Project/hpc-prefect/examples/prefect_bitcount_demo/create_blocks.py`
- `/Users/hitomi/Project/hpc-prefect/examples/prefect_bitcount_demo/bitcount_blocks.example.toml`
- `/Users/hitomi/Project/hpc-prefect/examples/prefect_bitcount_demo/flow_optimized.py`
- `/Users/hitomi/Project/hpc-prefect/examples/prefect_bitcount_demo/flow_tutorial_style.py`
- `/Users/hitomi/Project/hpc-prefect/examples/prefect_bitcount_demo/get_counts_integration.py`
- `/Users/hitomi/Project/hpc-prefect/examples/prefect_bitcount_demo/build_on_miyabi.sh`
- `/Users/hitomi/Project/hpc-prefect/examples/prefect_bitcount_demo/build_on_fugaku.sh`

## Build executable

Miyabi:

```bash
cd /Users/hitomi/Project/hpc-prefect
./examples/prefect_bitcount_demo/build_on_miyabi.sh
```

Fugaku:

```bash
cd /Users/hitomi/Project/hpc-prefect
./examples/prefect_bitcount_demo/build_on_fugaku.sh
```

## Create blocks

```bash
cd /Users/hitomi/Project/hpc-prefect
cp examples/prefect_bitcount_demo/bitcount_blocks.example.toml \
   examples/prefect_bitcount_demo/bitcount_blocks.toml
```

Miyabi defaults:

```bash
python examples/prefect_bitcount_demo/create_blocks.py \
  --config examples/prefect_bitcount_demo/bitcount_blocks.toml
```

Fugaku:

```bash
python examples/prefect_bitcount_demo/create_blocks.py \
  --config examples/prefect_bitcount_demo/bitcount_blocks.toml \
  --hpc-target fugaku
```

## Run optimized flow

Miyabi example:

```bash
python examples/prefect_bitcount_demo/flow_optimized.py \
  --runtime-block ibm-runner \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-mpi \
  --hpc-profile-block hpc-miyabi-bitcount \
  --options-variable miyabi-bitcount-options
```

Fugaku example:

```bash
python examples/prefect_bitcount_demo/flow_optimized.py \
  --runtime-block ibm-runner \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-fugaku \
  --hpc-profile-block hpc-fugaku-bitcount \
  --options-variable fugaku-bitcount-options \
  --script-filename bitcount_optimized.pjm
```

## Run legacy tutorial-style flow (Miyabi only)

```bash
python examples/prefect_bitcount_demo/flow_tutorial_style.py
```

`flow_tutorial_style.py` expects `miyabi-tutorial` block/variable and is intentionally Miyabi-only.
