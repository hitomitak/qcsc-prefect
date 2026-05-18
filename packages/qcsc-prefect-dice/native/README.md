# DICE Native Build Assets

This directory stores the native build scripts and site-specific build settings
used to build the DICE executable for qcsc-prefect workflows.

These scripts are source-tree helpers only. Installing `qcsc-prefect-dice` with
pip does not run them, download native dependencies, or install a DICE/SBD
binary.

The Python integration lives in:

```text
packages/qcsc-prefect-dice/src/qcsc_prefect_dice
```

The native executable is built separately and its absolute path is passed into
the Prefect block configuration as `dice_executable`.

## Build On A Local Linux Machine

Run the generic build script from this directory:

```bash
cd packages/qcsc-prefect-dice/native
bash ./build_dice.sh
```

If the build succeeds, the executable and runtime libraries are copied to:

```text
packages/qcsc-prefect-dice/native/bin
```

## Build On Miyabi

Because DICE downloads dependencies and produces native binaries, it is usually
better to build in shared storage under `/work` rather than inside the git
checkout.

The recommended workflow is to copy just this `native/` directory into a fresh
release directory under shared storage, then build there in a PrePost session:

```bash
SRC=/work/<user>/<repo>/packages/qcsc-prefect-dice/native
REL=/work/<group>/share/qcsc-prefect-dice/releases/$(date +%Y%m%d_%H%M%S)

mkdir -p "$REL"
cp -rf "$SRC" "$REL"
```

Then build in place on Miyabi:

```bash
cd "$REL/native"
bash ./build_dice_miyabi.sh
```

This release-directory approach avoids overwriting a `Dice` binary that may
still be used by running jobs.

```bash
SRC=/work/<user>/<repo>/packages/qcsc-prefect-dice/native
DST_BASE=/work/<group>/share/qcsc-prefect-dice

mkdir -p "$DST_BASE"
rm -rf "$DST_BASE/native"
cp -rf "$SRC" "$DST_BASE"

cd "$DST_BASE/native"
bash ./build_dice_miyabi.sh
```

The Miyabi-oriented build script:

- loads the required compiler and MPI modules
- passes Miyabi-specific compile and link flag overrides to `make`
- embeds `RUNPATH=$ORIGIN` into `Dice` so it can find bundled libraries in `bin/`
- verifies the copied binary still exposes that runtime path when `readelf` is available
- copies the final `Dice` binary and required shared libraries into `bin/`

You can confirm the runtime path after build with:

```bash
readelf -d "$REL/native/bin/Dice" | grep -E 'RPATH|RUNPATH'
ldd "$REL/native/bin/Dice" | grep 'not found'
```

## Build On Fugaku

Fugaku support uses the same Python integration and block model, but the DICE
binary itself still needs to be built on Fugaku. The build script here is now
aligned with the working interactive recipe:

```bash
cd packages/qcsc-prefect-dice/native
bash ./build_dice_fugaku.sh
```

This script:

- loads Spack and `fujitsu-mpi@head-gcc8`
- downloads Boost 1.85.0
- clones `caleb-johnson/Dice`
- disables `HAS_AVX2`
- copies `Dice` and bundled Boost `.so` files into `native/bin/`

If you want a one-command submission from the login node:

```bash
cd packages/qcsc-prefect-dice/native
export FUGAKU_GROUP=ra000000
bash ./submit_build_dice_fugaku.sh
```

`submit_build_dice_fugaku.sh` supports two modes:

- `FUGAKU_BUILD_SUBMIT_MODE=interact`
  Runs `pjsub --interact ...` and then executes `build_dice_fugaku.sh`
  automatically on the compute node
- `FUGAKU_BUILD_SUBMIT_MODE=batch`
  Generates a job script with `#PJM` directives and submits it with `pjsub`

Default is `auto`:

- if `FUGAKU_BUILD_RSCGRP=int`, it uses `interact`
- otherwise it uses `batch`

This is important because some Fugaku ACL settings reject normal batch
submission to `rscgrp=int` with:

```text
[ERR.] PJM 0070 pjsub No execute permission: pjsub batch.
```

The defaults match the working recipe:

- `node=1`
- `rscgrp=int`
- `elapse=3:00:00`
- `PJM_LLIO_GFSCACHE=/vol0004`
- `--llio cn-read-cache=off`
- `--mpi "max-proc-per-node=1"`
- `--name dice-build`

For `batch` mode, these values are written into the generated job script as
`#PJM` directives.

For `interact` mode, `pjsub --interact` does not use a batch script, so the
same values still have to be passed on the `pjsub` command line. `wait-time=600`
is only used in `interact` mode.

You can override them with:

```bash
export FUGAKU_BUILD_RSCGRP=int
export FUGAKU_BUILD_NODE_COUNT=1
export FUGAKU_BUILD_ELAPSE=3:00:00
export FUGAKU_BUILD_GFSCACHE=/vol0004
export FUGAKU_BUILD_MPI_OPTION=max-proc-per-node=1
export FUGAKU_BUILD_WAIT_TIME=600
export FUGAKU_BUILD_SUBMIT_MODE=interact
```

If you need to pin a DICE revision:

```bash
export DICE_REF=<git-ref>
bash ./build_dice_fugaku.sh
```

At runtime, Fugaku previously required:

```bash
export LD_LIBRARY_PATH=/path/to/.../bin:$LD_LIBRARY_PATH
```

`qcsc-prefect-dice.create_dice_blocks(...)` now injects that automatically for
Fugaku by prepending the directory that contains `dice_executable` to
`LD_LIBRARY_PATH` unless you explicitly override `environments["LD_LIBRARY_PATH"]`.

## Configure The Block

Point your DICE block configuration at the built executable:

```toml
dice_executable = "/abs/path/to/packages/qcsc-prefect-dice/native/bin/Dice"
```

Or, if you built in shared storage:

```toml
dice_executable = "/work/<group>/share/qcsc-prefect-dice/releases/<release>/native/bin/Dice"
```

For Fugaku, the runtime library path is derived automatically from the parent
directory of `dice_executable`, so the current package can be operated like the
Miyabi setup without manually exporting `LD_LIBRARY_PATH` before every run.
