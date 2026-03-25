# DICE Native Build Assets

This directory stores the native build scripts and site-specific build settings
used to build the DICE executable for qcsc-prefect workflows.

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

Sync just this `native/` directory to your shared work area, enter a PrePost
session, then run:

```bash
SRC=/path/to/qcsc-prefect/packages/qcsc-prefect-dice/native/
DST=/work/<group>/share/qcsc-prefect-dice/native/

mkdir -p "$DST"
rsync -a --delete "$SRC" "$DST"
```

Then build in place on Miyabi:

```bash
cd /work/<group>/share/qcsc-prefect-dice/native
bash ./build_dice_miyabi.sh
```

The Miyabi-oriented build script:

- loads the required compiler and MPI modules
- passes Miyabi-specific `CXX` and `CXXFLAGS` overrides to `make`
- embeds `RUNPATH=$ORIGIN` into `Dice` so it can find bundled libraries in `bin/`
- verifies the copied binary still exposes that runtime path when `readelf` is available
- copies the final `Dice` binary and required shared libraries into `bin/`

You can confirm the runtime path after build with:

```bash
readelf -d /work/<group>/share/qcsc-prefect-dice/native/bin/Dice | grep -E 'RPATH|RUNPATH'
ldd /work/<group>/share/qcsc-prefect-dice/native/bin/Dice | grep 'not found'
```

## Configure The Block

Point your DICE block configuration at the built executable:

```toml
dice_executable = "/abs/path/to/packages/qcsc-prefect-dice/native/bin/Dice"
```

Or, if you built in shared storage:

```toml
dice_executable = "/work/<group>/share/qcsc-prefect-dice/native/bin/Dice"
```
