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
- passes Miyabi-specific `CXX` and `CXXFLAGS` overrides to `make`
- embeds `RUNPATH=$ORIGIN` into `Dice` so it can find bundled libraries in `bin/`
- verifies the copied binary still exposes that runtime path when `readelf` is available
- copies the final `Dice` binary and required shared libraries into `bin/`

You can confirm the runtime path after build with:

```bash
readelf -d "$REL/native/bin/Dice" | grep -E 'RPATH|RUNPATH'
ldd "$REL/native/bin/Dice" | grep 'not found'
```

## Configure The Block

Point your DICE block configuration at the built executable:

```toml
dice_executable = "/abs/path/to/packages/qcsc-prefect-dice/native/bin/Dice"
```

Or, if you built in shared storage:

```toml
dice_executable = "/work/<group>/share/qcsc-prefect-dice/releases/<release>/native/bin/Dice"
```
