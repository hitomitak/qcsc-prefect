#!/usr/bin/env bash
set -euo pipefail

# Always operate in this script directory to avoid accidental cleanup in cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
SBD_COMMIT="${SBD_COMMIT:-$(cat "$SCRIPT_DIR/SBD_REVISION")}"
SBD_DIR="${SBD_DIR:-${SCRIPT_DIR}/sbd}"
CCCOM="${CCCOM:-mpic++}"
CCFLAGS="${CCFLAGS:--std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST}"
SYSLIB="${SYSLIB:--lblas -llapack}"

# UHF=1 builds the unrestricted GPU driver diag-gpu_uhf with -D_UHF (separate alpha/beta
# integrals, interleaved spin-orbital FCIDUMP; use it with SBDSolverJob(integrals="unrestricted")).
# Without it the restricted driver diag-gpu is built, as before.
OUTBIN="diag-gpu"
if [ "${UHF:-0}" = "1" ]; then
    CCFLAGS="$CCFLAGS -D_UHF"
    OUTBIN="diag-gpu_uhf"
fi

if ! command -v "$CCCOM" >/dev/null 2>&1; then
    echo "Compiler '$CCCOM' not found in PATH. Load the GPU compiler and MPI environment first" \
        "(the NVIDIA HPC SDK on Miyabi-G or ROQUO)." >&2
    exit 1
fi

# The default flags are NVIDIA HPC SDK flags, so the MPI wrapper must call nvc++. On Miyabi-G it
# does. An Open MPI wrapper that calls another compiler (on ROQUO, hpcx's mpic++ calls g++) is
# pointed at nvc++ when nvc++ is on PATH; a wrapper that still calls another compiler stops here.
if [[ "$CCFLAGS" == *"-gpu="* || "$CCFLAGS" == *"-cuda"* ]]; then
    if command -v nvc++ >/dev/null 2>&1; then
        export OMPI_CXX="${OMPI_CXX:-nvc++}"
        export OMPI_CC="${OMPI_CC:-nvc}"
    fi
    backend="$("$CCCOM" -show 2>/dev/null | awk '{print $1}')" || backend=""
    if [[ -n "$backend" && "$(basename "$backend")" != nvc++* ]]; then
        echo "'$CCCOM' calls '$backend', but CCFLAGS are NVIDIA HPC SDK flags, which need nvc++." \
            "Load the NVIDIA HPC SDK so that nvc++ is on PATH." >&2
        exit 1
    fi
fi

if [ ! -d "$SBD_DIR" ]; then
    echo "Cloning SBD repo..."
    git clone "$REPO_URL" "$SBD_DIR"
else
    echo "SBD repo already exists: $SBD_DIR"
fi

# Use the revision matching main.cc on every platform, including existing clones.
if ! git -C "$SBD_DIR" cat-file -e "${SBD_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$SBD_DIR" fetch origin "$SBD_COMMIT"
fi
git -C "$SBD_DIR" checkout --detach "$SBD_COMMIT"
echo "Using SBD commit: $(git -C "$SBD_DIR" rev-parse HEAD)"

# Clean previous build
rm -f "$SCRIPT_DIR"/*.o "$SCRIPT_DIR/$OUTBIN"

# Compile and link
echo "$CCCOM $CCFLAGS -c main.cc -o main.o -I$SBD_DIR/include"
$CCCOM $CCFLAGS -c main.cc -o main.o -I"$SBD_DIR/include"
echo "$CCCOM $CCFLAGS $SYSLIB -o $OUTBIN main.o"
$CCCOM $CCFLAGS $SYSLIB -o "$OUTBIN" main.o

echo "Build completed: $SCRIPT_DIR/$OUTBIN"
