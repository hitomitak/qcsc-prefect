#!/usr/bin/env bash
set -euo pipefail

# Always operate in this script directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SBD_DIR="${SBD_DIR:-sbd}"
REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
SBD_COMMIT="${SBD_COMMIT:-$(cat "$SCRIPT_DIR/SBD_REVISION")}"
CCCOM="${CCCOM:-mpiFCCpx}"
CCFLAGS="${CCFLAGS:--Nclang -std=c++17 -stdlib=libc++ -Kfast,openmp -Xpreprocessor -fopenmp}"
SYSLIB="${SYSLIB:--SSL2}"
# UHF=1 builds the unrestricted driver diag_uhf with -D_UHF (separate alpha/beta integrals,
# interleaved spin-orbital FCIDUMP; use it with SBDSolverJob(integrals="unrestricted")).
# Without it the restricted driver diag is built, as before.
OUTBIN="diag"
if [ "${UHF:-0}" = "1" ]; then
    CCFLAGS="$CCFLAGS -D_UHF"
    OUTBIN="diag_uhf"
fi

if ! command -v "$CCCOM" >/dev/null 2>&1; then
    echo "Compiler '$CCCOM' not found in PATH. Load Fugaku compiler/MPI modules first." >&2
    exit 1
fi

if [ ! -d "$SBD_DIR" ]; then
    echo "Cloning SBD repo..."
    git clone "$REPO_URL" "$SBD_DIR"
else
    echo "SBD repo already exists."
fi

# Use the revision matching main.cc on every platform, including existing clones.
if ! git -C "$SBD_DIR" cat-file -e "${SBD_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$SBD_DIR" fetch origin "$SBD_COMMIT"
fi
git -C "$SBD_DIR" checkout --detach "$SBD_COMMIT"
echo "Using SBD commit: $(git -C "$SBD_DIR" rev-parse HEAD)"

# Compile and link in a directory of this build only: builds of the restricted and the UHF
# driver can run at the same time, and a shared main.o would let one link the other's object.
rm -f "$SCRIPT_DIR/$OUTBIN"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sbd-build.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT
echo "$CCCOM $CCFLAGS -c main.cc -o $BUILD_DIR/main.o -I$SBD_DIR/include"
$CCCOM $CCFLAGS -c main.cc -o "$BUILD_DIR/main.o" -I"$SBD_DIR/include"
echo "$CCCOM $CCFLAGS $SYSLIB -o $BUILD_DIR/$OUTBIN $BUILD_DIR/main.o"
$CCCOM $CCFLAGS $SYSLIB -o "$BUILD_DIR/$OUTBIN" "$BUILD_DIR/main.o"
mv -f "$BUILD_DIR/$OUTBIN" "$SCRIPT_DIR/$OUTBIN"

echo "Build completed: $SCRIPT_DIR/$OUTBIN"
