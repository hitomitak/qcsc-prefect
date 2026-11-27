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

# MPI=1 builds the driver for several GPUs per solve, one MPI rank per GPU: <name>-mpi
# (diag-gpu-mpi, or diag-gpu_uhf-mpi with UHF=1). It adds NCCL collectives, host-staged MPI
# (for an MPI that is not CUDA-aware, such as hpcx on ROQUO) and SBD's rank distributions, the
# defines of the UHF-SQD source's build (kunalroy332/qcsc-prefect 3968e58, there named
# diag-gpu_uhf-mpi-hpcx250). NCCL is taken from the NVIDIA HPC SDK. -DSBD_USE_CUBLAS is left out,
# as in the source: Davidson then sums across ranks with NCCL without cuBLAS.
if [ "${MPI:-0}" = "1" ]; then
    command -v nvc++ >/dev/null 2>&1 || {
        echo "MPI=1 needs nvc++ on PATH: load the NVIDIA HPC SDK first." >&2; exit 1; }
    NV="${NVHPC_ROOT:-$(dirname "$(dirname "$(readlink -f "$(command -v nvc++)")")")}"
    NV="${NV%/compilers}"
    find_one() { find "$NV" -name "$1" -path "$2" 2>/dev/null | head -1 || true; }
    NCCL_H="${NCCL_INCLUDE:-$(find_one nccl.h '*nccl*')}"
    NCCL_SO="${NCCL_LIB:-$(find_one libnccl.so '*comm_libs*')}"
    if [[ -z "$NCCL_H" || -z "$NCCL_SO" ]]; then
        echo "MPI=1: NCCL not found under $NV (nccl.h: '${NCCL_H}', libnccl.so: '${NCCL_SO}')." \
            "Set NCCL_INCLUDE and NCCL_LIB to the two files." >&2
        exit 1
    fi
    NCCL_INC_DIR="$(dirname "$NCCL_H")"
    NCCL_LIB_DIR="$(dirname "$NCCL_SO")"
    echo "NCCL: $NCCL_INC_DIR, $NCCL_LIB_DIR"
    CCFLAGS="$CCFLAGS -DSBD_USE_NCCL -DSBD_NON_CUDA_AWARE_MPI -DSBD_THRUST_SAFE_MPI_ALLREDUCE"
    CCFLAGS="$CCFLAGS -DSBD_USE_RANK_DISTRIBUTION -DSBD_USE_BLOCK_RANK_DISTRIBUTION"
    CCFLAGS="$CCFLAGS -I$NCCL_INC_DIR"
    SYSLIB="$SYSLIB -L$NCCL_LIB_DIR -Wl,-rpath,$NCCL_LIB_DIR -lnccl"
    OUTBIN="${OUTBIN}-mpi"
fi

if ! command -v "$CCCOM" >/dev/null 2>&1; then
    echo "Compiler '$CCCOM' not found in PATH. Load the GPU compiler and MPI environment first" \
        "(the NVIDIA HPC SDK on Miyabi-G or ROQUO)." >&2
    exit 1
fi

# The default flags are NVIDIA HPC SDK flags, so the MPI wrapper must call nvc++. On Miyabi-G it
# does. An Open MPI wrapper that calls another compiler (on ROQUO, hpcx's mpic++ calls g++) is
# pointed at nvc++ when nvc++ is on PATH; a wrapper that still calls another compiler stops here.
backend=""
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

# nvc++ warns about unused variables in the SBD headers, and recommends nvcc for the <<<>>>
# kernels that nvc++ -cuda builds. Neither affects the build; WARNFLAGS="" shows them again.
if [[ -z "${WARNFLAGS+x}" ]]; then
    WARNFLAGS=""
    if [[ "$(basename "$backend")" == nvc++* ]]; then
        WARNFLAGS="--diag_suppress declared_but_not_referenced,set_but_not_used,cuda_compile"
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

# The multi-GPU driver needs SBD's NCCL helpers and two fixes (r-ccs-cms/sbd PR79, a deadlock
# with a one-rank h_comm, and PR84, an OpenMP livelock); the pinned revision has all three.
if [ "${MPI:-0}" = "1" ]; then
    inc="$SBD_DIR/include/sbd"
    missing=""
    [[ -f "$inc/framework/nccl_utility.h" ]] || missing+=" framework/nccl_utility.h;"
    grep -q "h_size == 1" "$inc/chemistry/basic/davidson_thrust.h" 2>/dev/null \
        || missing+=" PR79 (davidson_thrust.h);"
    grep -q "cr.reserve(2)" "$inc/chemistry/tpb/helper.h" 2>/dev/null \
        || missing+=" PR84 (tpb/helper.h);"
    if [[ -n "$missing" ]]; then
        echo "MPI=1: SBD $(git -C "$SBD_DIR" rev-parse --short HEAD) lacks:$missing" >&2
        exit 1
    fi
fi

# Compile and link in a directory of this build only: builds of the restricted and the UHF
# driver can run at the same time, and a shared main.o would let one link the other's object.
rm -f "$SCRIPT_DIR/$OUTBIN"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sbd-build.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT
echo "$CCCOM $CCFLAGS $WARNFLAGS -c main.cc -o $BUILD_DIR/main.o -I$SBD_DIR/include"
$CCCOM $CCFLAGS $WARNFLAGS -c main.cc -o "$BUILD_DIR/main.o" -I"$SBD_DIR/include"
echo "$CCCOM $CCFLAGS $SYSLIB -o $BUILD_DIR/$OUTBIN $BUILD_DIR/main.o"
$CCCOM $CCFLAGS $SYSLIB -o "$BUILD_DIR/$OUTBIN" "$BUILD_DIR/main.o"
# A multi-GPU driver that cannot find NCCL or MPI at run time fails only on the GPU node.
if [ "${MPI:-0}" = "1" ] && command -v ldd >/dev/null 2>&1; then
    if ldd "$BUILD_DIR/$OUTBIN" | grep "not found" >&2; then
        echo "Build FAILED: $OUTBIN has unresolved libraries (above)" >&2
        exit 1
    fi
fi
mv -f "$BUILD_DIR/$OUTBIN" "$SCRIPT_DIR/$OUTBIN"

echo "Build completed: $SCRIPT_DIR/$OUTBIN"
