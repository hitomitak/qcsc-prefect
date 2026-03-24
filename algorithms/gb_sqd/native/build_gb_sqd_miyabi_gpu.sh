#!/bin/bash
# Build script for GB SQD (Miyabi GPU)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GB_SQD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_URL="git@github.com:ibm-quantum-collaboration/gb_demo_2026.git"
SOURCE_DIR="$GB_SQD_DIR/gb_demo_2026"
BUILD_DIR="$SOURCE_DIR/build-miyabi-gpu"

echo "=========================================="
echo "Building GB SQD for Miyabi GPU"
echo "=========================================="
echo "GB SQD directory: $GB_SQD_DIR"
echo ""

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Cloning source code from GitHub..."
    git clone --recurse-submodules "$REPO_URL" "$SOURCE_DIR"
else
    echo "Source directory exists. Updating..."
    cd "$SOURCE_DIR"
    git pull
    git submodule update --init --recursive
    cd "$GB_SQD_DIR"
fi

echo ""
echo "Source code location: $SOURCE_DIR"
echo "Initializing submodules..."
cd "$SOURCE_DIR"
git submodule update --init --recursive

if [ ! -f "$SOURCE_DIR/CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found in $SOURCE_DIR"
    echo "The repository structure may have changed."
    exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "Running CMake configuration for Miyabi GPU..."

if ! command -v mpic++ >/dev/null 2>&1; then
    echo "Error: mpic++ not found in PATH."
    echo "Run this script on a Miyabi-G compute node with the GPU MPI toolchain available."
    exit 1
fi

cmake .. -DMIYABI_GPU=ON -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpic++

echo ""
echo "Building..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Executable: $BUILD_DIR/gb-demo"
echo ""
echo "To use with Prefect, create GPU blocks with:"
echo "  python create_blocks.py \\"
echo "    --hpc-target miyabi \\"
echo "    --resource-class gpu \\"
echo "    --project YOUR_PROJECT \\"
echo "    --queue regular-g \\"
echo "    --work-dir ~/work/gb_sqd \\"
echo "    --executable $BUILD_DIR/gb-demo"
