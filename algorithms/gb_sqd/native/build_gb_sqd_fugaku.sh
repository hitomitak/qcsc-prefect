#!/bin/bash
# Build script for GB SQD (Fugaku)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GB_SQD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_URL="https://github.com/ibm-quantum-collaboration/gb_demo_2026.git"
SOURCE_DIR="$GB_SQD_DIR/gb_demo_2026"

echo "=========================================="
echo "Building GB SQD for Fugaku"
echo "=========================================="
echo "GB SQD directory: $GB_SQD_DIR"
echo ""

# Clone or update the source repository
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
cd "$GB_SQD_DIR"

# Check if CMakeLists.txt exists
if [ ! -f "$SOURCE_DIR/CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found in $SOURCE_DIR"
    echo "The repository structure may have changed."
    exit 1
fi

# Load required modules
echo ""
echo "Loading Fugaku modules..."
module load LLVM/llvmorg-21.1.0 || {
    echo "Warning: Could not load LLVM module. Make sure you're on Fugaku."
}

# Create build directory
BUILD_DIR="$SOURCE_DIR/build"
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

echo ""
echo "Running CMake configuration for Fugaku..."
cmake .. \
    -DFUGAKU=ON \
    -DCMAKE_C_COMPILER=mpiclang \
    -DCMAKE_CXX_COMPILER=mpiclang++

echo ""
echo "Building..."
cmake --build . -j48

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Executable: $BUILD_DIR/gb-demo"
echo ""
echo "To use with Prefect, create blocks with:"
echo "  python create_blocks.py \\"
echo "    --hpc-target fugaku \\"
echo "    --project YOUR_PROJECT \\"
echo "    --queue large \\"
echo "    --work-dir /work/YOUR_GROUP/YOUR_USER/gb_sqd \\"
echo "    --executable $BUILD_DIR/gb-demo \\"
echo "    --num-nodes 1008 \\"
echo "    --mpiprocs 1 \\"
echo "    --ompthreads 48"

