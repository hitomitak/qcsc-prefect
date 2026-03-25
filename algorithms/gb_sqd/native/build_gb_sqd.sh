#!/bin/bash
# Build script for GB SQD (Miyabi/Local)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GB_SQD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_URL="git@github.com:ibm-quantum-collaboration/gb_demo_2026.git"
SOURCE_DIR="$GB_SQD_DIR/gb_demo_2026"

echo "=========================================="
echo "Building GB SQD for Miyabi/Local"
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

# Create build directory
BUILD_DIR="$SOURCE_DIR/build"
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

echo ""
echo "Running CMake configuration..."

# Detect if running on Miyabi
if command -v mpiicpc &> /dev/null; then
    echo "Detected Miyabi environment (mpiicpc found)"
    echo "Configuring with MIYABI=ON..."
    cmake .. -DMIYABI=ON -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc
else
    echo "Configuring for local environment..."
    cmake ..
fi

echo ""
echo "Building..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Executable: $BUILD_DIR/gb-demo"
echo ""
echo "To use with Prefect, create blocks with:"
echo "  python create_blocks.py \\"
echo "    --hpc-target miyabi \\"
echo "    --project YOUR_PROJECT \\"
echo "    --queue regular-c \\"
echo "    --work-dir ~/work/gb_sqd \\"
echo "    --executable $BUILD_DIR/gb-demo"

