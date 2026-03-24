# GB SQD Native Binaries

This directory contains build scripts for the GB SQD C++ implementation.

## Source Code

The C++ source code is maintained in a separate repository:
- Repository: https://github.com/ibm-quantum-collaboration/gb_demo_2026
- The build scripts will automatically clone this repository to `../gb_demo_2026/`

## Building

### Prerequisites

- C++ compiler with C++17 support
- CMake (>= 3.12)
- MPI implementation (OpenMPI, Intel MPI, or Fujitsu MPI)
- OpenBLAS (when not using Fujitsu compilers)

### Build Instructions

#### Miyabi CPU

```bash
./build_gb_sqd.sh
```

This will:
1. Clone or update the source code from GitHub (with submodules)
2. Initialize and update all Git submodules
3. Create a build directory
4. Run CMake configuration
5. Build the `gb-demo` executable
6. Place the binary in `../gb_demo_2026/build/gb-demo`

#### Miyabi GPU

Run this on a Miyabi-G compute node after entering an interactive GPU session.

```bash
./build_gb_sqd_miyabi_gpu.sh
```

This will:
1. Clone or update the source code from GitHub (with submodules)
2. Initialize and update all Git submodules
3. Configure `gb_demo_2026` with `-DMIYABI_GPU=ON`
4. Build the `gb-demo` executable with `mpicc` / `mpic++`
5. Place the binary in `../gb_demo_2026/build-miyabi-gpu/gb-demo`

#### Fugaku

```bash
./build_gb_sqd_fugaku.sh
```

This will:
1. Clone or update the source code from GitHub (with submodules)
2. Initialize and update all Git submodules
3. Load Fugaku-specific modules
4. Build with Fugaku-specific optimizations using `mpiclang++`
5. Place the binary in `../gb_demo_2026/build/gb-demo`

### Manual Build

If you prefer to build manually:

```bash
# First, clone the repository with submodules if not already done
cd ..
git clone --recurse-submodules https://github.com/ibm-quantum-collaboration/gb_demo_2026.git

# Or if already cloned, initialize submodules
cd gb_demo_2026
git submodule update --init --recursive

# Then build
mkdir -p build && cd build
cmake ..
cmake --build .
```

The executable will be at `gb_demo_2026/build/gb-demo`.

> [!IMPORTANT]
> The repository uses Git submodules. Always use `--recurse-submodules` when cloning, or run `git submodule update --init --recursive` after cloning.

## Usage with Prefect

The Prefect workflows expect the executable to be available. You can:

1. **Build and use default path**:
   - Miyabi CPU: `../gb_demo_2026/build/gb-demo`
   - Miyabi GPU: `../gb_demo_2026/build-miyabi-gpu/gb-demo`
2. **Specify custom path**: Use `--executable` when creating blocks:
   ```bash
   python create_blocks.py \
       --hpc-target miyabi \
        --resource-class cpu \
        --project gz00 \
        --queue regular-c \
        --work-dir ~/work \
        --executable /path/to/your/gb-demo
   ```

For Miyabi GPU blocks:

```bash
python create_blocks.py \
    --hpc-target miyabi \
    --resource-class gpu \
    --project gz00 \
    --queue regular-g \
    --work-dir ~/work \
    --executable /path/to/your/gpu/gb-demo
```

## Prerequisites for Building

Before running the build scripts, ensure you have:

1. **SSH access configured**: Follow the [SSH setup tutorial](../../../docs/tutorials/setup_ssh_keys_for_mdx_and_miyabi.md) to configure SSH keys for GitHub access
2. **Git configured**: Make sure you can clone from the private repository
3. **Required modules loaded** (for HPC systems):
   - Miyabi CPU: `module load intel impi`
   - Miyabi GPU: use a Miyabi-G compute node and ensure `mpicc` / `mpic++` are available
   - Fugaku: `module load LLVM/llvmorg-21.1.0`

## Troubleshooting

### Build fails with "MPI not found"

Make sure MPI modules are loaded:
```bash
# Miyabi
module load intel impi

# Fugaku
module load LLVM/llvmorg-21.1.0
```

### Executable not found when running workflow

Check the HPCProfileBlock's `executable_map`:
```python
from qcsc_prefect_blocks.common.blocks import HPCProfileBlock

block = HPCProfileBlock.load("hpc-miyabi-gb-sqd")
print(block.executable_map)
```

Update if needed:
```bash
python create_blocks.py --executable /correct/path/to/gb-demo ...
