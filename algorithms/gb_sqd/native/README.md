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

#### Local/Miyabi

```bash
./build_gb_sqd.sh
```

This will:
1. Clone or update the source code from GitHub
2. Create a build directory
3. Run CMake configuration
4. Build the `gb-demo` executable
5. Place the binary in `../gb_demo_2026/build/gb-demo`

#### Fugaku

```bash
./build_gb_sqd_fugaku.sh
```

This will:
1. Clone or update the source code from GitHub
2. Load Fugaku-specific modules
3. Build with Fugaku-specific optimizations using `mpiclang++`
4. Place the binary in `../gb_demo_2026/build/gb-demo`

### Manual Build

If you prefer to build manually:

```bash
# First, clone the repository if not already done
cd ..
git clone https://github.com/ibm-quantum-collaboration/gb_demo_2026.git

# Then build
cd gb_demo_2026
mkdir -p build && cd build
cmake ..
cmake --build .
```

The executable will be at `gb_demo_2026/build/gb-demo`.

## Usage with Prefect

The Prefect workflows expect the executable to be available. You can:

1. **Build and use default path**: The build scripts place the executable in `../gb_demo_2026/build/gb-demo`
2. **Specify custom path**: Use `--executable` when creating blocks:
   ```bash
   python create_blocks.py \
       --hpc-target miyabi \
       --project gz00 \
       --queue regular-c \
       --work-dir ~/work \
       --executable /path/to/your/gb-demo
   ```

## Prerequisites for Building

Before running the build scripts, ensure you have:

1. **SSH access configured**: Follow the [SSH setup tutorial](../../../docs/tutorials/setup_ssh_keys_for_mdx_and_miyabi.md) to configure SSH keys for GitHub access
2. **Git configured**: Make sure you can clone from the private repository
3. **Required modules loaded** (for HPC systems):
   - Miyabi: `module load intel impi`
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
from hpc_prefect_blocks.common.blocks import HPCProfileBlock

block = HPCProfileBlock.load("hpc-miyabi-gb-sqd")
print(block.executable_map)
```

Update if needed:
```bash
python create_blocks.py --executable /correct/path/to/gb-demo ...