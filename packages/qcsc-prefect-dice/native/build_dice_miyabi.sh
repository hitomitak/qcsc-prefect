#!/bin/bash
set -euo pipefail

# Load modules (Intel compiler, MPI, Boost with MPI, and Parallel HDF5)
module load intel/2023.2.0
module load impi/2021.10.0
module load phdf5/1.14.3

# Start with a fresh external dir
rm -rf external/ && mkdir external

# Get Boost
wget https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz \
    && tar -xzf boost_1_85_0.tar.gz -C "$(pwd)"/external \
    && rm boost_1_85_0.tar.gz

# Clone Dice default branch
git clone https://github.com/caleb-johnson/Dice.git "$(pwd)"/external/Dice

export BOOST_ROOT="$(pwd)"/external/boost_1_85_0
export CURC_HDF5_ROOT="$HDF5_DIR"
DICE_ROOT="$(pwd)"/external/Dice
DICE_BIN_PATH="$(pwd)"/bin

# Build boost
cd "$BOOST_ROOT"
./bootstrap.sh --with-libraries=serialization,mpi
echo 'using intel-linux : icpx : icpx ;' > user-config.jam
echo 'using mpi : mpiicpc ;' >> user-config.jam
./b2 toolset=intel-linux \
    cxxflags="-axSAPPHIRERAPIDS,CORE-AVX512 -diag-disable=10430" \
    --user-config=user-config.jam \
    -j"$(nproc)"

# Build DICE
cd "$DICE_ROOT"
dice_make_args=(
    'CXX=mpiicpc -cxx=icpx'
    'CXXFLAGS=-I. -I$(HDF5)/include -I$(EIGEN) -I$(BOOST) -I$(ZLIB) -axSAPPHIRERAPIDS,CORE-AVX512 -diag-disable=10430 -g -Wall -march=native -Wno-sign-compare -Werror -O3 -funroll-loops -std=c++0x -fopenmp -DUSE_HDF5_SERIAL -Dserialize_hash'
    'LIBS=-L. -L$(BOOST)/stage/lib -L$(MKLROOT)/lib/intel64 -L$(HDF5)/lib -Wl,--enable-new-dtags -Wl,-rpath,\$$ORIGIN -Wl,-rpath,$(MKLROOT)/lib/intel64 -Wl,-rpath,$(HDF5)/lib -mkl=sequential -fopenmp -lhdf5_cpp -lhdf5 -lboost_mpi -lboost_serialization'
)
make -j"$(nproc)" "${dice_make_args[@]}" Dice

# Put the runtime libraries in the package
mkdir -p "$DICE_BIN_PATH"
cp "$DICE_ROOT"/bin/Dice "$DICE_BIN_PATH"
cp "$BOOST_ROOT"/stage/lib/*.so* "$DICE_BIN_PATH"

if command -v readelf >/dev/null 2>&1; then
    if ! readelf -d "$DICE_BIN_PATH"/Dice | grep -Eq '\((RPATH|RUNPATH)\).*\$ORIGIN'; then
        echo "Dice binary is missing \$ORIGIN RUNPATH; bundled libraries may not be found at runtime." >&2
        exit 1
    fi
fi
