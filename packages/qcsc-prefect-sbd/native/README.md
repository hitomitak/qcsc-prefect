# Custom Program for Selected Basis Diagonalization

This submodule provides a custom C++ implementation for configuration recovery,
adapted from the [selected basis diagonalization sample](https://github.com/r-ccs-cms/sbd/tree/95bf45a5be9ffd977fac7b0aaa359833caf3401a/apps/chemistry_tpb_selected_basis_diagonalization) in the SBD project.
Refer to the upstream repository for detailed information about the original software.

## SBD API version

The driver uses upstream SBD's `det_vector<size_t, det_kind::half>` API for input and
carryover determinants. All four build scripts use the revision in `SBD_REVISION`,
currently `95bf45a5be9ffd977fac7b0aaa359833caf3401a` (upstream `main` as of 2026-11-04).
The revision is shared so that existing checkouts and all target platforms build
against the same API. Older SBD versions using nested `std::vector` determinants are
not compatible with this driver.

The scripts fetch the revision if it is missing and check it out without discarding
local changes. `SBD_COMMIT` can override the revision when testing a compatible SBD
update. Record the printed commit with your test results. Updating the default means
updating `SBD_REVISION` and checking the native solver tests before using the new API.

The Python-facing input/output files, including determinant byte order and carryover
padding, are unchanged. The native compatibility tests cover MPI broadcast, energy,
occupancies and the carryover round trip with a small analytic Hamiltonian.

The Python wrapper passes `--carryover_type` from the `SBDSolverJob.carryover_type`
field, 1 by default; upstream defaults to type 0, which produces no carryover even when
`--carryover_ratio` is set. User arguments are appended afterward and can override it.

The driver reads `fcidump.txt` and `AlphaDets.bin` unless `--fcidump` or `--adetfile`
name other files. With `--bdetfile` it reads a separate beta determinant set (open shell);
without it the beta set is the alpha set. GPU builds (`-DSBD_THRUST` with a CUDA
compiler) bind each rank to GPU `node-local rank % GPU count`, from `SLURM_LOCALID` or
`OMPI_COMM_WORLD_LOCAL_RANK`, before `MPI_Init`.

### Unrestricted (UHF) driver and the integral layout check

`UHF=1` with a CPU build script (`build_sbd.sh`, `build_sbd_fugaku.sh`, `build_sbd_slurm.sh`)
compiles `main.cc` with SBD's `-D_UHF` and writes `diag_uhf` next to `diag`. With
`build_sbd_gpu.sh` it writes `diag-gpu_uhf` next to `diag-gpu`; what this section says of
`diag_uhf` holds for it too. That build reads
an interleaved spin-orbital FCIDUMP (alpha of spatial orbital p is index 2p+1, beta 2p+2) with
separate alpha and beta integrals; the default build reads the restricted FCIDUMP.

The two layouts cannot be told apart from the file contents alone (zero integrals are omitted,
so even the largest index does not identify the layout), and a `-D_UHF` driver given a
restricted file finishes with a wrong Hamiltonian. The driver therefore checks the header key
`QCSC_INTEGRALS` against its build before any work and aborts with `[qcsc] ERROR: ...` on a
mismatch:

| FCIDUMP | `diag` (restricted) | `diag_uhf` (`-D_UHF`) |
| --- | --- | --- |
| no `QCSC_INTEGRALS` (all FCIDUMPs written before this check) | runs | aborts |
| `QCSC_INTEGRALS=RESTRICTED` | runs | aborts |
| `QCSC_INTEGRALS=SBD_UHF_INTERLEAVED` (`SBDSolverJob` unrestricted) | aborts | runs |
| `IUHF=0` (MOLPRO's restricted flag) | as without `IUHF` | as without `IUHF` |
| `IUHF` other than `0` (MOLPRO-style block UHF, not an SBD layout) | aborts | aborts |

An index outside the layout's range also aborts. On success the driver writes
`sbd_build_info.txt` (`integrals=RESTRICTED` or `integrals=SBD_UHF_INTERLEAVED`), which
`SBDSolverJob` checks after the job.

Drivers built before this check write no `sbd_build_info.txt`, and none of the aborts above
apply to them. `SBDSolverJob` still accepts them for restricted input, with a warning and
`SBDResult.integrals=None`, but cannot confirm the energy: a `-D_UHF` binary built before the
check and registered as the restricted executable completes on a restricted FCIDUMP with a
different Hamiltonian. Rebuild both drivers with the scripts here, and point the HPC profiles
at the rebuilt binaries; see "Drivers built before the layout check" in
[the package README](../README.md).

After building the CPU executable, run them with:

```bash
SBD_NATIVE_BINARY="$(realpath ./diag)" python -m pytest ../tests/test_sbd_native.py -q
SBD_NATIVE_BINARY="$(realpath ./diag)" SBD_NATIVE_UHF_BINARY="$(realpath ./diag_uhf)" \
  python -m pytest ../tests/test_sbd_native_uhf.py -q
```

`SBD_NATIVE_LAUNCHER` (default `mpiexec`) starts the driver with `-n <ranks>`, and
`SBD_NATIVE_OMP_THREADS` (default 1) sets its OpenMP threads per rank. For the GPU drivers, see
[Running the native tests on GPUs](#running-the-native-tests-on-gpus).

`test_sbd_native_uhf.py` compares the UHF driver's full-space energies for an open-shell and a
broken-symmetry molecule with PySCF FCI, and checks the layout table above.

The tests launch one and two MPI ranks locally, without a scheduler or QPU. CI builds
the CPU driver with Open MPI and OpenBLAS and runs the same tests on Ubuntu. Fugaku's
compiler and scheduler execution still need to be checked on Fugaku.

## Output Files

The modified program generates the following output files:
- `davidson_energy.txt`: Final energy computed by the Davidson solver
- `occ_a.txt`: Average occupancies of alpha orbitals
- `occ_b.txt`: Average occupancies of beta orbitals
- `carryover.bin`: Carryover bitstrings in `uint8` format with big-endian ordering
- `carryover_b.bin`: Beta carryover bitstrings in the same format, only with `--bdetfile`
- `sbd_build_info.txt`: The integral layout of the build (`integrals=RESTRICTED` or `integrals=SBD_UHF_INTERLEAVED`)

These files are consumed by the Prefect integration block to construct a corresponding Python data class.

## Building the Executable on a Slurm Cluster

For a generic Slurm cluster with an MPI C++ wrapper (`mpicxx` or `mpic++`) and OpenBLAS,
run:

```bash
bash ./build_sbd_slurm.sh
```

It produces `packages/qcsc-prefect-sbd/native/diag`. Launch it with `srun --mpi=pmix`
(match the PMI plugin of your Slurm and MPI), as the `slurm` profile of the open-loop SQD
workflow does with 4 ranks and 1 thread each.

## Building the Executable on Miyabi

To build the executable, run the provided build script:

```bash
bash ./build_sbd.sh
```

> [!NOTE]
> This build script is adapted for the Miyabi-C environment.
> Build in other environments may require modification of the compiler command and flags.

## Building the GPU Executable

For Miyabi-G, you can build `diag-gpu` with the dedicated shell script:

```bash
./build_sbd_gpu.sh
```

`UHF=1 ./build_sbd_gpu.sh` builds the unrestricted `diag-gpu_uhf` instead (see
[Unrestricted (UHF) driver](#unrestricted-uhf-driver-and-the-integral-layout-check)).

This script compiles `main.cc` directly, so a separate `CMakeLists.txt` is not required.
It defaults to a local checkout under:

```text
packages/qcsc-prefect-sbd/native/sbd
```

and produces:

```text
packages/qcsc-prefect-sbd/native/diag-gpu
```

> [!NOTE]
> The default compiler is `mpic++`, assuming it is backed by the NVIDIA HPC SDK (`nvc++`). An
> Open MPI wrapper that calls another compiler is pointed at `nvc++` (`OMPI_CXX=nvc++`) when
> `nvc++` is on `PATH`; a wrapper that still calls another compiler stops the build.
> If the local `sbd` checkout does not exist yet, the script clones `https://github.com/r-ccs-cms/sbd.git`.
> If your site uses a different wrapper, source tree, or BLAS/LAPACK link flags, override them via `CCCOM`, `CCFLAGS`, `SYSLIB`, `SBD_DIR`, or `SBD_REPO_URL`.

### ROQUO

ROQUO's compute nodes are GB200 nodes with aarch64 Grace CPUs; the login node is x86_64. Build on
a compute node, so that the drivers are built for that architecture. The build needs the NVIDIA
HPC SDK, not a GPU:

```bash
srun --account=<account> --partition=roquo --gpus-per-node=1 --time=01:00:00 --pty bash
module load cuda/13.2 nvhpc/26.5 && module load hpcx/2.50
export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5/compilers/bin:$PATH
cd packages/qcsc-prefect-sbd/native
bash build_sbd_gpu.sh            # diag-gpu
UHF=1 bash build_sbd_gpu.sh      # diag-gpu_uhf
```

`hpcx/2.50`'s `mpic++` calls `g++` by default; with `nvc++` on `PATH` the script points it at
`nvc++`. These are the steps of the UHF-SQD source workflow, which built `diag-gpu_uhf` this way
on ROQUO. They are not yet checked with this repository.

### Running the native tests on GPUs

The native tests run the drivers under `SBD_NATIVE_LAUNCHER`, so the Python side can stay on a
login node while the drivers run on a GPU node. GPU builds stop at startup with one OpenMP
thread, so set `SBD_NATIVE_OMP_THREADS`. On ROQUO, from the login node:

```bash
salloc --account=<account> --partition=roquo --gpus-per-node=1 --time=00:30:00
SBD_NATIVE_LAUNCHER="srun --gpus-per-node=1" SBD_NATIVE_OMP_THREADS=8 \
SBD_NATIVE_BINARY="$(realpath diag-gpu)" SBD_NATIVE_UHF_BINARY="$(realpath diag-gpu_uhf)" \
  uv run --no-sync python -m pytest ../tests/test_sbd_native.py ../tests/test_sbd_native_uhf.py \
  -q -k "not -2] and not [2-"
exit
```

`-k` leaves out the two-rank cases: with one GPU, two ranks share it, and `diag-gpu` passes GPU
buffers to MPI, which needs a CUDA-aware MPI.

## Building the Executable on Fugaku

For Fugaku, use the dedicated script:

```bash
bash ./build_sbd_fugaku.sh
```

> [!NOTE]
> Ensure a Fugaku MPI C++ compiler is available in `PATH` (for example `mpiFCCpx`).
> The script defaults to `mpiFCCpx` with Fugaku-oriented flags and `-SSL2`.
> You can override `CCCOM`, `CCFLAGS`, and `SYSLIB` via environment variables if your site requires different settings.

Upon successful compilation, an executable named `diag` will be created in this directory:

```text
packages/qcsc-prefect-sbd/native/diag
```

`build_sbd.sh` and `build_sbd_slurm.sh` write the same path. Set the absolute path to this
executable in the SBD block configuration file:

```toml
# algorithms/open_loop/sqd/sqd_blocks.toml, in its [sbd] table
# (or algorithms/closed_loop/sqd_de/sbd_blocks.toml, or the block config of another algorithm)
sbd_executable = "/abs/path/to/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag"
```

### Optional environment overrides

You can override upstream repository location:

```bash
SBD_REPO_URL="https://github.com/r-ccs-cms/sbd.git" \
SBD_DIR="/path/to/local/sbd" \
bash ./build_sbd.sh
```

For Fugaku build settings:

```bash
CCCOM="mpiFCCpx" \
CCFLAGS="-Nclang -std=c++17 -stdlib=libc++ -Kfast,openmp -Xpreprocessor -fopenmp" \
SYSLIB="-SSL2" \
bash ./build_sbd_fugaku.sh
```

For Miyabi-G build settings:

```bash
SBD_REPO_URL="https://github.com/r-ccs-cms/sbd.git" \
SBD_DIR="/path/to/local/sbd" \
CCCOM="mpic++" \
CCFLAGS="-std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST" \
SYSLIB="-lblas -llapack" \
./build_sbd_gpu.sh
```
