#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <deque>
#include <cstdlib>

#include <unistd.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include "sbd/sbd.h"
#include "mpi.h"

#if defined(SBD_THRUST) && defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cstdlib>
#endif


int main(int argc, char * argv[]) {

#if defined(SBD_THRUST) && defined(__CUDACC__)
  // One GPU per MPI rank on a node: the launcher starts ranks that all see every GPU of the
  // node, so pick device (node-local rank) % (#GPUs) before MPI_Init, so that MPI / NCCL set up
  // their CUDA context on that device. SLURM_LOCALID (srun) or OMPI_COMM_WORLD_LOCAL_RANK
  // (mpirun) is the node-local rank; without either (a single rank) this is GPU 0.
  {
    const char* local = std::getenv("SLURM_LOCALID");
    if( local == nullptr ) local = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int num_devices = 0;
    if( cudaGetDeviceCount(&num_devices) == cudaSuccess && num_devices > 0 ) {
      cudaSetDevice((local != nullptr ? std::atoi(local) : 0) % num_devices);
    }
  }
#endif

  int provided;
  int mpi_ierr = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  MPI_Comm comm = MPI_COMM_WORLD;
  int mpi_master = 0;
  int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
  int mpi_size; MPI_Comm_size(comm,&mpi_size);

#if defined(SBD_THRUST) && defined(__CUDACC__)
  if( mpi_size > 1 ) {
    int device = -1;
    cudaGetDevice(&device);
    std::cout << "[gpu] rank " << mpi_rank << "/" << mpi_size << " uses CUDA device " << device
              << std::endl;
  }
#endif

  auto sbd_data = sbd::tpb::generate_sbd_data(argc,argv);

  std::string adetfile("AlphaDets.bin");
  std::string bdetfile("");
  std::string fcidumpfile("fcidump.txt");
  std::string loadname("");
  std::string savename("");

  // Optional file overrides, matching the upstream
  // apps/chemistry_tpb_selected_basis_diagonalization reference:
  //   --fcidump   path to the FCIDUMP (default fcidump.txt); its header key QCSC_INTEGRALS
  //               must match the build (RESTRICTED, the default when absent, or
  //               SBD_UHF_INTERLEAVED for -D_UHF), see the layout check below
  //   --adetfile  alpha determinant file (.bin packed or .txt bitstrings; LoadAlphaDets
  //               branches on the extension)
  //   --bdetfile  beta determinant file, for a beta set that differs from alpha (open shell).
  //               Without it the beta set is the alpha set.
  for(int i = 0; i < argc; i++) {
    if( std::string(argv[i]) == "--fcidump" && i + 1 < argc ) {
      fcidumpfile = argv[i + 1];
    }
    if( std::string(argv[i]) == "--adetfile" && i + 1 < argc ) {
      adetfile = argv[i + 1];
    }
    if( std::string(argv[i]) == "--bdetfile" && i + 1 < argc ) {
      bdetfile = argv[i + 1];
    }
  }

  int L;
  int N;
  double energy;
  std::vector<double> density;
  sbd::det_vector<size_t, sbd::det_kind::half> co_adet;
  sbd::det_vector<size_t, sbd::det_kind::half> co_bdet;
  std::vector<std::vector<double>> one_p_rdm;
  std::vector<std::vector<double>> two_p_rdm;
  sbd::FCIDump fcidump;

  std::cout.precision(16);

  /**
     load fcidump data
   */
  if( mpi_rank == 0 ) {
    fcidump = sbd::LoadFCIDump(fcidumpfile);
  }
  sbd::MpiBcast(fcidump,0,comm);

  /**
     Integral layout check. -D_UHF is a compile-time switch of SBD: such a build reads an
     interleaved spin-orbital FCIDUMP, the default build a restricted one, and neither file is
     self-describing (a -D_UHF build runs a restricted file to completion with a wrong
     Hamiltonian). The FCIDUMP declares its layout with the header key QCSC_INTEGRALS; a file
     without it is a restricted file, so FCIDUMPs written before this check still work with
     restricted builds. A mismatch, or a MOLPRO-style block UHF file (IUHF other than 0),
     aborts before any work. The index range check is secondary: zero integrals are omitted
     from FCIDUMPs, so the largest index cannot identify the layout by itself. The layout is
     reported in sbd_build_info.txt.
   */
  {
#if defined(_UHF)
    const std::string build_layout("SBD_UHF_INTERLEAVED");
#elif defined(_UHFCPLX)
    const std::string build_layout("SBD_UHFCPLX_INTERLEAVED");
#else
    const std::string build_layout("RESTRICTED");
#endif
    std::string file_layout("RESTRICTED");
    bool has_marker = false;
    int norb_header = -1;
    for(const auto & [key,value] : fcidump.header) {
      if( key == std::string("QCSC_INTEGRALS") ) {
        file_layout = value;
        has_marker = true;
      }
      if( key == std::string("NORB") ) {
        norb_header = std::atoi(value.c_str());
      }
    }
    // IUHF=0 is MOLPRO's restricted flag and changes nothing; any other value means
    // MOLPRO-style block UHF, which no SBD build reads.
    bool molpro_uhf = false;
    std::string iuhf_value;
    if( fcidump.header.count("IUHF") ) {
      iuhf_value = fcidump.header.at("IUHF");
      char * end = nullptr;
      const long iuhf = std::strtol(iuhf_value.c_str(), &end, 10);
      molpro_uhf = iuhf_value.empty() || *end != '\0' || iuhf != 0;
    }
    std::string error;
    if( molpro_uhf ) {
      error = "the FCIDUMP has IUHF=" + iuhf_value
              + " (MOLPRO-style block UHF), which is not an SBD layout";
    } else if( file_layout != build_layout ) {
      error = "the FCIDUMP layout is " + file_layout
              + (has_marker ? "" : " (no QCSC_INTEGRALS key)")
              + " but this binary reads " + build_layout;
    } else {
      const int max_index = (build_layout == "RESTRICTED") ? norb_header : 2 * norb_header;
      for(const auto & [value,i,j,k,l] : fcidump.integrals) {
        if( i < 0 || j < 0 || k < 0 || l < 0
            || i > max_index || j > max_index || k > max_index || l > max_index ) {
          error = "an integral index is out of range for the " + build_layout + " layout";
          break;
        }
      }
    }
    if( !error.empty() ) {
      if( mpi_rank == 0 ) {
        std::cerr << "[qcsc] ERROR: " << error << std::endl;
      }
      MPI_Abort(comm, 3);
    }
    if( mpi_rank == 0 ) {
      std::cout << "[qcsc] integrals layout: " << build_layout << std::endl;
      std::ofstream ofs_info("sbd_build_info.txt");
      ofs_info << "integrals=" << build_layout << std::endl;
    }
  }

  for(const auto & [key,value] : fcidump.header) {
    if( key == std::string("NORB") ) {
      L = std::atoi(value.c_str());
    }
    if( key == std::string("NELEC") ) {
      N = std::atoi(value.c_str());
    }
  }

  /**
     setup determinants for alpha and beta spin orbitals
   */
  sbd::det_vector<size_t, sbd::det_kind::half> adet;
  sbd::det_vector<size_t, sbd::det_kind::half> bdet;
  if( mpi_rank == 0 ) {
    sbd::LoadAlphaDets(adetfile,adet,sbd_data.bit_length,L);
    std::cout << "Loaded " << adet.size() << " alpha determinants." << std::endl;
    if( !bdetfile.empty() ) {
      // The file format is the same for both spins; only the reader's name says "alpha".
      sbd::LoadAlphaDets(bdetfile,bdet,sbd_data.bit_length,L);
      std::cout << "Loaded " << bdet.size() << " beta determinants." << std::endl;
    }
  }

  sbd::MpiBcast(adet,0,comm);
  if( !bdetfile.empty() ) {
    sbd::MpiBcast(bdet,0,comm);
  } else {
    bdet = adet;
  }

  /**
     sample-based diagonalization using data for fcidump, adet, bdet.
   */
  sbd::tpb::diag(comm,sbd_data,fcidump,adet,bdet,loadname,savename,
              energy,density,co_adet,co_bdet,one_p_rdm,two_p_rdm);

  if( mpi_rank == 0 ) {
    std::cout << "Davidson energy: " << energy << std::endl;
    std::ofstream ofs_energy("davidson_energy.txt");
    ofs_energy.precision(16);
    ofs_energy << energy << std::endl;
    ofs_energy.close();
    std::ofstream ofs_occa("occ_a.txt");
    ofs_occa.precision(16);
    std::ofstream ofs_occb("occ_b.txt");
    ofs_occb.precision(16);
    for(size_t i=0; i < density.size()/2; i++) {
      ofs_occa << density[2*i] << std::endl;
      ofs_occb << density[2*i + 1] << std::endl;
    }
    ofs_occa.close();
    ofs_occb.close();
    const size_t bytes_per_config = (L + 7) / 8;

    // Pack a carryover determinant list into the SBD binary format:
    // (L+7)/8 bytes per config, big-endian bit order, sbd::makestring index ordering.
    auto write_carryover = [&](const std::string & filename,
                               const sbd::det_vector<size_t, sbd::det_kind::half> & co_dets) {
      std::ofstream ofs(filename, std::ios::binary);
      std::vector<uint8_t> bytes(bytes_per_config);
      for (size_t i = 0; i < co_dets.size(); ++i) {
        std::fill(bytes.begin(), bytes.end(), 0);
        for (size_t j = 0; j < L; ++j) {
          size_t rev_idx = L - 1 - j;                 // sbd::makestring order
          size_t pw = rev_idx % sbd_data.bit_length;  // position in word
          size_t bw = rev_idx / sbd_data.bit_length;  // index of word
          bool bit = (co_dets[i][bw] >> pw) & 1ULL;
          size_t pb = 7 - (j % 8);                    // big-endian bit order
          size_t bb = j / 8;                          // index of byte
          bytes[bb] |= static_cast<uint8_t>(bit << pb);
        }
        ofs.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      }
      ofs.close();
    };

    std::cout << "Number of carryover determinants: " << co_adet.size() << std::endl;
    write_carryover("carryover.bin", co_adet);
    if( !bdetfile.empty() ) {
      // A separate beta set has its own carryover; write it alongside carryover.bin.
      std::cout << "Number of beta carryover determinants: " << co_bdet.size() << std::endl;
      write_carryover("carryover_b.bin", co_bdet);
    }
  }

  /**
     Finalize
  */

  MPI_Finalize();
  return 0;
}
