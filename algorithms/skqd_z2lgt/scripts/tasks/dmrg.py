"""Compute the approximate ground state through DMRG."""
import os
import argparse
import tempfile
import logging
import h5py
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.ising_dmrg import ising_dmrg, get_mps_probs

JULIA_BIN = ['julia', '--sysimage', '/opt/julia/iiyama/sysimages/sys_itensors.so']
LOG = logging.getLogger(__name__)


def main(filename: str):
    with h5py.File(filename, 'r', swmr=True) as source:
        configuration = dict(source.attrs)

    dual_lattice = TriangularZ2Lattice(configuration['lattice']).plaquette_dual()
    ising_hamiltonian = dual_lattice.make_hamiltonian(configuration['plaquette_energy'])

    LOG.info('Invoking ITensorMPS DMRG function')
    with tempfile.NamedTemporaryFile() as tfile:
        filename = tfile.name
    dmrg_energy = ising_dmrg(ising_hamiltonian, filename=filename, julia_bin=JULIA_BIN)
    LOG.info('Sampling the MPS for probability distribution over the computational basis')
    states, probs = get_mps_probs(filename, julia_bin=JULIA_BIN)
    os.unlink(filename)

    with h5py.File(filename, 'r+') as out:
        try:
            del out['dmrg']
        except KeyError:
            pass
        group = out.create_group('dmrg')
        group.create_dataset('energy', data=dmrg_energy)
        group.create_dataset('mps_states', data=states)
        group.create_dataset('mps_probs', data=probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))

    main(options.filename)
