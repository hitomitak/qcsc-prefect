"""Compute the approximate ground state through DMRG."""
import os
import tempfile
import logging
from typing import Optional
import h5py
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.ising_dmrg import ising_dmrg, get_mps_probs
from skqd_z2lgt.mwpm import minimum_weight_link_state
from skqd_z2lgt.parameters import Parameters

JULIA_BIN = ['julia', '--sysimage', '/opt/julia/iiyama/sysimages/sys_itensors.so']


def dmrg(parameters: Parameters, logger: Optional[logging.Logger] = None) -> float:
    """Run DMRG on the dual Ising hamiltonian."""
    logger = logger or logging.getLogger(__name__)

    with h5py.File(parameters.output_filename, 'r', libver='latest') as source:
        if (group := source.get('dmrg')) is not None:
            logger.info('DMRG result already exists in the output file.')
            return group['energy'][()]

    lattice = TriangularZ2Lattice(parameters.lgt.lattice)
    base_link_state = minimum_weight_link_state(parameters.lgt.charged_vertices, lattice)
    dual_lattice = lattice.plaquette_dual(base_link_state)
    ising_hamiltonian = dual_lattice.make_hamiltonian(parameters.lgt.plaquette_energy)

    logger.info('Invoking ITensorMPS DMRG function')
    with tempfile.NamedTemporaryFile() as tfile:
        filename = tfile.name
    dmrg_energy = ising_dmrg(ising_hamiltonian, filename=filename, julia_bin=JULIA_BIN)
    logger.info('Sampling the MPS for probability distribution over the computational basis')
    states, probs = get_mps_probs(filename, julia_bin=JULIA_BIN)
    os.unlink(filename)

    with h5py.File(parameters.output_filename, 'r+', libver='latest') as out:
        group = out.create_group('dmrg')
        group.create_dataset('energy', data=dmrg_energy)
        group.create_dataset('mps_states', data=states)
        group.create_dataset('mps_probs', data=probs)

    return dmrg_energy


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('parameters')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))

    with open(options.parameters, 'r', encoding='utf-8') as src:
        params = Parameters(**yaml.load(src, yaml.Loader))

    dmrg(params)
