"""Compute the approximate ground state through DMRG."""
import os
from collections.abc import Callable
import tempfile
import logging
from pathlib import Path
from typing import Optional
import h5py
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.ising_dmrg import ising_dmrg, get_mps_probs
from skqd_z2lgt.mwpm import minimum_weight_link_state
from skqd_z2lgt.parameters import Parameters


def dmrg_flow(
    parameters: Parameters,
    dmrg_fn: Callable,
    logger: Optional[logging.Logger] = None
) -> float:
    """Run DMRG on the dual Ising hamiltonian."""
    logger = logger or logging.getLogger(__name__)

    path = Path(parameters.pkgpath) / 'dmrg.h5'
    if os.path.exists(path):
        logger.info('DMRG result already exists in the output file.')
        with h5py.File(path, 'r', libver='latest') as source:
            return source['energy'][()]

    lattice = TriangularZ2Lattice(parameters.lgt.lattice)
    base_link_state = minimum_weight_link_state(parameters.lgt.charged_vertices, lattice)
    dual_lattice = lattice.plaquette_dual(base_link_state)
    ising_hamiltonian = dual_lattice.make_hamiltonian(parameters.lgt.plaquette_energy)

    logger.info('Invoking ITensorMPS DMRG function and sampling the MPS')
    energy, states, probs = dmrg_fn(ising_hamiltonian)

    with h5py.File(path, 'w', libver='latest') as out:
        out.create_dataset('energy', data=energy)
        out.create_dataset('mps_states', data=states)
        out.create_dataset('mps_probs', data=probs)

    return energy


def dmrg(parameters: Parameters, logger: Optional[logging.Logger] = None) -> float:
    dmrg_params = parameters.dmrg
    julia_bin = 'julia'
    if dmrg_params.julia_sysimage:
        julia_bin = ['julia', '--sysimage', dmrg_params.julia_sysimage]

    def dmrg_fn(hamiltonian):
        with tempfile.NamedTemporaryFile() as tfile:
            filename = tfile.name

        energy = ising_dmrg(hamiltonian, filename=filename,
                            nsweeps=dmrg_params.nsweeps, maxdim=dmrg_params.maxdim,
                            cutoff=dmrg_params.cutoff, julia_bin=julia_bin)
        states, probs = get_mps_probs(filename, num_samples=dmrg_params.num_samples,
                                      julia_bin=julia_bin)
        os.unlink(filename)
        return energy, states, probs

    return dmrg_flow(parameters, dmrg_fn, logger)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))

    with open(options.parameters, 'r', encoding='utf-8') as src:
        params = Parameters.model_validate_json(src.read())

    dmrg(params)
