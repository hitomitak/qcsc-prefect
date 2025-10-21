# pylint: disable=invalid-name
"""SKQD with no configuration recovery."""
import os
import argparse
import logging
import numpy as np
import h5py
import jax
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.sqd import sqd

LOG = logging.getLogger(__name__)


def main(filename: str, multi_gpu: bool = False):
    with h5py.File(filename, 'r') as source:
        configuration = dict(source.attrs)
        plaq_data = []
        for istep in range(configuration['num_steps']):
            dataset = source[f'data/plaq/exp_step{istep}']
            plaq_data.append(np.unpackbits(dataset[()], axis=1)[..., :dataset.attrs['num_bits']])

    dual_lattice = TriangularZ2Lattice(configuration['lattice']).plaquette_dual()
    hamiltonian = dual_lattice.make_hamiltonian(configuration['plaquette_energy'])

    states = np.concatenate(plaq_data, axis=0)[:, ::-1]
    energy, eigvec, sqd_states, ham_proj = sqd(hamiltonian, states,
                                               jax_device_id=-1 if multi_gpu else None)

    with h5py.File(filename, 'r+') as out:
        try:
            del out['skqd_raw']
        except KeyError:
            pass

        group = out.create_group('skqd_raw')
        dataset = group.create_dataset('sqd_states', data=np.packbits(sqd_states, axis=1))
        dataset.attrs['num_bits'] = dual_lattice.num_plaquettes
        group.create_dataset('energy', data=energy)
        group.create_dataset('eigvec', data=eigvec)
        subgroup = group.create_group('ham_proj')
        subgroup.create_dataset('data', data=ham_proj.data)
        subgroup.create_dataset('indices', data=ham_proj.indices)
        subgroup.create_dataset('indptr', data=ham_proj.indptr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--gpu', nargs='+')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(options.gpu)
    jax.config.update('jax_enable_x64', True)

    main(options.filename, multi_gpu=options.gpu and len(options.gpu) > 1)
