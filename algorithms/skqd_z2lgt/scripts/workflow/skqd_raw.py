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
        configuration = {}
        for key in source['configuration'].keys():
            record = source[f'configuration/{key}'][()]
            if isinstance(record, bytes):
                record = record.decode()
            configuration[key] = record

        num_plaq = source['data/num_plaq'][()]
        exp_plaq_data = np.unpackbits(source['data/exp_plaq_data'][()], axis=2)[..., :num_plaq]

    dual_lattice = TriangularZ2Lattice(configuration['lattice']).plaquette_dual()
    ising_hamiltonian = dual_lattice.make_hamiltonian(configuration['plaquette_energy'])

    states = exp_plaq_data.reshape(-1, num_plaq)[:, ::-1]
    energy, eigvec, sqd_states, ham_proj = sqd(ising_hamiltonian, states,
                                               jax_device_id=-1 if multi_gpu else None)

    with h5py.File(filename, 'r+') as out:
        try:
            del out['skqd_raw']
        except KeyError:
            pass

        group = out.create_group('skqd_raw')
        group.create_dataset('num_plaq', data=num_plaq)
        group.create_dataset('sqd_states', data=np.packbits(sqd_states, axis=1))
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
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'
    jax.config.update('jax_enable_x64', True)

    main(options.filename, multi_gpu=options.gpu and len(options.gpu) > 1)
