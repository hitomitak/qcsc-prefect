# pylint: disable=invalid-name
"""SKQD with random bit flips."""
import os
import argparse
import logging
from typing import Optional
import numpy as np
import h5py
import jax
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.sqd import sqd

LOG = logging.getLogger(__name__)


def main(
    filename: str,
    iexps: int | list[int],
    num_gen: int = 5,
    gpus: Optional[int | list[int]] = None,
    out_filename: Optional[str] = None
):
    try:
        iexps = list(iexps)
    except TypeError:
        iexps = [iexps]
    if gpus is not None:
        try:
            gpus = list(gpus)
        except TypeError:
            gpus = [gpus]

    with h5py.File(filename, 'r') as source:
        configuration = {}
        for key in source['configuration'].keys():
            record = source[f'configuration/{key}'][()]
            if isinstance(record, bytes):
                record = record.decode()
            configuration[key] = record

        num_plaq = source['data/num_plaq'][()]
        exp_plaq_data = np.unpackbits(source['data/exp_plaq_data'][()], axis=2)[..., :num_plaq]
        ref_plaq_data = np.unpackbits(source['data/ref_plaq_data'][()], axis=2)[..., :num_plaq]

    lattice = TriangularZ2Lattice(configuration['lattice'])
    dual_lattice = lattice.plaquette_dual()
    ising_hamiltonian = dual_lattice.make_hamiltonian(configuration['plaquette_energy'])

    mean_activation = np.mean(ref_plaq_data, axis=1)

    if not gpus or len(gpus) == 1:
        device_id = 0
    else:
        device_id = -1

    file_mode = 'w' if out_filename else 'r+'
    out_filename = out_filename or filename

    for iexp in iexps:
        LOG.info('Starting experiment %d', iexp)
        rng = np.random.default_rng(12345 + iexp)
        num_steps, shots, num_plaq = exp_plaq_data.shape  # pylint: disable=redefined-outer-name
        uniform = rng.random((num_steps, shots, num_gen, num_plaq))
        flips = np.asarray(uniform < mean_activation[:, None, None, :], dtype=np.uint8)
        states = np.concatenate([
            exp_plaq_data.reshape((-1, num_plaq)),
            (exp_plaq_data[:, :, None, :] ^ flips).reshape((-1, num_plaq))
        ], axis=0)[:, ::-1]

        energy, eigvec = sqd(ising_hamiltonian, states, jax_device_id=device_id,
                             return_states=False, return_hproj=False)

        groupname = f'skqd_rnd_{iexp}'
        with h5py.File(out_filename, file_mode, libver='latest') as out:
            try:
                del out[groupname]
            except KeyError:
                pass

            group = out.create_group(groupname)
            group.create_dataset('energy', data=energy)
            group.create_dataset('eigvec', data=eigvec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('iexp', type=int, nargs='+')
    parser.add_argument('--num-gen', type=int, default=5)
    parser.add_argument('--gpu', nargs='+')
    parser.add_argument('--out-filename')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(options.gpu)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'
    jax.config.update('jax_enable_x64', True)

    main(options.filename, options.iexp,
         num_gen=options.num_gen, gpus=options.gpu, out_filename=options.out_filename)
