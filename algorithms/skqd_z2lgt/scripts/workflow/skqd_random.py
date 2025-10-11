"""SKQD with random bit flips."""
import os
import argparse
import numpy as np
import h5py
import jax
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.sqd import sqd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('iexp', type=int)
    parser.add_argument('--gpu')
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--out')
    options = parser.parse_args()

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    with h5py.File(options.filename, 'r') as source:
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

    rng = np.random.default_rng(12345 + options.iexp)
    num_steps, shots, num_plaq = exp_plaq_data.shape  # pylint: disable=redefined-outer-name
    uniform = rng.random((num_steps, shots, options.num, num_plaq))
    flips = np.asarray(uniform < mean_activation[:, None, None, :], dtype=np.uint8)
    states = np.concatenate([
        exp_plaq_data.reshape((-1, num_plaq)),
        (exp_plaq_data[:, :, None, :] ^ flips).reshape((-1, num_plaq))
    ], axis=0)[:, ::-1]
    sqd_states, ham_proj, energy, _ = sqd(ising_hamiltonian, states)

    out_filename = options.out or options.filename
    groupname = f'skqd_rnd_{options.iexp}'  # pylint: disable=invalid-name
    with h5py.File(out_filename, 'w-' if options.out else 'r+', libver='latest') as out:
        try:
            del out[groupname]
        except KeyError:
            pass

        group = out.create_group(groupname)
        group.create_dataset('num_plaq', data=num_plaq)
        group.create_dataset('sqd_states', data=np.packbits(sqd_states, axis=1))
        group.create_dataset('energy', data=energy)
        subgroup = group.create_group('ham_proj')
        subgroup.create_dataset('data', data=ham_proj.data)
        subgroup.create_dataset('indices', data=ham_proj.indices)
        subgroup.create_dataset('indptr', data=ham_proj.indptr)
