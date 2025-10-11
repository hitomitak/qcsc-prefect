"""SKQD with random bit flips."""
import os
import argparse
import numpy as np
import h5py
import jax
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.sqd import sqd
from skqd_z2lgt.crbm import ConditionalRBM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--gpu')
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--gen-batch-size', type=int, default=10_000)
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
        num_vtx = source['data/num_vtx'][()]
        exp_plaq_data = np.unpackbits(source['data/exp_plaq_data'][()], axis=2)[..., :num_plaq]
        exp_vtx_data = np.unpackbits(source['data/exp_vtx_data'][()], axis=2)[..., :num_vtx]

        models = [ConditionalRBM.load(source[f'crbm_step{istep}'])
                  for istep in range(configuration['num_steps'])]

    lattice = TriangularZ2Lattice(configuration['lattice'])
    dual_lattice = lattice.plaquette_dual()
    ising_hamiltonian = dual_lattice.make_hamiltonian(configuration['plaquette_energy'])

    num_batches = int(np.ceil(exp_vtx_data.shape[1] / options.gen_batch_size).astype(int))
    gen_shape = (configuration['shots'], options.num, num_plaq)

    gen_data = []
    for istep, model in enumerate(models):
        data = np.empty(gen_shape, dtype=np.uint8)
        for ibatch in range(num_batches):
            start = ibatch * options.gen_batch_size
            end = start + options.gen_batch_size
            sample = model.sample(exp_vtx_data[istep, start:end], size=options.num)
            flips = sample.transpose((1, 0, 2))
            data[start:end] = exp_plaq_data[istep, start:end, None, :] ^ flips
        data = data.reshape((-1, num_plaq))
        gen_data.append(data)

    states = np.concatenate([exp_plaq_data.reshape((-1, num_plaq))] + gen_data, axis=0)[:, ::-1]
    sqd_states, ham_proj, energy, _ = sqd(ising_hamiltonian, states)

    groupname = 'skqd_rcv'  # pylint: disable=invalid-name
    with h5py.File(options.filename, 'r+') as out:
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
